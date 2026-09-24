"""
fusion.py

Stage 3: trust-weighted fusion controller.

Reuses detector.py (unchanged -- for cusum_stat and the calibrated
threshold h) and dead_reckoning.py (unchanged -- for the IMU/ATTITUDE
loading and the gravity-compensated strapdown integration primitives)
rather than reimplementing either. This script only adds the fusion
logic on top:

  1. trust_weight = clip(1 - cusum_stat/h, 0, 1) -- a continuous
     decay from "fully trust GPS" (near 1, healthy) toward "fully
     trust dead-reckoning" (near 0, as CUSUM approaches its alarm
     threshold h) instead of detector.py's binary latched flag -- but
     latched (via latch_trust_decay()) so it can only fall, never climb
     back up, once detector.py's predicted_spoofed first fires, and
     floored at TRUST_FLOOR rather than 0 once latched, so a trial
     whose DR trajectory eventually drifts past GPS-only's error late
     in a long flight doesn't fully commit to DR forever. Both the
     latch and the floor value are justified from data in
     latch_trust_decay()'s and TRUST_FLOOR's comments, not asserted.

  2. Two fused trajectories per trial, computed over the SAME GPS-rate
     time grid detector.py already aligned everything to:

       - trust-weighted: fused = trust_weight*gps + (1-trust_weight)*dr,
         where dr is NOT Stage 2's raw dead_reckoning_output.csv (which
         drifts unboundedly for the whole flight, per Stage 2's own
         numbers) but a SEPARATE reintegration of the same IMU stream
         that periodically re-anchors (resets position AND velocity
         state) to the current GPS fix whenever trust_weight is high
         (> REANCHOR_TRUST_THRESHOLD). This is what stops ambient DR
         drift from contaminating the fused estimate while GPS is
         healthy -- see dead_reckon_reanchored() below.

       - hard-switch baseline (the ablation): fused = gps when
         predicted_spoofed==0, else Stage 2's raw (never-reanchored)
         dr_x/dr_y. No blending, no reanchoring -- this is deliberately
         the naive thing trust-weighted fusion is being compared
         against, so it must NOT get the reanchoring trick.

Usage:
    python3 fusion.py
Runs all 6 trials already on disk (flight_log_*.csv, spoofed_gps_*.csv,
dead_reckoning_output_*.csv, detector_output_*.csv) and writes:
    fusion_output_<trial>.csv           (trust-weighted trajectory)
    fusion_hardswitch_output_<trial>.csv (hard-switch trajectory)
    fusion_results_summary.csv          (one row per trial, comparison)
"""

import numpy as np
import pandas as pd

import detector
import dead_reckoning as drmod


TRIALS = [
    (vp, ap)
    for vp in ("gentle", "moderate", "aggressive", "circular", "figure_eight", "sustained_cruise")
    for ap in ("slow_drift", "fast_drift", "sudden_jump", "intermittent")
]

# Re-anchor DR's internal state to GPS whenever trust_weight exceeds this.
REANCHOR_TRUST_THRESHOLD = 0.9

# Half-width of the window (seconds) around the predicted_spoofed 0->1
# flip within which we look for the largest single-timestep position
# jump, for both fusion methods.
TRANSITION_WINDOW_S = 2.0

# Floor on trust_weight once latched (see latch_trust_decay()). Derived
# empirically, not guessed: sweeping this floor from 0.00 to 0.30 in
# 0.01 steps against the 6 trials already on disk (fresh run this
# session) shows moderate_slow_drift is the only trial where the
# floor=0 latch (fully committing to DR once alarmed) makes
# trust-weighted's MEAN error worse than GPS-only's -- because that
# trial's DR trajectory grows past GPS-only's error by ~1.9x by the end
# of the flight (all other 5 trials keep DR at or below GPS-only error
# for the whole post-alarm tail, so they don't need a floor at all).
# moderate_slow_drift's trust-weighted mean crosses under its own
# gps-only mean (5.9698m) between floor=0.11 (6.0057m, still loses) and
# floor=0.12 (5.8931m, wins) -- see the sweep in this session. 0.15
# gives clear margin above that measured crossover while the two
# trials that get monotonically WORSE with any floor (aggressive_slow,
# aggressive_fast -- their DR is already better than GPS the whole
# tail, so blending in floor-weighted spoofed GPS only hurts) still
# beat GPS-only by 2-3x at this value, so 0.15 isn't pushed needlessly
# high just to pad one trial's margin.
TRUST_FLOOR = 0.15

# --- Bounded-offset recovery window (added after the 24-trial expanded
# matrix run exposed two regressions the original 6-trial validation
# never surfaced: trust-weighted fusion beat GPS-only in only 8/24
# trials there, specifically failing on sudden_jump -- 6/6 -- and
# intermittent -- most of 6/6). See latch_trust_decay()'s docstring for
# the full mechanism and the data that justifies these two constants.
#
# JUMP_MODE_TRUST: the trust_weight applied during a bounded-offset
# recovery window. Set just above REANCHOR_TRUST_THRESHOLD (0.9) so
# dead_reckon_reanchored()'s EXISTING re-anchor mechanism activates
# during recovery -- reusing that machinery rather than adding a new
# reset path -- which is what actually caps sudden_jump's unbounded
# DR free-drift (the root mechanism behind its regression: GPS-only
# error stays bounded at the fixed jump offset, ~3m observed across
# the 6 sudden_jump trials, but never-reanchored DR grows unboundedly
# for the rest of the flight once latched). Not 1.0 -- keeps the same
# "never fully commit" principle TRUST_FLOOR encodes, just inverted
# toward GPS instead of DR for this specific, verified-bounded regime.
#
# Tested and rejected: capping this below REANCHOR_TRUST_THRESHOLD
# (tried 0.70, deliberately avoiding any re-anchor at all) removes the
# regression risk below but also mostly erases the intermittent win --
# e.g. aggressive_intermittent's trust-weighted mean only improved to
# 26.46m (vs GPS-only 5.29m, still a clear loss) without re-anchoring,
# versus 2.17m (a clear win) with it -- because re-anchoring to a GPS
# fix that is GENUINELY correct (intermittent's off-blocks) is what
# actually stops further DR drift; blending alone barely helps. 0.92
# is kept despite the known moderate_sudden_jump regression this
# causes (see latch_trust_decay()'s docstring and the Part A report)
# because the net effect across all 24 trials is clearly better with
# re-anchoring enabled than without it -- this is a real, measured
# trade-off, not an oversight.
JUMP_MODE_TRUST = 0.92

# RECOVERY_HOLD_S: how long a bounded-offset recovery window stays open
# after a raw jump_delta_m spike (crossing detector.py's own calibrated
# jump_threshold), before lapsing back to the standard floor-latch
# behavior if no further spike renews it. Derived from data, not
# guessed: measured the actual spacing between consecutive raw
# jump_delta_m threshold-crossings in all 6 intermittent trials on disk
# (detector.py's jump-detector path fires a fresh spike at EVERY
# on/off-block transition -- both intermittent's 8s-on/4s-off cadence
# show up as spikes) -- the gaps cluster at exactly 4.0s and 8.0s
# (e.g. aggressive_intermittent: 39.9, 43.9, 51.89, 55.89, ... gaps
# [4.0, 7.99, 4.0, 8.01, ...]; identical pattern in all 6 profiles,
# max observed gap 8.03s). RECOVERY_HOLD_S=10.0 sits above that worst
# observed gap (~1.25x margin, in line with this file's and
# detector.py's existing margin conventions) so consecutive transition
# spikes keep re-arming the window continuously through intermittent's
# whole on/off cycle, rather than dropping and reacquiring every block.
RECOVERY_HOLD_S = 10.0


def nearest_indices(sorted_times, query_times):
    """
    Vectorized nearest-neighbor lookup, same semantics as the scalar
    nearest_index() used throughout detector.py/dead_reckoning.py
    (next-or-equal from searchsorted alone is wrong whenever the
    previous sample is actually closer -- this checks both candidates).
    """
    sorted_times = np.asarray(sorted_times)
    query_times = np.asarray(query_times)
    n = len(sorted_times)
    idx = np.searchsorted(sorted_times, query_times)
    idx_after = np.minimum(idx, n - 1)
    idx_before = np.maximum(idx - 1, 0)
    after_t = sorted_times[idx_after]
    before_t = sorted_times[idx_before]
    choose_after = np.abs(after_t - query_times) < np.abs(before_t - query_times)
    result = np.where(choose_after, idx_after, idx_before)
    result = np.where(idx <= 0, 0, result)
    result = np.where(idx >= n, n - 1, result)
    return result


def latch_trust_decay(raw_trust, predicted_spoofed, cusum_stat, h, jump_delta, jump_threshold, t,
                       trust_floor=TRUST_FLOOR, jump_mode_trust=JUMP_MODE_TRUST,
                       recovery_hold_s=RECOVERY_HOLD_S):
    """
    Ratchets raw_trust (the smooth, cusum_stat-derived value) so it can
    never climb back up once detector.py's predicted_spoofed has fired,
    and never decays below `trust_floor` once latched -- EXCEPT during a
    bounded-offset recovery window (see below), which is the one case
    this version allows trust to rise back above the floor.

    ADDED after the 24-trial expanded matrix run: trust-weighted fusion
    beat GPS-only mean error in only 8/24 trials there (down from 6/6 on
    the original 6 gradual-attack trials), because the pure one-way
    floor-latch below was validated ONLY against sustained gradual
    attacks, where "once alarmed, keep distrusting GPS" is correct
    (GPS-only error keeps growing for the rest of the flight). Two
    attack types this project added later break that assumption:

      - sudden_jump: GPS-only error is BOUNDED after the jump (stays
        near the fixed offset, ~3m observed across all 6 sudden_jump
        trials) while raw, never-reanchored DR grows UNBOUNDED for the
        rest of the flight once latched -- so the floor-latch commits
        to the trajectory that gets WORSE, not better, over time.
      - intermittent: confirmed directly (fusion_output_aggressive_
        intermittent.csv, prior session): trust drops to the floor at
        the first on-block and never recovers even though is_spoofed
        genuinely reverts to 0 for multi-second windows afterward.

    Recovery mechanism, and why it's restricted this narrowly (multiple
    broader designs were tried and rejected against this data before
    landing here -- see the fusion.py.bak_partA-era exploration notes
    in the project history; the measurements that ruled them out are
    load-bearing, not skipped for brevity):

      1. Naively recovering trust whenever slope_gap/cusum_stat drops
         back near its calibrated baseline is UNSAFE and was measured,
         not assumed, to be unsafe: aggressive_slow_drift's cusum_stat
         sits at exactly 0 for up to 40.29s DURING a still-ongoing
         gradual attack (its long trailing window re-baselining around
         the new, still-spoofed rate -- the exact phenomenon that
         motivated the original one-way latch in the first place, see
         below), and sustained_cruise_slow_drift shows the same up to
         38.14s. Any recovery window long enough to be safe against
         that (>40s) is also too long to ever fire within
         intermittent's actual off-block-to-off-block cycle (measured
         max useful zero-run there: 15.82s, several profiles under 5s)
         -- so a pure "cusum_stat near baseline for N seconds" design
         cannot separate "attack genuinely stopped" from "gradual
         attack's own known re-baselining artifact" at ANY window
         length. This is why raw_trust/cusum_stat alone, no matter how
         it's windowed, is not used as the recovery trigger below.

      2. What DOES separate cleanly, verified against all 24 trials:
         raw jump_delta_m (detector.py's un-latched, per-sample,
         vector-diff jump signal) crossing its OWN calibrated
         jump_threshold. Measured 0 occurrences (out of 11 of the 12
         gradual trials; 1 isolated sample in the 12th,
         moderate_fast_drift) across every slow_drift/fast_drift trial
         -- gradual attacks are gradual BY CONSTRUCTION (spoof_injector.py
         walks the fake position smoothly, no per-sample jumps), so
         this signal structurally almost never fires there, unlike
         slope_gap/cusum_stat which the attack's own growth naturally
         drives up. It fires in all 6 sudden_jump trials (that's its
         original purpose) AND at every intermittent on/off-block
         transition (confirmed: each transition snaps the spoofed
         position abruptly toward or away from truth, which is itself
         a jump_delta_m spike -- see RECOVERY_HOLD_S's derivation).

      3. So: a jump_delta_m spike, while cusum_stat is NOT concurrently
         above h (an actively escalating gradual-style attack is a
         stronger, overriding signal, checked fresh every sample, not
         latched -- so if a jump-triggered trial's CUSUM starts
         climbing on top of it, e.g. because the offset compounds
         with the always-present ambient DR-drift signal late in a
         long flight, recovery mode stands down immediately rather
         than staying stuck open), opens a bounded window of
         `jump_mode_trust` (not full 1.0 -- see JUMP_MODE_TRUST's
         comment) for `recovery_hold_s` seconds, renewable by another
         spike. This directly targets both regressions: sudden_jump's
         initial jump IS this spike (with no subsequent cusum growth
         for a fixed offset, so the window stays open, capping DR's
         free-drift via dead_reckon_reanchored's existing re-anchor
         logic once trust exceeds REANCHOR_TRUST_THRESHOLD); each
         intermittent on/off transition re-arms the same window, so
         (as long as cusum_stat hasn't independently crossed h from
         that cycle's own on-block growth) trust gets a real chance to
         recover during the genuinely-clean off-blocks instead of
         staying pinned at TRUST_FLOOR for the whole rest of the
         flight.

      4. This is a strictly ADDITIVE capability on top of the original
         ratchet, not a replacement -- trust is
         max(ratchet_floor, recovery_value), so every trial that never
         produces a qualifying jump_delta_m spike (all 6 original
         gradual trials, plus most of any future gradual-only trial)
         runs through byte-identical logic to the pre-this-change
         version. This is verified in the per-trial before/after
         numbers this change was validated against, not just argued.

    The rest of this docstring (below) describes the base ratchet,
    unchanged from the original design:

    Why the latch was needed: predicted_spoofed itself is a genuine
    one-way latch (detector.py's `alarmed` bool is only ever set True,
    never reset -- verified directly in detect()'s loop). But raw_trust
    is built from the raw, UNlatched cusum_stat, and for a sustained
    constant-rate attack, cusum_stat decays back toward 0 once the
    detector's long window re-baselines around the new (still-spoofed)
    rate -- confirmed directly in moderate_slow_drift/aggressive_slow_drift
    (cusum_stat hits exactly 0 well before the flight ends while
    is_spoofed stays 1 for the rest of the flight). Following raw_trust
    naively there drags fusion back to pure GPS mid-attack.

    Why the floor was ALSO needed on top of the latch: a pure latch
    (trust_floor=0) commits 100% to DR for the rest of the flight once
    alarmed. DR error grows roughly quadratically with elapsed time
    (per dead_reckoning.py's own residual specific-force bias
    analysis), while GPS-only error under these constant-rate attacks
    only grows linearly -- so for a long enough post-alarm window, DR
    eventually gets WORSE than GPS-only. Measured directly on
    moderate_slow_drift: with trust_floor=0, DR ends the flight at
    ~1.9x GPS-only's error, and trust-weighted's mean error (7.274m)
    ends up worse than GPS-only's (5.970m). A nonzero floor keeps a
    minimum blend of GPS in the fused estimate even while latched,
    capping how far a bad-post-alarm-DR trial can diverge. See
    TRUST_FLOOR's comment for exactly how that value was chosen.

    Before the first predicted_spoofed==1 sample, trust follows
    raw_trust exactly (unchanged pre-attack behavior, already
    validated -- trust_floor plays no role here). From the first
    predicted_spoofed==1 sample onward, trust is
    max(trust_floor, running_minimum(raw_trust)) -- it can keep falling
    (down to trust_floor) if cusum_stat climbs further, but can never
    rise back toward 1, even if cusum_stat (or, hypothetically,
    predicted_spoofed itself) later drops back down. That "even if
    predicted_spoofed reverts" choice is deliberate: detector.py's own
    docstring reasoning -- "a security alarm should not silently
    self-clear" -- applies just as much to the trust signal fusion is
    built on, so once an alarm has fired at all in this trial, trust
    is latched low (down to the floor) for the remainder of the flight
    rather than re-arming. (predicted_spoofed can't actually revert to
    0 with the current detector.py, since its own latch is global too,
    so this only matters if that ever changes.)

    Whether trust_weight should recover toward 1 if predicted_spoofed
    ever DID revert to 0 (rather than stay latched low): deliberately
    staying latched low, for the same alarm-should-not-self-clear
    reasoning above -- an attacker capable of making the detector
    briefly de-alarm mid-attack (e.g. by matching the ambient rate just
    long enough to reset a recovering trust signal) shouldn't be able
    to walk fusion's trust back up by doing so; that would just move
    the re-baselining exploit from cusum_stat down into trust_weight
    instead of removing it. NOTE this reasoning is now only PARTIALLY
    upheld by the recovery mechanism above: it is deliberately gated on
    a structurally different, harder-to-imitate signal (a genuine
    jump_delta_m spike, not just a quiet cusum_stat) precisely to avoid
    reopening this exploit via the easy route -- but an adversary who
    injects small abrupt back-and-forth position snaps specifically to
    keep re-arming recovery windows while drifting slowly underneath is
    a residual risk this design does not close, and is not evaluated
    against here (this project's attack set doesn't include an
    adaptive attacker of that shape); flagged as a real caveat, not
    fixed.

    The transition into the latch stays smooth (no new discontinuity):
    predicted_spoofed only flips to 1 once cusum_stat > h, at which
    point raw_trust = clip(1 - cusum_stat/h, 0, 1) is already clipped
    to 0 -- so the ratchet's pre-floor value at that instant is exactly
    the smooth, already-validated decay curve's value, and trust only
    steps up to trust_floor if that value is below it.
    """
    n = len(raw_trust)
    trust = np.empty(n)
    latched = False
    floor = 1.0
    recovery_until_t = -np.inf
    for i in range(n):
        if not latched and predicted_spoofed[i] == 1:
            latched = True
            prev = trust[i - 1] if i > 0 else raw_trust[i]
            floor = max(trust_floor, min(prev, raw_trust[i]))
        elif latched:
            floor = max(trust_floor, min(floor, raw_trust[i]))
        base_trust = floor if latched else raw_trust[i]

        # --- bounded-offset recovery window (see docstring above) ---
        # Arming (a jump_delta_m spike) and granting (cusum not
        # currently escalating) are deliberately checked independently,
        # not both at the same sample: a genuine instantaneous jump
        # typically spikes cusum_stat's raw statistic too, in the SAME
        # single step (one big residual is enough to push the
        # cumulative sum over h immediately, even though CUSUM's
        # sustained-evidence design doesn't structurally "catch" a
        # jump the way it catches a ramp -- see detector.py's own
        # docstring on this). Requiring not-escalating AT the spike
        # sample itself was tried first and measured to never arm at
        # all for several sudden_jump trials (jump_delta_m is a
        # one-sample-diff signal, already back to ~0 again a few
        # samples later exactly when cusum_escalating finally clears) --
        # decoupling them so the spike arms the window regardless, and
        # cusum_escalating only gates whether elevated trust is
        # GRANTED on each subsequent sample, is what actually lets the
        # window open.
        cusum_escalating = cusum_stat[i] > h
        if latched and jump_delta[i] > jump_threshold:
            recovery_until_t = t[i] + recovery_hold_s
        if latched and not cusum_escalating and t[i] <= recovery_until_t:
            trust[i] = max(base_trust, jump_mode_trust)
        else:
            trust[i] = base_trust
    return trust


def compute_trust_weight(spoofed_path, dr_path):
    """
    Re-runs detector.py's own load_and_align()/detect() (unmodified) to
    get cusum_stat plus the calibrated threshold h from df.attrs, which
    detector_output_*.csv does not persist. Returns the aligned df (same
    content as detector_output_*.csv) plus the derived trust_weight and h.
    """
    df = detector.load_and_align(spoofed_path, dr_path)
    df = detector.detect(df)
    h = df.attrs.get("cusum_h")
    jump_threshold = df.attrs.get("jump_threshold_m")
    if not h or h <= 0:
        # No calibration was possible (see detector.py's WARNING branch) --
        # cusum_stat is all zeros / predicted_spoofed all zeros there, so
        # "fully trust GPS" is the consistent trust_weight in that case.
        raw_trust = np.ones(len(df))
    else:
        raw_trust = np.clip(1.0 - df["cusum_stat"].values / h, 0.0, 1.0)
    trust_weight = latch_trust_decay(
        raw_trust, df["predicted_spoofed"].values,
        df["cusum_stat"].values, h if h else np.inf,
        df["jump_delta_m"].values, jump_threshold if jump_threshold else np.inf,
        df["t"].values,
    )
    return df, trust_weight, h


def dead_reckon_reanchored(imu, gps_t, gps_x, gps_y, trust_t, trust_w, threshold):
    """
    Same strapdown integration as dead_reckoning.dead_reckon(), but at
    every IMU step, if the trust_weight nearest that timestamp is above
    `threshold`, the position AND velocity state are reset to match the
    GPS fix nearest that timestamp. This is what actually runs the
    re-anchoring during the fusion loop (not just a comment) -- it is
    what keeps this DR trajectory from accumulating the ambient drift
    Stage 2 documented while GPS is healthy.
    """
    n = len(imu)
    x = np.zeros(n)
    y = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)
    g_ned = np.array([0.0, 0.0, drmod.GRAVITY_MSS])

    imu_t = imu["t"].values
    xacc = imu["xacc"].values
    yacc = imu["yacc"].values
    zacc = imu["zacc"].values
    roll = imu["att_roll"].values
    pitch = imu["att_pitch"].values
    yaw = imu["att_yaw"].values

    reset_gps_idx = nearest_indices(gps_t, imu_t)
    reset_trust_idx = nearest_indices(trust_t, imu_t)
    trust_at_imu = trust_w[reset_trust_idx]
    should_reset = trust_at_imu > threshold

    for i in range(1, n):
        dt = imu_t[i] - imu_t[i - 1]
        if dt <= 0 or dt > 1.0:
            x[i], y[i] = x[i - 1], y[i - 1]
            vx[i], vy[i] = vx[i - 1], vy[i - 1]
        else:
            f_body = np.array([
                xacc[i] * drmod.ACCEL_SCALE,
                yacc[i] * drmod.ACCEL_SCALE,
                zacc[i] * drmod.ACCEL_SCALE,
            ])
            R = drmod.body_to_ned(roll[i], pitch[i], yaw[i])
            a_ned = R @ f_body + g_ned
            a_north, a_east = a_ned[0], a_ned[1]

            vx[i] = vx[i - 1] + a_east * dt
            vy[i] = vy[i - 1] + a_north * dt
            x[i] = x[i - 1] + vx[i] * dt
            y[i] = y[i - 1] + vy[i] * dt

        if should_reset[i]:
            k = reset_gps_idx[i]
            x[i] = gps_x[k]
            y[i] = gps_y[k]
            vx[i] = 0.0
            vy[i] = 0.0

    return x, y


def largest_transition_jump(t, fused_x, fused_y, predicted_spoofed):
    """
    Largest single-timestep displacement in the fused trajectory within
    +/-TRANSITION_WINDOW_S seconds of the first predicted_spoofed 0->1
    flip. Returns None if there is no flip in this trial.
    """
    flips = np.where((predicted_spoofed[:-1] == 0) & (predicted_spoofed[1:] == 1))[0]
    if len(flips) == 0:
        return None
    flip_t = t[flips[0] + 1]

    jumps = np.hypot(np.diff(fused_x), np.diff(fused_y))
    jump_t = t[1:]  # timestamp of the sample landed ON after each jump
    mask = (jump_t >= flip_t - TRANSITION_WINDOW_S) & (jump_t <= flip_t + TRANSITION_WINDOW_S)
    if not np.any(mask):
        return None
    return float(jumps[mask].max())


def run_trial(vp, ap):
    trial = f"{vp}_{ap}"
    flight_log_path = f"flight_log_{trial}.csv"
    spoofed_path = f"spoofed_gps_{trial}.csv"
    dr_path = f"dead_reckoning_output_{trial}.csv"
    det_out_path = f"detector_output_{trial}.csv"

    det_df, trust_weight, h = compute_trust_weight(spoofed_path, dr_path)

    saved = pd.read_csv(det_out_path)
    if not np.allclose(det_df["cusum_stat"].values, saved["cusum_stat"].values):
        raise RuntimeError(
            f"{trial}: recomputed cusum_stat doesn't match saved "
            f"detector_output_{trial}.csv -- detector.py must have changed "
            f"since that file was generated."
        )

    flight = pd.read_csv(flight_log_path)
    imu = drmod.load_imu(flight)
    att = drmod.load_attitude(flight)
    imu = drmod.attach_attitude(imu, att)

    gps_t = det_df["t"].values
    gps_x = det_df["spoofed_x_m"].values
    gps_y = det_df["spoofed_y_m"].values
    true_x = det_df["true_x_m"].values
    true_y = det_df["true_y_m"].values
    is_spoofed = det_df["is_spoofed"].values
    predicted_spoofed = det_df["predicted_spoofed"].values
    raw_dr_x = det_df["dr_x"].values   # Stage 2's raw, never-reanchored DR
    raw_dr_y = det_df["dr_y"].values

    reanchored_x, reanchored_y = dead_reckon_reanchored(
        imu, gps_t, gps_x, gps_y, gps_t, trust_weight, REANCHOR_TRUST_THRESHOLD
    )
    imu_t = imu["t"].values
    match_idx = nearest_indices(imu_t, gps_t)
    dr_re_x = reanchored_x[match_idx]
    dr_re_y = reanchored_y[match_idx]

    # --- trust-weighted fusion ---
    fused_tw_x = trust_weight * gps_x + (1 - trust_weight) * dr_re_x
    fused_tw_y = trust_weight * gps_y + (1 - trust_weight) * dr_re_y

    # --- naive hard-switch baseline ---
    fused_hs_x = np.where(predicted_spoofed == 0, gps_x, raw_dr_x)
    fused_hs_y = np.where(predicted_spoofed == 0, gps_y, raw_dr_y)

    pd.DataFrame({
        "t": gps_t, "gps_x": gps_x, "gps_y": gps_y,
        "dr_x": dr_re_x, "dr_y": dr_re_y,
        "fused_x": fused_tw_x, "fused_y": fused_tw_y,
        "trust_weight": trust_weight,
        "is_spoofed": is_spoofed,
    }).to_csv(f"fusion_output_{trial}.csv", index=False)

    pd.DataFrame({
        "t": gps_t, "gps_x": gps_x, "gps_y": gps_y,
        "dr_x": raw_dr_x, "dr_y": raw_dr_y,
        "fused_x": fused_hs_x, "fused_y": fused_hs_y,
        "predicted_spoofed": predicted_spoofed,
        "is_spoofed": is_spoofed,
    }).to_csv(f"fusion_hardswitch_output_{trial}.csv", index=False)

    # --- error metrics vs ground truth ---
    err_tw = np.hypot(fused_tw_x - true_x, fused_tw_y - true_y)
    err_hs = np.hypot(fused_hs_x - true_x, fused_hs_y - true_y)
    err_gps = np.hypot(gps_x - true_x, gps_y - true_y)
    err_dr = np.hypot(raw_dr_x - true_x, raw_dr_y - true_y)

    pre_mask = is_spoofed == 0
    pre_tw_mean = float(err_tw[pre_mask].mean()) if pre_mask.any() else float("nan")
    pre_gps_mean = float(err_gps[pre_mask].mean()) if pre_mask.any() else float("nan")

    tw_jump = largest_transition_jump(gps_t, fused_tw_x, fused_tw_y, predicted_spoofed)
    hs_jump = largest_transition_jump(gps_t, fused_hs_x, fused_hs_y, predicted_spoofed)

    attack_start_idx = np.argmax(is_spoofed == 1) if np.any(is_spoofed == 1) else None
    attack_start_t = float(gps_t[attack_start_idx] - gps_t[0]) if attack_start_idx is not None else None

    return {
        "velocity_profile": vp,
        "attack_profile": ap,
        "n_samples": len(gps_t),
        "cusum_h": h,
        "attack_start_t_s": attack_start_t,
        "trust_weighted_error_mean": float(err_tw.mean()),
        "trust_weighted_error_max": float(err_tw.max()),
        "trust_weighted_error_final": float(err_tw[-1]),
        "hard_switch_error_mean": float(err_hs.mean()),
        "hard_switch_error_max": float(err_hs.max()),
        "hard_switch_error_final": float(err_hs[-1]),
        "gps_only_error_mean": float(err_gps.mean()),
        "gps_only_error_max": float(err_gps.max()),
        "gps_only_error_final": float(err_gps[-1]),
        "dr_only_error_mean": float(err_dr.mean()),
        "dr_only_error_max": float(err_dr.max()),
        "dr_only_error_final": float(err_dr[-1]),
        "trust_weighted_max_jump_m": tw_jump,
        "hard_switch_max_jump_m": hs_jump,
        "pre_attack_trust_weighted_error_mean": pre_tw_mean,
        "pre_attack_gps_only_error_mean": pre_gps_mean,
        "pre_attack_reanchoring_delta_m": pre_tw_mean - pre_gps_mean,
    }


def main():
    rows = []
    for vp, ap in TRIALS:
        trial = f"{vp}_{ap}"
        print(f"=== {trial} ===")
        row = run_trial(vp, ap)
        rows.append(row)
        print(f"  trust-weighted: mean={row['trust_weighted_error_mean']:.3f}m "
              f"max={row['trust_weighted_error_max']:.3f}m "
              f"final={row['trust_weighted_error_final']:.3f}m "
              f"max_jump={row['trust_weighted_max_jump_m']}")
        print(f"  hard-switch:    mean={row['hard_switch_error_mean']:.3f}m "
              f"max={row['hard_switch_error_max']:.3f}m "
              f"final={row['hard_switch_error_final']:.3f}m "
              f"max_jump={row['hard_switch_max_jump_m']}")
        print(f"  gps-only:       mean={row['gps_only_error_mean']:.3f}m")
        print(f"  dr-only:        mean={row['dr_only_error_mean']:.3f}m")
        print(f"  pre-attack check: trust-weighted mean={row['pre_attack_trust_weighted_error_mean']:.4f}m "
              f"vs gps-only mean={row['pre_attack_gps_only_error_mean']:.4f}m "
              f"(delta={row['pre_attack_reanchoring_delta_m']:.4f}m)")

    fieldnames = [
        "velocity_profile", "attack_profile", "n_samples", "cusum_h",
        "attack_start_t_s",
        "trust_weighted_error_mean", "trust_weighted_error_max", "trust_weighted_error_final",
        "hard_switch_error_mean", "hard_switch_error_max", "hard_switch_error_final",
        "gps_only_error_mean", "gps_only_error_max", "gps_only_error_final",
        "dr_only_error_mean", "dr_only_error_max", "dr_only_error_final",
        "trust_weighted_max_jump_m", "hard_switch_max_jump_m",
        "pre_attack_trust_weighted_error_mean", "pre_attack_gps_only_error_mean",
        "pre_attack_reanchoring_delta_m",
    ]
    out_df = pd.DataFrame(rows)[fieldnames]
    out_df.to_csv("fusion_results_summary.csv", index=False)
    print("\nSaved fusion_results_summary.csv")


if __name__ == "__main__":
    main()
