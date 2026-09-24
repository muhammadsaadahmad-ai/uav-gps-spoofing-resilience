"""
detector.py  (v3 — replaces absolute-divergence thresholding with a
change-point detector on the divergence GROWTH RATE)

Changes from v2:
  v2 flagged spoofing by thresholding WINDOW_SECONDS-windowed absolute
  GPS-vs-DR divergence against a fixed DIVERGENCE_THRESHOLD_M=1.0. That
  has the same shape of bug the mavlink_reader.py and dead_reckoning.py
  audits both found: it silently couples detection to a quantity that
  isn't attack-specific.

  Verified directly against this flight (not assumed): binning the v2
  divergence_m in 5s intervals across the WHOLE flight shows it rising
  *monotonically* from t=0 -- 0.057m (0-5s) -> 0.432m (10-15s) -> 0.801m
  (20-25s) -> 1.000m (25-30s), crossing the 1.0m threshold right around
  when the attack happens to start at t=31.0s, then continuing to climb
  the same way afterwards. There is no step at the attack boundary in
  this signal -- ambient dead-reckoning drift (root-caused in the
  dead_reckoning.py audit as residual specific-force bias, which grows
  roughly QUADRATICALLY with elapsed flight time -- see body_to_ned()/
  dead_reckon()) makes absolute divergence UNBOUNDED and growing
  whether or not any attack is happening. Re-deriving divergence as the
  raw (unwindowed) absolute gap confirms it even more starkly: this
  flight's ambient DR-vs-GPS divergence reaches ~9m by t=31s and ~70m by
  t=77s from drift alone (checked against dead_reckoning.py's own
  GPS-truth comparison: 55.9m mean error over this same 77s flight).
  ANY fixed absolute threshold will eventually be crossed by pure
  ambient drift -- it is just a matter of how long the flight runs. The
  34 FPs in the v2 run are exactly what that predicts: clustered in the
  handful of seconds before the attack label flips, where drift alone
  happened to cross 1.0m.

  v3 detects a SUSTAINED CHANGE IN THE GROWTH RATE of divergence instead
  of an absolute divergence level:

    1. divergence_m is now the raw (unwindowed) absolute GPS-vs-DR gap
       hypot(spoofed - dr) at each sample -- simpler and more legible
       for manual inspection than v2's windowed-delta definition, and
       what the redesigned decision logic below is built on.

    2. Two FIXED trailing windows are used to estimate the *current*
       rate of divergence growth: short_slope (SHORT_WINDOW_S, reacts
       quickly) and long_slope (LONG_WINDOW_S, a slower-moving
       baseline). slope_gap_mps = short_slope - long_slope.

       Why this is structurally different from just thresholding a
       derivative: for a smooth quadratic drift d(t) ~= 0.5*a*t^2 (the
       established ambient error shape), the instantaneous slope at
       time t is ~= a*t, and the average slope over a trailing window
       of length W is ~= a*(t - W/2). With two FIXED window lengths,
       slope_gap = short_slope - long_slope ~= a*(LONG_WINDOW_S -
       SHORT_WINDOW_S)/2 -- approximately CONSTANT over time, because
       both windows' systematic offset from "now" scales the same way.
       This is the standard trend-differencing idea (comparable to
       ARIMA-style differencing to strip a polynomial trend): it turns
       an unbounded, ever-growing quantity (raw divergence) into a
       roughly stationary one (slope_gap), which is the right kind of
       signal to put a threshold on. Verified against this flight:
       slope_gap sits in a ~0.09-0.13 m/s band for the entire pre-attack
       window (t=15s, when it first becomes computable, through t=31s),
       then jumps to 0.15-0.56 within seconds of the true attack onset.

       This was NOT the first thing tried: using "average rate since
       flight start" as the long window (instead of a second FIXED
       window) does NOT flatten -- the gap keeps growing the whole
       flight (0.02 at t=5s to 0.25 at t=28s), because that comparison
       is (instantaneous slope) vs (average slope since t=0), which for
       a quadratic differs by a term that itself grows with t. Both
       windows have to be fixed trailing windows for the cancellation
       above to hold. This is exactly the kind of thing the last two
       audits flagged: verify the mechanism against data, don't trust
       that a plausible-sounding derivative fixes a drift confound.

    3. A one-sided CUSUM change-point detector accumulates evidence that
       slope_gap is running persistently above its calibrated baseline:
           S[i] = max(0, S[i-1] + (slope_gap[i] - baseline - k) )
       flagging spoofed once S crosses h, and latching (a security
       alarm should not silently self-clear). k and h are set as
       MULTIPLES OF THE CALIBRATED BASELINE MEAN, not multiples of the
       calibration window's standard deviation. That distinction
       matters and was verified against this data, not assumed: this
       SITL flight's pre-attack signal is almost perfectly deterministic
       (near-zero process noise -- the calibration-window std of
       slope_gap is ~0.002, three orders of magnitude smaller than its
       mean of ~0.112). A std-based CUSUM (k=0.5*sigma, h=5*sigma, the
       standard textbook choice) computed h=0.0104 here -- so tight that
       it fired at t=27.9s, 3s before the real attack, on nothing but
       the slow, continuous drift of the ambient baseline itself. That
       is the SAME confound this whole redesign exists to remove, just
       one derivative level down. A SITL sim's near-zero sensor noise
       makes the in-sample std unrepresentative of any real noise floor
       (see the writeup for the fuller honesty note on this). Scaling
       k/h off the calibrated MEAN instead is far less sensitive to
       that: it says "alarm once the growth rate is running at ~double
       its calibrated normal level, sustained" rather than "alarm at
       five sigma of whatever the noise happened to be in the first
       10 seconds," and degrades gracefully if per-flight noise varies.

  CALIBRATION_DURATION_S is a fixed early window assumed attack-free (a
  boot-time calibration hold), the same category of assumption v2's
  fixed WINDOW_SECONDS made, but now used to characterize a rate rather
  than picked as a magic absolute distance threshold.

Changes from v3 (this version): adds a SECOND, independent detection path
for INSTANTANEOUS jumps, which the CUSUM path above is structurally blind
to. Smoke-tested directly against a new "sudden_jump" attack type (a fixed
offset applied in a single sample, no ramp): TP=0, FN=546, precision=0,
recall=0 -- the CUSUM path never fired at all. Root cause: a one-sample
step to a new CONSTANT offset produces only a single-sample derivative
spike; both the 3s and 15s trailing-slope windows average it away almost
immediately, so slope_gap barely moves and cusum_stat never approaches h.
This is not a bug in the CUSUM design -- it is precisely what "detect a
SUSTAINED change in growth rate" is supposed to ignore -- it is a genuine
blind spot for a *different* attack shape that needs its own signal.

  1. jump_delta_m[i] = |(spoofed[i]-spoofed[i-1]) - (dr[i]-dr[i-1])|, the
     magnitude of the CHANGE in the (spoofed-dr) gap VECTOR -- not
     |divergence[i]-divergence[i-1]| (the difference of the gap's
     MAGNITUDE), and not raw GPS-position displacement. Both alternatives
     were tried first and both failed for reasons only visible against
     real data, not assumed:

       - |divergence[i]-divergence[i-1]| (first attempt) is a difference
         of magnitudes, which the reverse triangle inequality permits to
         be much SMALLER than the actual change in the gap. Caught this
         directly against the sudden_jump validation flight: a 5m jump
         injected at t=37s, once dead-reckoning had already drifted
         ~21m from truth in some other direction, produced only a
         ~1.0m change in |divergence| -- the injected offset partially
         CANCELLED the pre-existing drift vector. The jump detector
         missed the attack entirely (TP=0) on the first implementation,
         not because the threshold was miscalibrated but because the
         chosen quantity itself was silently attenuating late-onset
         jumps. The vector-difference form doesn't have this problem:
         since DR's own sample-to-sample movement is always tiny
         (real vehicle dynamics, not attacker-controlled), the vector
         difference is dominated by however much the SPOOFED position
         jumped, regardless of the pre-existing drift magnitude or
         direction -- confirmed on the same flight: the vector form
         reads 4.86m at that exact same jump sample, matching the
         injected 5m offset almost exactly.

       - Raw GPS-position displacement (considered, not implemented) was
         also checked against data and rejected: several trials'
         calibration windows fall during a hover/stabilize hold before
         any velocity command runs (moderate_fast_drift's calibration
         window has EXACTLY 0.0 displacement), which would calibrate a
         near-zero threshold that fires on any subsequent real motion at
         all; and real per-sample motion during "aggressive"-style
         maneuvering reaches up to 1.28m, too close to a 5m attack to
         leave a hardcoded floor that's both safe and sensitive. The
         (spoofed-dr) vector-difference doesn't have this problem either:
         during genuine vehicle motion, both the spoofed feed and DR are
         tracking the SAME real movement, so they mostly cancel in the
         difference, leaving only their (small) disagreement -- unlike
         raw position displacement, which sees the full real velocity as
         "signal" regardless of whether DR agrees with it.

  2. Calibrated the same principled way as CUSUM's k/h (a fixed early
     boot-time window, verified against real per-flight data, not a
     hardcoded guess) -- but off that window's MAX, not mean or std, for
     a reason verified directly against this project's own trials:
     jump_delta_m is a bursty, maneuver-driven signal, not a stationary
     noisy one. aggressive_fast_drift's own 10s calibration window has
     jump_delta mean=0.033m, std=0.007m -- but later in that SAME
     pre-attack flight, once ambient DR drift has grown large, jump_delta
     spikes to 0.989m, a real dynamical event, ~20x the calibration
     window's own max. A std- or mean-based threshold from the early
     window would have been blown through by this later, otherwise-benign
     maneuvering spike -- the same "SITL's near-zero early-window noise
     doesn't bound real later variance" lesson CUSUM's own k/h derivation
     already had to learn, one level down.

     A pure "calibration max x multiplier" wasn't enough by itself,
     though -- checked directly against the new circular/figure_eight/
     sustained_cruise profiles, not assumed: the circular-profile flight's
     own calibration window has a ~4x higher baseline jump_delta max
     (0.198m) than aggressive_fast_drift's (0.049m), just from being a
     continuously-turning profile rather than mostly straight segments.
     A multiplier sized to safely clear aggressive_fast_drift's 20x
     worst-case ratio (needs ~30x) becomes too LOOSE when applied to
     circular's already-higher baseline (30 x 0.198m = 5.94m -- bigger
     than the 5m attack itself, which would have missed it again). So
     the threshold is min(JUMP_THRESHOLD_MULT * calib_max,
     JUMP_THRESHOLD_CEILING_M): the relative multiplier gives real
     margin on quiet flights with a tiny genuine calibration baseline,
     and the absolute ceiling stops a single noisier-baseline flight
     from inflating the threshold past where it can still catch a
     real attack. Both constants are justified with exact numbers in
     their own comments below, verified against every trial and
     validation flight available in this project, not picked upfront.

  3. Latched exactly like the CUSUM alarm -- once jump_delta_m crosses
     its threshold, this signal stays fired for the rest of the flight
     (same "a security alarm should not silently self-clear" principle).

  4. Combined with the CUSUM alarm via OR: predicted_spoofed = 1 if
     EITHER latch has fired. Each signal's own internal logic (CUSUM's
     slope-gap accumulator, the jump latch's threshold crossing) is
     untouched by the other -- this is a detection-level OR, not a
     shared statistic.

CALIBRATION_DURATION_S is a fixed early window assumed attack-free (a
boot-time calibration hold), the same category of assumption v2's fixed
WINDOW_SECONDS made, but now used to characterize a rate (and, as of this
version, a jump-size ceiling) rather than picked as a magic absolute
distance threshold.

Usage (unchanged):
    python3 detector.py spoofed_gps.csv dead_reckoning_output.csv
"""

import sys
import numpy as np
import pandas as pd


# --- Windows for the short-vs-long growth-rate comparison ---
SHORT_WINDOW_S = 3.0     # reacts quickly to a genuine rate change
LONG_WINDOW_S = 15.0     # slow-moving baseline rate

# --- Calibration: fixed early hold, assumed attack-free ---
# Starts at LONG_WINDOW_S because slope_gap isn't computable before that
# (long_slope needs LONG_WINDOW_S seconds of history).
CALIBRATION_DURATION_S = 10.0
MIN_CALIBRATION_SAMPLES = 20

# --- CUSUM parameters, as multiples of the calibrated baseline MEAN
# (not std -- see module docstring for why std is unreliable here) ---
CUSUM_K_MULT = 0.5   # slack: half the baseline rate is "normal" fluctuation
CUSUM_H_MULT = 2.0   # alarm once sustained excess reaches the baseline rate again

# --- Jump-detector parameters. threshold = min(JUMP_THRESHOLD_MULT *
# calib_max, JUMP_THRESHOLD_CEILING_M), floored at MIN_JUMP_THRESHOLD_M
# -- see module docstring for the full reasoning; summarized here with
# the exact numbers that produced each constant.
#
# JUMP_THRESHOLD_MULT: a multiple of the calibration window's MAX
# per-sample jump_delta_m (not mean/std -- this signal is bursty/
# maneuver-driven, so mean/std from a calm early window badly
# underestimate a later real maneuvering spike; verified directly:
# aggressive_fast_drift's calibration window has jump_delta max=0.049m,
# but later in that SAME pre-attack flight it spikes to 0.989m, a real
# maneuvering event, ~20x the calibration window's own max). 30x gives
# that specific (worst-observed) case ~1.5x margin above its own worst
# pre-attack spike.
JUMP_THRESHOLD_MULT = 30.0
#
# JUMP_THRESHOLD_CEILING_M: an absolute cap on top of the multiplier,
# needed because the relative multiplier alone doesn't transfer safely
# across flights with different inherent baseline noise -- verified
# directly: the circular-profile validation flight's calibration window
# has a jump_delta max of 0.198m, ~4x aggressive_fast_drift's 0.049m,
# just from being a continuously-turning profile. JUMP_THRESHOLD_MULT
# sized for aggressive_fast_drift's worst case would give circular's
# flight a threshold of 30*0.198=5.94m -- ABOVE the 5m sudden_jump
# attack, which would miss it again. 2.5m sits below every attack this
# project tests (the 5m sudden_jump case clears it by ~1.9-3.4x
# depending on how much the vector-difference signal is itself affected
# by that flight's own noise) while still clearing every trial and
# validation flight's own strict pre-attack (t < attack_start_t) max by
# at least 1.5x -- verified against all 6 original trials plus the
# circular-profile sudden_jump/intermittent validation flights, not
# just the one worst case.
JUMP_THRESHOLD_CEILING_M = 2.5
#
# Defensive floor in case a calibration window happens to be perfectly
# static (calib max == 0) -- never engaged by any trial or validation
# flight in this project (all have calib max >= 0.049m), but without it
# a zero-noise calibration window would produce a zero threshold that
# fires on any nonzero jitter at all.
MIN_JUMP_THRESHOLD_M = 0.5


def nearest_index(sorted_times, t):
    """
    Correct nearest-neighbor lookup (next-or-equal from np.searchsorted
    alone is wrong whenever the previous sample is actually closer).
    """
    idx = np.searchsorted(sorted_times, t)
    if idx <= 0:
        return 0
    if idx >= len(sorted_times):
        return len(sorted_times) - 1
    before, after = idx - 1, idx
    if abs(sorted_times[after] - t) < abs(sorted_times[before] - t):
        return after
    return before


def load_and_align(spoofed_path, dr_path):
    spoofed = pd.read_csv(spoofed_path).sort_values("t").reset_index(drop=True)
    dr = pd.read_csv(dr_path).sort_values("t").reset_index(drop=True)

    dr_t = dr["t"].values
    matched_dr_x, matched_dr_y = [], []
    for t in spoofed["t"]:
        idx = nearest_index(dr_t, t)
        matched_dr_x.append(dr["dr_x"].iloc[idx])
        matched_dr_y.append(dr["dr_y"].iloc[idx])

    spoofed["dr_x"] = matched_dr_x
    spoofed["dr_y"] = matched_dr_y
    return spoofed


def _trailing_slope(t, d, i, window_s):
    """
    Average rate of change of d over the trailing window [t[i]-window_s,
    t[i]], using nearest_index for the lookup (same alignment approach
    used throughout this pipeline). Returns NaN if there isn't yet
    window_s seconds of history.
    """
    target_t = t[i] - window_s
    if target_t < t[0] - 1e-9:
        return np.nan
    j = nearest_index(t, target_t)
    if t[i] == t[j]:
        return np.nan
    return (d[i] - d[j]) / (t[i] - t[j])


def detect(df):
    n = len(df)
    t = df["t"].values

    # Raw absolute GPS-vs-DR divergence at each sample (meters).
    divergence = np.hypot(
        df["spoofed_x_m"].values - df["dr_x"].values,
        df["spoofed_y_m"].values - df["dr_y"].values,
    )

    # Magnitude of the sample-to-sample CHANGE in the (spoofed-dr) gap
    # VECTOR -- the signal the jump-detection path below is built on.
    # Independent of slope_gap/CUSUM; see module docstring for why this
    # (not |divergence[i]-divergence[i-1]|, not raw GPS displacement) is
    # what reliably catches sudden_jump attacks regardless of how much
    # ambient DR drift has already accumulated. jump_delta[0] is defined
    # as 0 (no prior sample to diff against) via prepend.
    d_spoofed_x = np.diff(df["spoofed_x_m"].values, prepend=df["spoofed_x_m"].values[0])
    d_spoofed_y = np.diff(df["spoofed_y_m"].values, prepend=df["spoofed_y_m"].values[0])
    d_dr_x = np.diff(df["dr_x"].values, prepend=df["dr_x"].values[0])
    d_dr_y = np.diff(df["dr_y"].values, prepend=df["dr_y"].values[0])
    jump_delta = np.hypot(d_spoofed_x - d_dr_x, d_spoofed_y - d_dr_y)

    short_slope = np.array([_trailing_slope(t, divergence, i, SHORT_WINDOW_S) for i in range(n)])
    long_slope = np.array([_trailing_slope(t, divergence, i, LONG_WINDOW_S) for i in range(n)])
    slope_gap = short_slope - long_slope  # ~stationary under smooth ambient drift

    # Calibrate against a fixed early, assumed-attack-free window.
    calib_start = t[0] + LONG_WINDOW_S
    calib_end = calib_start + CALIBRATION_DURATION_S
    calib_mask = (t >= calib_start) & (t < calib_end) & ~np.isnan(slope_gap)

    n_calib = int(calib_mask.sum())
    if n_calib < MIN_CALIBRATION_SAMPLES:
        print(f"WARNING: only {n_calib} calibration samples available "
              f"(need >= {MIN_CALIBRATION_SAMPLES}); flight is too short "
              f"relative to LONG_WINDOW_S+CALIBRATION_DURATION_S "
              f"({LONG_WINDOW_S + CALIBRATION_DURATION_S:.0f}s) for this "
              f"detector to calibrate. Predicting no spoofing throughout.")
        df = df.copy()
        df["divergence_m"] = divergence
        df["slope_gap_mps"] = slope_gap
        df["cusum_stat"] = np.zeros(n)
        df["jump_delta_m"] = jump_delta
        df["jump_alarm"] = np.zeros(n, dtype=int)
        df["predicted_spoofed"] = np.zeros(n, dtype=int)
        return df

    baseline_gap = float(np.median(slope_gap[calib_mask]))
    k = CUSUM_K_MULT * baseline_gap
    h = CUSUM_H_MULT * baseline_gap

    residual = slope_gap - baseline_gap
    cusum = np.zeros(n)
    cusum_predicted = np.zeros(n, dtype=int)
    alarmed = False
    for i in range(1, n):
        if np.isnan(residual[i]):
            cusum[i] = cusum[i - 1]
        else:
            cusum[i] = max(0.0, cusum[i - 1] + residual[i] - k)
            if cusum[i] > h:
                alarmed = True
        cusum_predicted[i] = 1 if alarmed else 0  # latched: a raised alarm doesn't self-clear

    # --- Jump-detection path (independent of CUSUM above; see module
    # docstring for why the threshold basis/multiplier were chosen this
    # way). Calibrated from the SAME fixed early window as CUSUM, off
    # that window's MAX per-sample jump (not mean/std -- see docstring).
    calib_jump_max = float(jump_delta[calib_mask].max())
    jump_threshold = min(
        max(JUMP_THRESHOLD_MULT * calib_jump_max, MIN_JUMP_THRESHOLD_M),
        JUMP_THRESHOLD_CEILING_M,
    )

    jump_alarm = np.zeros(n, dtype=int)
    jump_alarmed = False
    for i in range(1, n):
        if jump_delta[i] > jump_threshold:
            jump_alarmed = True
        jump_alarm[i] = 1 if jump_alarmed else 0  # latched, same principle as CUSUM's

    # --- Combine: predicted_spoofed = 1 if EITHER latch has fired.
    # Detection-level OR only -- neither signal's own internal logic
    # (CUSUM's accumulator above, the jump latch above) is touched here.
    predicted = np.maximum(cusum_predicted, jump_alarm)

    df = df.copy()
    df["divergence_m"] = divergence
    df["slope_gap_mps"] = slope_gap
    df["cusum_stat"] = cusum
    df["jump_delta_m"] = jump_delta
    df["jump_alarm"] = jump_alarm
    df["predicted_spoofed"] = predicted
    df.attrs["baseline_gap_mps"] = baseline_gap
    df.attrs["cusum_k"] = k
    df.attrs["cusum_h"] = h
    df.attrs["n_calib"] = n_calib
    df.attrs["calib_jump_max_m"] = calib_jump_max
    df.attrs["jump_threshold_m"] = jump_threshold
    return df


def evaluate(df):
    y_true = df["is_spoofed"].values
    y_pred = df["predicted_spoofed"].values

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if "baseline_gap_mps" in df.attrs:
        print(f"Calibration: {df.attrs['n_calib']} samples, "
              f"baseline slope_gap={df.attrs['baseline_gap_mps']:.4f} m/s, "
              f"k={df.attrs['cusum_k']:.4f}, h={df.attrs['cusum_h']:.4f}")
        print(f"Jump-detector calibration: calib max jump_delta="
              f"{df.attrs['calib_jump_max_m']:.4f} m, "
              f"threshold={df.attrs['jump_threshold_m']:.4f} m")

    if "jump_alarm" in df.columns:
        jump_arr = df["jump_alarm"].values
        jump_fired = bool(jump_arr.max()) if len(jump_arr) else False
        jump_first_idx = int(np.argmax(jump_arr == 1)) if jump_fired else None
        print(f"Jump-detector latch fired this run? {jump_fired}"
              + (f" (first at t={df['t'].iloc[jump_first_idx]:.2f})" if jump_fired else ""))

    print(f"Confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")

    attack_start_idx = np.argmax(y_true == 1) if np.any(y_true == 1) else None
    if attack_start_idx is not None:
        naive = (np.arange(len(y_true)) >= attack_start_idx).astype(int)
        identical = np.array_equal(naive, y_pred)
        print(f"Identical to naive elapsed-time rule? {identical}")

        first_detect_idx = None
        for i in range(attack_start_idx, len(y_true)):
            if y_pred[i] == 1:
                first_detect_idx = i
                break
        if first_detect_idx is not None:
            t_attack = df["t"].iloc[attack_start_idx]
            t_detect = df["t"].iloc[first_detect_idx]
            print(f"Detection latency: {t_detect - t_attack:.2f} s "
                  f"({first_detect_idx - attack_start_idx} samples)")
        else:
            print("Detector never caught the attack in this run.")

        # Where do the false positives actually sit relative to the attack
        # boundary? This is the check that caught v2's confound -- keep
        # doing it explicitly rather than trusting a clean-looking F1.
        t_attack = df["t"].iloc[attack_start_idx]
        fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
        if len(fp_idx) == 0:
            print("False positives: none.")
        else:
            offsets = df["t"].iloc[fp_idx].values - t_attack
            print(f"False positives: {len(fp_idx)}, at t-attack_start (s) = "
                  f"[{offsets.min():.2f} .. {offsets.max():.2f}]")
            print("  (negative = before the attack label starts; "
                  "clustering just before it means the same confound "
                  "v2 had, just smaller)")


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 detector.py spoofed_gps.csv dead_reckoning_output.csv")
        sys.exit(1)

    df = load_and_align(sys.argv[1], sys.argv[2])
    df = detect(df)
    evaluate(df)

    df.to_csv("detector_output.csv", index=False)
    print("\nSaved detector_output.csv")


if __name__ == "__main__":
    main()
