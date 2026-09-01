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
        df["predicted_spoofed"] = np.zeros(n, dtype=int)
        return df

    baseline_gap = float(np.median(slope_gap[calib_mask]))
    k = CUSUM_K_MULT * baseline_gap
    h = CUSUM_H_MULT * baseline_gap

    residual = slope_gap - baseline_gap
    cusum = np.zeros(n)
    predicted = np.zeros(n, dtype=int)
    alarmed = False
    for i in range(1, n):
        if np.isnan(residual[i]):
            cusum[i] = cusum[i - 1]
        else:
            cusum[i] = max(0.0, cusum[i - 1] + residual[i] - k)
            if cusum[i] > h:
                alarmed = True
        predicted[i] = 1 if alarmed else 0  # latched: a raised alarm doesn't self-clear

    df = df.copy()
    df["divergence_m"] = divergence
    df["slope_gap_mps"] = slope_gap
    df["cusum_stat"] = cusum
    df["predicted_spoofed"] = predicted
    df.attrs["baseline_gap_mps"] = baseline_gap
    df.attrs["cusum_k"] = k
    df.attrs["cusum_h"] = h
    df.attrs["n_calib"] = n_calib
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
        t0 = df["t"].iloc[0]
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
