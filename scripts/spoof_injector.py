"""
spoof_injector.py

Stage 1 data-generation tool: takes real GPS ground truth (from
dead_reckoning.py's output, gps_ground_truth.csv) and produces a
SPOOFED version of the trajectory -- a fake GPS feed that starts by
matching the real signal closely, then smoothly drifts away.

This mimics a realistic spoofing attack profile: an attacker doesn't
jump the reported position instantly (that would be trivially easy to
detect via an implausible position jump). Instead, they walk the fake
position away gradually, staying within a physically plausible
velocity envelope, so the drone's own consistency checks don't
immediately flag it.

Output: spoofed_gps.csv, with columns [t, x_m, y_m, is_spoofed]
so the detector (Day 5) can be trained/evaluated against ground truth
labels of when spoofing was actually active.

Usage:
    python3 spoof_injector.py gps_ground_truth.csv
"""

import sys
import numpy as np
import pandas as pd


# --- Attack profile parameters ---
# When (as a fraction of the flight duration) the spoofing attack begins.
ATTACK_START_FRACTION = 0.4

# How fast the spoofed position drifts away from truth, in meters/second.
# Kept deliberately low (well within plausible drone speeds) so the
# attack isn't a trivial "impossible jump" -- that's what makes
# spoofing hard to detect via naive thresholding alone.
DRIFT_RATE_MPS = 0.6

# Direction the fake position drifts, in radians (0 = +x axis).
# A fixed direction produces a smooth, deliberate-looking pull,
# consistent with a real spoofing attack profile.
DRIFT_DIRECTION_RAD = np.deg2rad(35)

# Small sensor-realistic noise added to the fake signal so it doesn't
# look suspiciously perfect/noise-free compared to real GPS.
NOISE_STD_M = 0.05


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 spoof_injector.py gps_ground_truth.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    gt = pd.read_csv(csv_path).sort_values("t").reset_index(drop=True)

    t0 = gt["t"].iloc[0]
    t_end = gt["t"].iloc[-1]
    duration = t_end - t0
    attack_start_t = t0 + ATTACK_START_FRACTION * duration

    rng = np.random.default_rng(42)  # reproducible noise

    spoofed_x = []
    spoofed_y = []
    is_spoofed = []

    for _, row in gt.iterrows():
        t = row["t"]
        true_x, true_y = row["x_m"], row["y_m"]

        if t < attack_start_t:
            # Before the attack: spoofed feed == real feed (attacker
            # hasn't started yet, or is still "capturing" the real signal
            # before taking over, as real spoofing attacks do).
            fx, fy = true_x, true_y
            spoofed_flag = 0
        else:
            # After attack start: drift the reported position away from
            # truth at a constant rate in a fixed direction, so the gap
            # between real and spoofed grows linearly (smooth, gradual,
            # hard to catch via a single-sample jump-detection check).
            elapsed = t - attack_start_t
            drift_mag = DRIFT_RATE_MPS * elapsed
            dx = drift_mag * np.cos(DRIFT_DIRECTION_RAD)
            dy = drift_mag * np.sin(DRIFT_DIRECTION_RAD)
            fx = true_x + dx + rng.normal(0, NOISE_STD_M)
            fy = true_y + dy + rng.normal(0, NOISE_STD_M)
            spoofed_flag = 1

        spoofed_x.append(fx)
        spoofed_y.append(fy)
        is_spoofed.append(spoofed_flag)

    out = pd.DataFrame({
        "t": gt["t"],
        "true_x_m": gt["x_m"],
        "true_y_m": gt["y_m"],
        "spoofed_x_m": spoofed_x,
        "spoofed_y_m": spoofed_y,
        "is_spoofed": is_spoofed,
    })

    out.to_csv("spoofed_gps.csv", index=False)

    n_spoofed = sum(is_spoofed)
    print(f"Total samples: {len(out)}")
    print(f"Attack starts at t={attack_start_t - t0:.1f}s into the flight "
          f"({n_spoofed} of {len(out)} samples spoofed)")

    # Report final divergence for a quick sanity check
    final = out.iloc[-1]
    final_err = np.hypot(final["spoofed_x_m"] - final["true_x_m"],
                          final["spoofed_y_m"] - final["true_y_m"])
    print(f"Final spoofed-vs-true position error: {final_err:.2f} m")
    print("Saved spoofed_gps.csv")


if __name__ == "__main__":
    main()
