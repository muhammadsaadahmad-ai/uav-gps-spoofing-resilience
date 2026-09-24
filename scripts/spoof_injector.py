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

Output: spoofed_gps.csv, with columns
[t, true_x_m, true_y_m, spoofed_x_m, spoofed_y_m, is_spoofed] so
detector.py can be evaluated against ground truth labels of when
spoofing was actually active.

Three attack types, selected via the optional 4th positional arg
(attack_type, default "gradual" -- so every existing invocation of this
script, argument-for-argument, is unaffected):

  gradual (default, unchanged from the original design): the fake
    position walks away from truth at a constant rate in a fixed
    direction starting at attack_start_t, so the gap grows linearly and
    smoothly -- hard to catch via a single-sample jump check. This is
    the attack type all 6 original trials were generated and validated
    with.

  sudden_jump: the fake position jumps to a FIXED offset from truth in
    a single sample at attack_start_t and stays there (plus the same
    small noise) for the rest of the flight -- no ramp. Included
    specifically because reference papers flag sudden jumps as the
    "naive," easily-detected case: showing the detector handles both
    this easy case and the harder gradual case is a stronger claim than
    testing the hard case alone.

  intermittent: the attack cycles on/off in fixed-length blocks
    (INTERMITTENT_ON_S spoofed, INTERMITTENT_OFF_S reverted to true GPS)
    starting at attack_start_t and repeating for the rest of the
    flight. is_spoofed is computed per-sample from which block that
    sample's timestamp falls in -- not a single global start flag --
    so it correctly toggles 0/1/0/1... at each block boundary. During
    "on" blocks the drift magnitude is still gradual_drift_rate_mps *
    (t - attack_start_t) -- i.e. elapsed time keeps advancing across
    the "off" gaps rather than resetting each time a block turns back
    on. That models an attacker whose underlying drift computation runs
    continuously but who only intermittently asserts it (e.g. an
    intermittent jammer/spoofer), rather than restarting a fresh small
    ramp on every "on" block -- and it reuses the exact same drift
    formula as "gradual" with no new parameter.

Usage:
    python3 spoof_injector.py gps_ground_truth.csv \
        [attack_start_fraction] [drift_rate_mps] [attack_type]
"""

import sys
import numpy as np
import pandas as pd


ATTACK_TYPES = ("gradual", "sudden_jump", "intermittent")

# --- Attack profile parameters (gradual, and shared by all types) ---
# When (as a fraction of the flight duration) the spoofing attack begins.
ATTACK_START_FRACTION = 0.4

# How fast the spoofed position drifts away from truth, in meters/second.
# Kept deliberately low (well within plausible drone speeds) so the
# attack isn't a trivial "impossible jump" -- that's what makes gradual
# spoofing hard to detect via naive thresholding alone.
DRIFT_RATE_MPS = 0.6

# Direction the fake position drifts/offsets, in radians (0 = +x axis).
# A fixed direction produces a smooth, deliberate-looking pull,
# consistent with a real spoofing attack profile.
DRIFT_DIRECTION_RAD = np.deg2rad(35)

# Small sensor-realistic noise added to the fake signal so it doesn't
# look suspiciously perfect/noise-free compared to real GPS.
NOISE_STD_M = 0.05

# --- sudden_jump-specific parameter ---
# Fixed offset magnitude (meters) the spoofed position jumps to at
# attack_start_t and holds thereafter. Well above NOISE_STD_M so the
# jump is unambiguous, and comparable in scale to what "gradual" only
# reaches after several seconds of drift -- deliberately the "easy,
# should-be-obvious" attack shape rather than a subtle one.
JUMP_OFFSET_M = 5.0

# --- intermittent-specific parameters ---
INTERMITTENT_ON_S = 8.0    # spoofed-block duration
INTERMITTENT_OFF_S = 4.0   # true-GPS-block duration, repeating after attack_start_t


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 spoof_injector.py gps_ground_truth.csv "
              "[attack_start_fraction] [drift_rate_mps] [attack_type]")
        sys.exit(1)

    csv_path = sys.argv[1]
    attack_start_fraction = float(sys.argv[2]) if len(sys.argv) > 2 else ATTACK_START_FRACTION
    drift_rate_mps = float(sys.argv[3]) if len(sys.argv) > 3 else DRIFT_RATE_MPS
    attack_type = sys.argv[4] if len(sys.argv) > 4 else "gradual"
    if attack_type not in ATTACK_TYPES:
        print(f"Unknown attack_type '{attack_type}'. Choices: {ATTACK_TYPES}")
        sys.exit(1)

    gt = pd.read_csv(csv_path).sort_values("t").reset_index(drop=True)

    t0 = gt["t"].iloc[0]
    t_end = gt["t"].iloc[-1]
    duration = t_end - t0
    attack_start_t = t0 + attack_start_fraction * duration
    intermittent_cycle_s = INTERMITTENT_ON_S + INTERMITTENT_OFF_S

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
            # before taking over, as real spoofing attacks do). Same for
            # all three attack types.
            fx, fy = true_x, true_y
            spoofed_flag = 0

        elif attack_type == "gradual":
            # Drift the reported position away from truth at a constant
            # rate in a fixed direction, so the gap between real and
            # spoofed grows linearly (smooth, gradual, hard to catch via
            # a single-sample jump-detection check).
            elapsed = t - attack_start_t
            drift_mag = drift_rate_mps * elapsed
            dx = drift_mag * np.cos(DRIFT_DIRECTION_RAD)
            dy = drift_mag * np.sin(DRIFT_DIRECTION_RAD)
            fx = true_x + dx + rng.normal(0, NOISE_STD_M)
            fy = true_y + dy + rng.normal(0, NOISE_STD_M)
            spoofed_flag = 1

        elif attack_type == "sudden_jump":
            # Jump directly to a fixed offset -- no ramp, no dependence
            # on elapsed time, so the full offset is present from the
            # very first spoofed sample.
            dx = JUMP_OFFSET_M * np.cos(DRIFT_DIRECTION_RAD)
            dy = JUMP_OFFSET_M * np.sin(DRIFT_DIRECTION_RAD)
            fx = true_x + dx + rng.normal(0, NOISE_STD_M)
            fy = true_y + dy + rng.normal(0, NOISE_STD_M)
            spoofed_flag = 1

        else:  # intermittent
            elapsed = t - attack_start_t
            phase = elapsed % intermittent_cycle_s
            if phase < INTERMITTENT_ON_S:
                # "On" block: same gradual drift formula, driven by
                # elapsed time since the ORIGINAL attack_start_t (not
                # reset at each block boundary -- see module docstring).
                drift_mag = drift_rate_mps * elapsed
                dx = drift_mag * np.cos(DRIFT_DIRECTION_RAD)
                dy = drift_mag * np.sin(DRIFT_DIRECTION_RAD)
                fx = true_x + dx + rng.normal(0, NOISE_STD_M)
                fy = true_y + dy + rng.normal(0, NOISE_STD_M)
                spoofed_flag = 1
            else:
                # "Off" block: reverts to true GPS.
                fx, fy = true_x, true_y
                spoofed_flag = 0

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
    print(f"Attack type: {attack_type}")
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
