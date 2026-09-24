"""
dead_reckoning.py  (v2 — fixes identified in the Claude Code audit)

Changes from v1:
  1. GRAVITY / ATTITUDE COMPENSATION (the dominant fix). v1 rotated raw
     body-frame accel into world frame using YAW ONLY, with no roll/pitch.
     On a quadrotor, any tilt redirects part of the gravity reaction into
     the horizontal accelerometer axes -- v1 was integrating that
     gravity-leakage as if it were real translational acceleration, which
     the audit identified as the dominant error source (not sensor noise).
     v2 uses the full roll/pitch/yaw rotation (body -> NED) from the
     ATTITUDE message and explicitly adds back the gravity vector, so only
     genuine specific-force-driven acceleration remains before integration.

  2. Uses the corrected sample rate (50Hz IMU, requested by
     mavlink_reader.py v2) instead of silently integrating at ~4Hz.

  3. Fixes the alignment bug: v1 used np.searchsorted alone, which returns
     "next-or-equal," not "nearest" -- silently wrong whenever the true
     nearest sample is the previous one. v2 checks both candidates and
     picks whichever is actually closer in time.

Usage:
    python3 dead_reckoning.py flight_log.csv
"""

import sys
import numpy as np
import pandas as pd


ACCEL_SCALE = 0.00980665     # mG -> m/s^2 (verified against ArduPilot's
                              # send_raw_imu(): accel * 1000 / GRAVITY_MSS)
GYRO_SCALE = 0.001           # mrad/s -> rad/s
GRAVITY_MSS = 9.80665

METERS_PER_DEG_LAT = 111320.0


def nearest_index(sorted_times, t):
    """
    Correct nearest-neighbor lookup. np.searchsorted alone gives the
    first index >= t ("next-or-equal"), which is wrong whenever the
    previous sample is actually closer -- this checks both candidates.
    """
    idx = np.searchsorted(sorted_times, t)
    if idx <= 0:
        return 0
    if idx >= len(sorted_times):
        return len(sorted_times) - 1
    before = idx - 1
    after = idx
    if abs(sorted_times[after] - t) < abs(sorted_times[before] - t):
        return after
    return before


def load_imu(df):
    imu = df[df["msg_type"] == "RAW_IMU"].copy()
    imu["t"] = imu["timestamp"].astype(float)
    imu = imu.sort_values("t").reset_index(drop=True)
    return imu


def load_attitude(df):
    att = df[df["msg_type"] == "ATTITUDE"].copy()
    att["t"] = att["timestamp"].astype(float)
    att = att.sort_values("t").reset_index(drop=True)
    return att


def load_gps_ground_truth(df):
    gps = df[df["msg_type"] == "GLOBAL_POSITION_INT"].copy()
    gps["t"] = gps["timestamp"].astype(float)
    gps = gps.sort_values("t").reset_index(drop=True)

    lat0, lon0 = gps["lat"].iloc[0], gps["lon"].iloc[0]
    meters_per_deg_lon = METERS_PER_DEG_LAT * np.cos(np.radians(lat0))

    gps["x_m"] = (gps["lon"] - lon0) * meters_per_deg_lon  # local East
    gps["y_m"] = (gps["lat"] - lat0) * METERS_PER_DEG_LAT  # local North
    return gps


def attach_attitude(imu, att):
    """
    For each IMU sample, attach the nearest-in-time roll/pitch/yaw so we
    can rotate that sample's specific force into world frame correctly.
    """
    att_t = att["t"].values
    rolls, pitches, yaws = [], [], []
    for t in imu["t"]:
        idx = nearest_index(att_t, t)
        rolls.append(att["roll"].iloc[idx])
        pitches.append(att["pitch"].iloc[idx])
        yaws.append(att["yaw"].iloc[idx])

    imu = imu.copy()
    imu["att_roll"] = rolls
    imu["att_pitch"] = pitches
    imu["att_yaw"] = yaws
    return imu


def body_to_ned(roll, pitch, yaw):
    """
    Standard aerospace ZYX Euler rotation matrix, body -> NED.
    Body frame: x forward, y right, z down (matches ArduPilot's RAW_IMU
    and ATTITUDE conventions).
    """
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy,     sx * cy,                cx * cy],
    ])
    return R


def dead_reckon(imu):
    """
    Strapdown integration with gravity compensation:
      a_ned = R(roll,pitch,yaw) @ specific_force_body + g_ned

    specific_force is what the accelerometer actually measures (it reads
    ~[0,0,-g] at rest, the reaction to gravity, not gravity itself), so we
    add the true gravity vector g_ned = [0,0,+g] back in NED to recover
    the genuine (non-gravity) acceleration -- this is what strips out the
    gravity-leakage-during-tilt error v1 had no way to remove.

    Horizontal (North/East) components of a_ned are then double-integrated
    to position, matching the x_m (East) / y_m (North) convention used by
    load_gps_ground_truth() for direct comparison.
    """
    n = len(imu)
    x = np.zeros(n)   # East
    y = np.zeros(n)   # North
    vx = np.zeros(n)
    vy = np.zeros(n)
    g_ned = np.array([0.0, 0.0, GRAVITY_MSS])

    for i in range(1, n):
        dt = imu["t"].iloc[i] - imu["t"].iloc[i - 1]
        if dt <= 0 or dt > 1.0:
            x[i], y[i] = x[i - 1], y[i - 1]
            vx[i], vy[i] = vx[i - 1], vy[i - 1]
            continue

        f_body = np.array([
            imu["xacc"].iloc[i] * ACCEL_SCALE,
            imu["yacc"].iloc[i] * ACCEL_SCALE,
            imu["zacc"].iloc[i] * ACCEL_SCALE,
        ])

        R = body_to_ned(
            imu["att_roll"].iloc[i],
            imu["att_pitch"].iloc[i],
            imu["att_yaw"].iloc[i],
        )
        a_ned = R @ f_body + g_ned  # gravity-compensated true acceleration

        a_north = a_ned[0]
        a_east = a_ned[1]

        vx[i] = vx[i - 1] + a_east * dt
        vy[i] = vy[i - 1] + a_north * dt
        x[i] = x[i - 1] + vx[i] * dt
        y[i] = y[i - 1] + vy[i] * dt

    imu = imu.copy()
    imu["dr_x"] = x   # East, meters -- matches gps x_m
    imu["dr_y"] = y   # North, meters -- matches gps y_m
    imu["dr_vx"] = vx
    imu["dr_vy"] = vy
    return imu


def compute_drift(dr, gps):
    errors = []
    times = []
    gps_t = gps["t"].values
    for _, row in dr.iterrows():
        idx = nearest_index(gps_t, row["t"])
        gt = gps.iloc[idx]
        err = np.hypot(row["dr_x"] - gt["x_m"], row["dr_y"] - gt["y_m"])
        errors.append(err)
        times.append(row["t"])
    return np.array(times), np.array(errors)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 dead_reckoning.py flight_log.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    imu = load_imu(df)
    att = load_attitude(df)
    gps = load_gps_ground_truth(df)

    if len(imu) < 2 or len(gps) < 2:
        print("Not enough IMU or GPS samples in this log to dead-reckon.")
        sys.exit(1)
    if len(att) < 2:
        print("WARNING: no/insufficient ATTITUDE data found. Did you use "
              "the v2 mavlink_reader.py, which logs ATTITUDE? Falling back "
              "is not implemented -- re-log the flight with v2.")
        sys.exit(1)

    imu_dt = imu["t"].diff().dropna()
    print(f"IMU samples: {len(imu)}  |  mean dt: {imu_dt.mean():.4f}s "
          f"({1/imu_dt.mean():.1f} Hz)")

    imu = attach_attitude(imu, att)
    dr = dead_reckon(imu)
    times, errors = compute_drift(dr, gps)

    print(f"GPS samples: {len(gps)}")
    print("Dead-reckoning position error (meters) vs GPS ground truth:")
    print(f"  mean:  {errors.mean():.3f} m")
    print(f"  max:   {errors.max():.3f} m")
    print(f"  final: {errors[-1]:.3f} m  (at t={times[-1] - times[0]:.1f}s into the log)")

    out = dr[["t", "dr_x", "dr_y"]].copy()
    out.to_csv("dead_reckoning_output.csv", index=False)
    gps[["t", "x_m", "y_m"]].to_csv("gps_ground_truth.csv", index=False)
    print("\nSaved dead_reckoning_output.csv and gps_ground_truth.csv")


if __name__ == "__main__":
    main()
