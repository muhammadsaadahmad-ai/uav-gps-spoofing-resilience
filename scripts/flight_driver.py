"""
flight_driver.py

Drives a real GUIDED-mode flight programmatically via MAVLink commands
(no manual SITL console typing). mavlink_reader.py is already bound to
udp:127.0.0.1:14550 (the fixed destination MAVProxy's --out link sends
telemetry to), so a second process cannot also bind that same port.

Instead, this script sends commands directly to MAVProxy's *own* UDP
socket for that out-link (discovered dynamically via lsof/ss on the
running mavproxy.py process -- an ephemeral port the OS assigned it,
e.g. 45772). That socket is a plain, unfiltered recvfrom() on MAVProxy's
side, so any valid MAVLink packet sent to it is accepted exactly as if
it came from the GCS at 14550, and MAVProxy forwards it on to the
ArduCopter master link. This is real MAVLink command traffic over the
wire -- not console text injection.

Because MAVProxy's out-link only ever *replies* to its fixed 14550
destination (see pymavlink mavudp.write()), this script has no return
channel of its own. State feedback (armed/alt/position) instead comes
from tailing flight_log.csv, which mavlink_reader.py is continuously
appending -- i.e. we "poll GLOBAL_POSITION_INT" via the same live data
the rest of the pipeline will use, just already log-persisted.

Usage: python3 flight_driver.py [profile_name]
    profile_name: one of "gentle", "moderate", "aggressive", "circular",
    "figure_eight", "sustained_cruise" (default "gentle") -- see
    VELOCITY_PROFILES for the full list.
"""

import math
import subprocess
import sys
import time

from pymavlink import mavutil

FLIGHT_LOG = "flight_log.csv"
TARGET_SYSTEM = 1
TARGET_COMPONENT = 1
TAKEOFF_ALT_M = 10.0
ALT_TOLERANCE_M = 1.0

# Velocity profiles: list of (vx, vy, vz, duration_s) segments in
# MAV_FRAME_LOCAL_NED (vx=North, vy=East, vz=Down). vz is always 0 --
# these are horizontal-only maneuvers so altitude stays put. A trailing
# (0,0,0, HOVER_TAIL_S) settle segment is appended automatically after
# whichever profile runs, before the landing command.
HOVER_TAIL_S = 4.0


def _rotating_segments(n_segments, seg_duration_s, speed_mps, angle_step_deg,
                        start_angle_deg=0.0):
    """
    Builds n_segments (vx, vy, 0.0, seg_duration_s) tuples that hold speed
    roughly constant while stepping the velocity DIRECTION by
    angle_step_deg each segment -- used to approximate a continuous turn
    (circular/figure_eight) out of the same discrete velocity-command
    segments every other profile uses, rather than a new motion primitive.
    """
    segments = []
    angle = math.radians(start_angle_deg)
    step = math.radians(angle_step_deg)
    for _ in range(n_segments):
        vx = speed_mps * math.cos(angle)
        vy = speed_mps * math.sin(angle)
        segments.append((round(vx, 4), round(vy, 4), 0.0, seg_duration_s))
        angle += step
    return segments


VELOCITY_PROFILES = {
    "gentle": [
        (2.0, 2.0, 0.0, 5.0),
        (1.0, 1.0, 0.0, 5.0),
    ],
    "moderate": [
        (4.0, 4.0, 0.0, 8.0),
        (2.0, 2.0, 0.0, 8.0),
    ],
    "aggressive": [
        (5.0, 0.0, 0.0, 6.0),
        (0.0, 5.0, 0.0, 6.0),
        (-3.0, -3.0, 0.0, 4.0),
    ],
    # Continuous one-directional turn: 8 segments x 45deg x 4s = one full
    # 360deg loop at constant ~3 m/s, so heading keeps changing throughout
    # instead of the straight-line-segment shape of the profiles above --
    # stresses dead-reckoning/fusion under continuous heading change.
    "circular": _rotating_segments(
        n_segments=8, seg_duration_s=4.0, speed_mps=3.0, angle_step_deg=45.0,
    ),
    # Two back-to-back 360deg loops in OPPOSITE rotational directions
    # (+45deg/segment then -45deg/segment), continuing from the heading
    # the first loop ends on so the direction reversal at the midpoint is
    # just another 45deg step, not a velocity discontinuity. Stresses a
    # sustained heading-rotation REVERSAL, distinct from "circular"'s
    # one-directional turning.
    "figure_eight": (
        _rotating_segments(
            n_segments=8, seg_duration_s=4.0, speed_mps=3.0, angle_step_deg=45.0,
            start_angle_deg=0.0,
        )
        + _rotating_segments(
            n_segments=8, seg_duration_s=4.0, speed_mps=3.0, angle_step_deg=-45.0,
            start_angle_deg=45.0 * 8,
        )
    ),
    # The control case the maneuvering profiles above are missing:
    # sustained, simple, straight-line motion with no direction changes
    # at all, over a longer single segment than any other profile, to see
    # how drift/detection behave with zero maneuvering noise.
    "sustained_cruise": [
        (3.0, 0.0, 0.0, 28.0),
    ],
}


def find_mavproxy_out_port():
    out = subprocess.check_output(
        ["bash", "-c", "pgrep -f 'mavproxy.py' | head -1"]
    ).decode().strip()
    if not out:
        raise RuntimeError("mavproxy.py process not found")
    pid = out.splitlines()[0]
    lsof_out = subprocess.check_output(
        ["lsof", "-p", pid, "-a", "-i", "UDP"]
    ).decode()
    for line in lsof_out.splitlines():
        if "UDP" in line and "*:" in line:
            port = line.strip().split(":")[-1]
            return int(port)
    raise RuntimeError(f"Could not find MAVProxy UDP out-link port:\n{lsof_out}")


def tail_latest_global_position():
    """
    Return (t, lat, lon, relative_alt) of the most recent
    GLOBAL_POSITION_INT row in flight_log.csv, or None.
    """
    try:
        with open(FLIGHT_LOG, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    for line in reversed(lines):
        parts = line.rstrip("\n").split(",")
        if len(parts) < 9:
            continue
        if parts[1] == "GLOBAL_POSITION_INT":
            try:
                t = float(parts[0])
                lat = float(parts[2])
                lon = float(parts[3])
                relative_alt = float(parts[5])
                return (t, lat, lon, relative_alt)
            except ValueError:
                continue
    return None


def wait_for(predicate, timeout_s, poll_s=0.5, description=""):
    start = time.time()
    last = None
    while time.time() - start < timeout_s:
        last = tail_latest_global_position()
        if last is not None and predicate(last):
            return last
        time.sleep(poll_s)
    print(f"  TIMEOUT waiting for: {description}. Last sample: {last}")
    return last


def main():
    profile_name = sys.argv[1] if len(sys.argv) > 1 else "gentle"
    if profile_name not in VELOCITY_PROFILES:
        print(f"Unknown profile '{profile_name}'. Choices: {list(VELOCITY_PROFILES)}")
        sys.exit(1)
    profile = VELOCITY_PROFILES[profile_name]
    print(f"Using velocity profile '{profile_name}': {profile}")

    print("Locating MAVProxy out-link UDP port...")
    port = find_mavproxy_out_port()
    print(f"  -> sending commands to 127.0.0.1:{port}")

    conn = mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}", source_system=250, source_component=1
    )

    def send_heartbeat():
        conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0,
        )

    def set_mode_guided():
        conn.mav.command_long_send(
            TARGET_SYSTEM, TARGET_COMPONENT,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            4,  # ArduCopter GUIDED mode number
            0, 0, 0, 0, 0,
        )

    def arm(force=False):
        conn.mav.command_long_send(
            TARGET_SYSTEM, TARGET_COMPONENT,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1,  # arm
            21196 if force else 0,
            0, 0, 0, 0, 0,
        )

    def takeoff(alt):
        conn.mav.command_long_send(
            TARGET_SYSTEM, TARGET_COMPONENT,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, alt,
        )

    def send_velocity(vx, vy, vz):
        type_mask = 0b0000111111000111  # use vx,vy,vz only
        conn.mav.set_position_target_local_ned_send(
            0, TARGET_SYSTEM, TARGET_COMPONENT,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            0, 0, 0,          # position (ignored)
            vx, vy, vz,       # velocity, m/s, NED
            0, 0, 0,          # accel (ignored)
            0, 0,             # yaw, yaw_rate (ignored)
        )

    for _ in range(5):
        send_heartbeat()
        time.sleep(0.1)

    print("Setting mode to GUIDED...")
    for _ in range(5):
        set_mode_guided()
        time.sleep(0.3)

    print("Arming...")
    for _ in range(5):
        arm(force=False)
        time.sleep(0.3)
    time.sleep(2)
    sample = tail_latest_global_position()
    print(f"  latest GLOBAL_POSITION_INT sample = {sample}")

    print(f"Commanding takeoff to {TAKEOFF_ALT_M}m...")
    for _ in range(5):
        takeoff(TAKEOFF_ALT_M)
        time.sleep(0.3)

    start_alt_sample = tail_latest_global_position()
    print(f"  pre-takeoff sample: {start_alt_sample}")

    print("Waiting for relative_alt to reach target...")
    sample = wait_for(
        lambda s: s[3] >= TAKEOFF_ALT_M - 2.0,
        timeout_s=30, poll_s=1.0,
        description=f"relative_alt >= {TAKEOFF_ALT_M - 2.0}m",
    )
    print(f"  sample after climb wait: {sample}")

    if sample is None or sample[3] < TAKEOFF_ALT_M - 2.0:
        print("ALTITUDE DID NOT CLIMB -- diagnosing before continuing.")
        for _ in range(5):
            set_mode_guided()
            time.sleep(0.2)
        print("Retrying arm with force + takeoff...")
        for _ in range(5):
            arm(force=True)
            time.sleep(0.3)
        for _ in range(5):
            takeoff(TAKEOFF_ALT_M)
            time.sleep(0.3)
        sample = wait_for(
            lambda s: s[3] >= TAKEOFF_ALT_M - 2.0,
            timeout_s=30, poll_s=1.0,
            description=f"relative_alt >= {TAKEOFF_ALT_M - 2.0}m (retry)",
        )
        print(f"  sample after retry climb wait: {sample}")
        if sample is None or sample[3] < TAKEOFF_ALT_M - 2.0:
            print("FATAL: could not reach target altitude after retry. Aborting.")
            sys.exit(1)

    print("Waiting for altitude to stabilize near target...")
    stable_start = time.time()
    stable_count = 0
    while time.time() - stable_start < 20:
        s = tail_latest_global_position()
        if s and abs(s[3] - TAKEOFF_ALT_M) <= ALT_TOLERANCE_M:
            stable_count += 1
            if stable_count >= 3:
                print(f"  Altitude stable: {s}")
                break
        else:
            stable_count = 0
        time.sleep(1.0)

    pos_before = tail_latest_global_position()
    print(f"Position before translational motion: {pos_before}")

    segments = list(profile) + [(0.0, 0.0, 0.0, HOVER_TAIL_S)]
    for vx, vy, vz, duration in segments:
        print(f"Sending velocity command vx={vx}, vy={vy}, vz={vz} for {duration}s...")
        t_end = time.time() + duration
        while time.time() < t_end:
            send_velocity(vx, vy, vz)
            time.sleep(0.2)
        print(f"  position: {tail_latest_global_position()}")

    pos_after = tail_latest_global_position()
    print(f"Position after motion profile: {pos_after}")

    # Single before/after point samples are a coincidence-prone way to
    # judge motion (a sample can land inside the initial ramp-up lag).
    # Instead, scan every GLOBAL_POSITION_INT row logged since pos_before
    # and take the max displacement from the start point -- this is what
    # the orchestrator/pipeline will also do independently from the CSV.
    if pos_before is not None:
        try:
            with open(FLIGHT_LOG, "r") as f:
                lines = f.readlines()
            lat0, lon0 = pos_before[1], pos_before[2]
            max_disp_m = 0.0
            for line in lines:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 4 or parts[1] != "GLOBAL_POSITION_INT":
                    continue
                try:
                    t = float(parts[0])
                    lat, lon = float(parts[2]), float(parts[3])
                except ValueError:
                    continue
                if t < pos_before[0]:
                    continue
                dlat_m = (lat - lat0) * 111320.0
                dlon_m = (lon - lon0) * 111320.0 * 0.8137  # approx cos(-35.4deg)
                disp = (dlat_m ** 2 + dlon_m ** 2) ** 0.5
                max_disp_m = max(max_disp_m, disp)
            print(f"Max displacement from pre-motion position: {max_disp_m:.2f} m")
            print(f"REAL TRANSLATIONAL MOTION CONFIRMED: {max_disp_m > 1.0}")
        except FileNotFoundError:
            print("WARNING: could not re-read flight log for motion confirmation.")

    print("Commanding gentle landing...")
    conn.mav.command_long_send(
        TARGET_SYSTEM, TARGET_COMPONENT,
        mavutil.mavlink.MAV_CMD_NAV_LAND, 0,
        0, 0, 0, 0, 0, 0, 0,
    )

    print("Flight sequence complete.")


if __name__ == "__main__":
    main()
