"""
mavlink_reader.py  (v3 — root-caused the 50Hz request not holding)

Changes from v2:
  3. PERIODICALLY RE-ASSERTS message intervals instead of requesting them
     once at startup. Root cause (verified against source + a live SITL
     capture, not guessed):

       MAVProxy's main loop calls set_stream_rates() every tick, gated by
       `msg_period = mavutil.periodic_event(1.0/15)` (mavproxy.py:1534) --
       this fires every 15s *regardless of whether any setting changed*
       (mavproxy.py:1032-1049) and re-sends
       REQUEST_DATA_STREAM(MAV_DATA_STREAM_ALL, rate=<mavproxy's
       --streamrate, default 4>, start_stop=1) to the vehicle on every
       firing, unconditionally.

       ArduPilot's handle_request_data_stream() (GCS_Param.cpp:129) treats
       MAV_DATA_STREAM_ALL as "reinitialize every stream group's message
       intervals from streamRates[]" (GCS_Param.cpp:148-161), and
       initialise_message_intervals_for_stream() (GCS_Common.cpp:7119)
       unconditionally overwrites the per-message interval -- via
       set_ap_message_interval() -- for every ap_message in that group.
       RAW_IMU (STREAM_RAW_SENSORS) and ATTITUDE (STREAM_EXTRA1) are both
       members of groups this touches, so MAVProxy's periodic "ALL"
       request silently stomps our MAV_CMD_SET_MESSAGE_INTERVAL request
       back down to MAVProxy's 4Hz default -- with no ack/notification we
       can react to except the rate dropping.

       This was confirmed against a live SITL flight_log.csv: RAW_IMU ran
       at exactly 50Hz for ~11.5s after connecting, then dropped to a flat
       4.0Hz for the rest of the 129s flight (checked in 1s bins). One
       clean step, not decay or periodic sawtoothing -- because that
       build of this script requested the interval once and never again,
       so it only took MAVProxy's first 15s reset to kill it permanently.

       We do NOT have a way to stop this at the source from here: MAVProxy
       only skips the resend if its *own* --streamrate is -1, which is a
       MAVProxy launch/console setting, not something reachable over
       MAVLink from this script. The obvious MAVLink-side fix -- setting
       GCS_MAVLINK::Option::NOSTREAMOVERRIDE (GCS.h:545) via the link's
       SERIALx_OPTIONS parameter, so ArduPilot ignores REQUEST_DATA_STREAM
       entirely -- does not exist for this link: sim_vehicle.py's TCP
       master (tcp:127.0.0.1:5760, which MAVProxy relays to us) is
       SERIAL0, and SERIAL0 only exposes SERIAL0_BAUD/SERIAL0_PROTOCOL
       (AP_SerialManager.cpp:165-180, guarded by HAL_HAVE_SERIAL0_PARAMS)
       -- no SERIAL0_OPTIONS param exists to set that bit on. Confirmed
       live: SERIAL1_OPTIONS/SERIAL2_OPTIONS both answered a PARAM
       request; SERIAL0_OPTIONS never did, on repeated tries.

       So the correct fix *from this script* is re-assertion -- the user's
       original instinct was right in spirit. But a blind fixed-period
       poll is the wrong shape for it: MAVProxy's 15s timer runs on its
       own clock, with an unknown phase offset relative to when we
       connect, so a fixed reassert interval T has a worst case of nearly
       T seconds of stale 4Hz data after a reset lands just after a
       reassert. That's not hypothetical -- it's what a first attempt at
       this fix (7s period, "always 2+ reassertions per 15s window")
       actually produced when tested live for 33s against this SITL
       instance: two ~3-4s drops to a flat 4.0Hz (at ~t=10s and ~t=25s,
       15s apart, exactly the MAVProxy period), because the reset landed
       a few seconds after a reassert had just fired and nothing noticed
       until the next scheduled one.

       So instead of guessing at a period, we detect the actual symptom
       (RAW_IMU/ATTITUDE stop arriving at the requested rate) and react
       immediately: track time since the last RAW_IMU/ATTITUDE sample,
       and re-issue the interval request the moment that gap exceeds
       STALE_THRESHOLD_S, rate-limited by MIN_REASSERT_GAP_S so we don't
       hammer the link while a just-sent correction is still propagating.
       This closes the detection window to roughly one missed sample
       instead of up to one full poll period, and -- unlike blind
       periodic polling -- sends zero redundant commands while the rate
       is healthy, which matters because GCS_MAVLINK's own send budget is
       itself scheduler-constrained (see out_of_time() /
       min_loop_time_remaining_for_message_send_us() in GCS.cpp/GCS.h) --
       spending it on unneeded reassert commands would compete with the
       very data we're trying to receive.

       If you control the SITL launch command, the actually-clean fix is
       to stop MAVProxy from ever sending this: launch with
       `--mavproxy-args="streamrate=-1 streamrate2=-1"` on sim_vehicle.py.
       That removes the interference at the source instead of racing it.
       This script still re-asserts defensively even if you do that, in
       case another GCS/MAVProxy instance joins the link later.

Changes from v1:
  1. Explicitly REQUESTS message intervals via MAV_CMD_SET_MESSAGE_INTERVAL.
     v1 relied on whatever rate MAVProxy happened to forward (measured at
     ~4Hz), which was silently starving the dead-reckoning integration of
     resolution. RAW_IMU and ATTITUDE are now requested at 50Hz; GPS
     messages stay slower (10Hz) since GPS itself doesn't update faster
     than that on real receivers.
  2. Logs ATTITUDE (roll, pitch, yaw) -- previously received but discarded.
     dead_reckoning.py v2 needs this to compensate for gravity leaking into
     the horizontal accelerometer axes during tilt, which the audit
     identified as the dominant error source in v1's drift.

Usage: unchanged from v1.
    python3 mavlink_reader.py
"""

import csv
import time
import signal
from pymavlink import mavutil

CONNECTION_STRING = "udp:127.0.0.1:14550"
OUTPUT_CSV = "flight_log.csv"

# MAVProxy re-sends REQUEST_DATA_STREAM(ALL) every 15s (mavproxy.py:1534's
# periodic_event(1.0/15), unconditionally, see mavproxy.py:1032-1049), which
# silently overwrites our per-message intervals back to MAVProxy's 4Hz
# default (see module docstring for the full traced chain, and for why this
# is reactive rather than a fixed poll period).
STALE_THRESHOLD_S = 0.15   # ~3 expected periods at 50Hz before we call it stale
MIN_REASSERT_GAP_S = 1.0   # don't re-send faster than this while waiting for effect

# MAVLink message IDs (from common.xml) and target rates in Hz.
STREAM_RATES_HZ = {
    "RAW_IMU": (27, 50),
    "ATTITUDE": (30, 50),
    "GLOBAL_POSITION_INT": (33, 10),
    "GPS_RAW_INT": (24, 10),
}

CSV_HEADERS = [
    "timestamp",
    "msg_type",
    "lat", "lon", "alt", "relative_alt", "vx", "vy", "vz", "hdg",  # GLOBAL_POSITION_INT
    "xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro",             # RAW_IMU
    "roll", "pitch", "yaw",                                        # ATTITUDE (radians)
    "fix_type", "satellites_visible",                              # GPS_RAW_INT
]

running = True


def handle_sigint(sig, frame):
    global running
    print("\nStopping logger...")
    running = False


def request_message_intervals(conn):
    """
    Ask the autopilot to stream each message at the rate we actually need,
    instead of accepting whatever MAVProxy defaults to. interval_us=0
    would mean "as fast as possible"; we pass an explicit period instead
    so we get a known, predictable sample rate for the integration math.
    """
    for name, (msg_id, hz) in STREAM_RATES_HZ.items():
        interval_us = int(1_000_000 / hz)
        conn.mav.command_long_send(
            conn.target_system,
            conn.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,          # confirmation
            msg_id,     # param1: message ID
            interval_us,  # param2: interval in microseconds
            0, 0, 0, 0, 0,
        )
        print(f"Requested {name} (id {msg_id}) at {hz} Hz")
        time.sleep(0.05)  # avoid flooding the link with back-to-back commands


def main():
    signal.signal(signal.SIGINT, handle_sigint)

    print(f"Connecting to {CONNECTION_STRING} ...")
    conn = mavutil.mavlink_connection(CONNECTION_STRING)
    conn.wait_heartbeat()
    print(f"Heartbeat received from system {conn.target_system}, "
          f"component {conn.target_component}")

    request_message_intervals(conn)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()

        print(f"Logging to {OUTPUT_CSV} ... (Ctrl+C to stop)")
        msg_count = 0
        now = time.time()
        last_fast_msg = now    # last time we saw RAW_IMU or ATTITUDE
        last_reassert = now    # last time we (re-)requested intervals

        while running:
            now = time.time()
            if (now - last_fast_msg > STALE_THRESHOLD_S
                    and now - last_reassert > MIN_REASSERT_GAP_S):
                # RAW_IMU/ATTITUDE have gone quiet for longer than the
                # requested 50Hz period can explain -- almost certainly
                # MAVProxy's periodic REQUEST_DATA_STREAM(ALL) stomped our
                # interval back to its 4Hz default. Re-assert immediately.
                request_message_intervals(conn)
                last_reassert = time.time()

            msg = conn.recv_match(blocking=True, timeout=0.1)
            if msg is None:
                continue

            msg_type = msg.get_type()
            if msg_type in ("RAW_IMU", "ATTITUDE"):
                last_fast_msg = time.time()
            row = {h: "" for h in CSV_HEADERS}
            row["timestamp"] = time.time()
            row["msg_type"] = msg_type

            if msg_type == "GLOBAL_POSITION_INT":
                row.update({
                    "lat": msg.lat / 1e7,
                    "lon": msg.lon / 1e7,
                    "alt": msg.alt / 1000.0,
                    "relative_alt": msg.relative_alt / 1000.0,
                    "vx": msg.vx / 100.0,
                    "vy": msg.vy / 100.0,
                    "vz": msg.vz / 100.0,
                    "hdg": msg.hdg / 100.0,
                })

            elif msg_type == "RAW_IMU":
                row.update({
                    "xacc": msg.xacc,
                    "yacc": msg.yacc,
                    "zacc": msg.zacc,
                    "xgyro": msg.xgyro,
                    "ygyro": msg.ygyro,
                    "zgyro": msg.zgyro,
                })

            elif msg_type == "ATTITUDE":
                row.update({
                    "roll": msg.roll,    # radians
                    "pitch": msg.pitch,  # radians
                    "yaw": msg.yaw,      # radians
                })

            elif msg_type == "GPS_RAW_INT":
                row.update({
                    "fix_type": msg.fix_type,
                    "satellites_visible": msg.satellites_visible,
                })

            else:
                continue

            writer.writerow(row)
            msg_count += 1

            if msg_count % 200 == 0:
                print(f"  ... {msg_count} messages logged")

    print(f"Done. {msg_count} messages written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
