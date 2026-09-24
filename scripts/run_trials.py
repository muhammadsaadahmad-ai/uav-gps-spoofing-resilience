"""
run_trials.py

Orchestrates 3 velocity profiles x 2 attack profiles = 6 end-to-end
trials (flight -> dead_reckoning -> spoof_injector -> detector), each
with a fresh flight, saving trial-suffixed CSVs and a results_summary.csv.

Reuses the existing scripts as subprocesses (mavlink_reader.py,
flight_driver.py, dead_reckoning.py, spoof_injector.py, detector.py)
rather than reimplementing their logic. Each of those scripts writes
fixed-name output files, so this script renames them to a trial-specific
suffix right after each stage completes.

Usage:
    python3 run_trials.py

Assumes the standard ArduPilot repo layout this script lives in
(<ardupilot_root>/ArduCopter/mavlink/run_trials.py, with
<ardupilot_root>/Tools/autotest/sim_vehicle.py alongside it) -- paths
below are derived from this file's own location, not hardcoded to any
one machine, so a clone anywhere still works. Set ARDUPILOT_ROOT to
override if this script is ever relocated out of that layout.
"""

import csv
import os
import re
import shutil
import signal
import subprocess
import time

MAVLINK_DIR = os.path.dirname(os.path.abspath(__file__))
ARDUCOPTER_DIR = os.path.dirname(MAVLINK_DIR)
ARDUPILOT_ROOT = os.environ.get("ARDUPILOT_ROOT", os.path.dirname(ARDUCOPTER_DIR))
SIM_VEHICLE = os.path.join(ARDUPILOT_ROOT, "Tools", "autotest", "sim_vehicle.py")

VELOCITY_PROFILES = ["gentle", "moderate", "aggressive"]
ATTACK_PROFILES = {
    "slow_drift": {"start_fraction": 0.3, "drift_rate": 0.3},
    "fast_drift": {"start_fraction": 0.6, "drift_rate": 1.2},
}

MAX_FLIGHT_RETRIES = 2
MIN_DISPLACEMENT_M = 3.0
LANDING_POLL_TIMEOUT_S = 60
LANDING_SETTLE_S = 2.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- SITL health

def sitl_alive():
    for pattern in ["arducopter", "mavproxy.py"]:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        if r.returncode != 0 or not r.stdout.strip():
            return False
    return True


def respawn_sitl():
    log("SITL appears unresponsive -- respawning via sim_vehicle.py")
    for pattern in ["sim_vehicle.py", "mavproxy.py", "arducopter"]:
        subprocess.run(["pkill", "-f", pattern])
    time.sleep(5)
    logf = open(os.path.join(MAVLINK_DIR, "sitl_respawn.log"), "a")
    subprocess.Popen(
        ["python3", SIM_VEHICLE, "-v", "ArduCopter", "--", "--streamrate=50"],
        cwd=ARDUCOPTER_DIR, stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    start = time.time()
    while time.time() - start < 120:
        if sitl_alive():
            log("SITL processes back up; giving it 20s to finish EKF/GPS init...")
            time.sleep(20)
            return True
        time.sleep(3)
    log("FATAL: SITL respawn did not come up within timeout.")
    return False


# ---------------------------------------------------------------- flight + log

def tail_last_global_position(csv_path):
    try:
        with open(csv_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        parts = line.rstrip("\n").split(",")
        if len(parts) < 6 or parts[1] != "GLOBAL_POSITION_INT":
            continue
        try:
            return (float(parts[0]), float(parts[2]), float(parts[3]), float(parts[5]))
        except ValueError:
            continue
    return None


def run_one_flight(velocity_profile):
    """
    Starts mavlink_reader.py, runs flight_driver.py <profile>, waits for
    touchdown, stops the reader cleanly. Returns (ok, flight_log_path)
    where flight_log_path is 'flight_log.csv' in MAVLINK_DIR (still at
    its default name -- caller renames it).
    """
    log_path = os.path.join(MAVLINK_DIR, "flight_log.csv")
    if os.path.exists(log_path):
        os.remove(log_path)

    reader_stdout = open(os.path.join(MAVLINK_DIR, "reader_stdout.log"), "w")
    reader = subprocess.Popen(
        ["python3", "mavlink_reader.py"], cwd=MAVLINK_DIR,
        stdout=reader_stdout, stderr=subprocess.STDOUT,
    )

    # give the reader a moment to bind + get its first heartbeat, and log
    # a few seconds of pre-motion baseline
    time.sleep(5)

    driver = subprocess.run(
        ["python3", "flight_driver.py", velocity_profile],
        cwd=MAVLINK_DIR, capture_output=True, text=True, timeout=180,
    )
    driver_log_path = os.path.join(MAVLINK_DIR, f"driver_stdout_{velocity_profile}.log")
    with open(driver_log_path, "w") as f:
        f.write(driver.stdout)
        f.write(driver.stderr)

    driver_ok = driver.returncode == 0
    if not driver_ok:
        log(f"  flight_driver.py exited with code {driver.returncode}, see {driver_log_path}")

    # wait for touchdown: relative_alt near 0 and settled, or timeout
    log("  waiting for touchdown before stopping logger...")
    start = time.time()
    settled_since = None
    while time.time() - start < LANDING_POLL_TIMEOUT_S:
        sample = tail_last_global_position(log_path)
        if sample is not None and sample[3] < 0.5:
            if settled_since is None:
                settled_since = time.time()
            elif time.time() - settled_since > LANDING_SETTLE_S:
                break
        else:
            settled_since = None
        time.sleep(1.0)

    time.sleep(1.0)  # small tail margin
    reader.send_signal(signal.SIGINT)
    try:
        reader.wait(timeout=15)
    except subprocess.TimeoutExpired:
        reader.kill()
        reader.wait(timeout=5)
    reader_stdout.close()

    return driver_ok, log_path


def measure_max_displacement(flight_log_path):
    """
    Max displacement (m, approx flat-earth) of GLOBAL_POSITION_INT
    samples from the first sample in the file.
    """
    rows = []
    with open(flight_log_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["msg_type"] == "GLOBAL_POSITION_INT" and row["lat"]:
                rows.append((float(row["timestamp"]), float(row["lat"]), float(row["lon"])))
    if len(rows) < 2:
        return 0.0, 0.0
    rows.sort()
    lat0, lon0 = rows[0][1], rows[0][2]
    import math
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    max_disp = 0.0
    for _, lat, lon in rows:
        dx = (lon - lon0) * meters_per_deg_lon
        dy = (lat - lat0) * meters_per_deg_lat
        max_disp = max(max_disp, math.hypot(dx, dy))
    duration = rows[-1][0] - rows[0][0]
    return max_disp, duration


def get_flight_with_retries(velocity_profile):
    for attempt in range(1, MAX_FLIGHT_RETRIES + 2):
        if not sitl_alive():
            if not respawn_sitl():
                continue
        log(f"  flight attempt {attempt} for profile '{velocity_profile}'...")
        ok, log_path = run_one_flight(velocity_profile)
        if not ok:
            log("  flight_driver.py reported failure, retrying...")
            continue
        disp, duration = measure_max_displacement(log_path)
        log(f"  max displacement={disp:.2f}m over {duration:.1f}s")
        if disp >= MIN_DISPLACEMENT_M:
            return log_path, disp, duration
        log(f"  displacement below {MIN_DISPLACEMENT_M}m threshold -- treating as "
            f"hover-only / failed motion, retrying...")
    return None, 0.0, 0.0


# ---------------------------------------------------------------- pipeline stages

def run_dead_reckoning(flight_log_path):
    r = subprocess.run(
        ["python3", "dead_reckoning.py", flight_log_path],
        cwd=MAVLINK_DIR, capture_output=True, text=True, timeout=120,
    )
    return r.stdout, r.returncode == 0


def run_spoof_injector(gps_gt_path, start_fraction, drift_rate):
    r = subprocess.run(
        ["python3", "spoof_injector.py", gps_gt_path, str(start_fraction), str(drift_rate)],
        cwd=MAVLINK_DIR, capture_output=True, text=True, timeout=60,
    )
    return r.stdout, r.returncode == 0


def run_detector(spoofed_path, dr_output_path):
    r = subprocess.run(
        ["python3", "detector.py", spoofed_path, dr_output_path],
        cwd=MAVLINK_DIR, capture_output=True, text=True, timeout=60,
    )
    return r.stdout, r.returncode == 0


# ---------------------------------------------------------------- stdout parsing

def parse_dead_reckoning(stdout):
    out = {}
    m = re.search(r"mean dt: [\d.]+s \(([\d.]+) Hz\)", stdout)
    out["imu_sample_rate_hz"] = float(m.group(1)) if m else None
    m = re.search(r"mean:\s+([\d.]+) m", stdout)
    out["dr_drift_mean_m"] = float(m.group(1)) if m else None
    m = re.search(r"max:\s+([\d.]+) m", stdout)
    out["dr_drift_max_m"] = float(m.group(1)) if m else None
    m = re.search(r"final:\s+([\d.]+) m", stdout)
    out["dr_drift_final_m"] = float(m.group(1)) if m else None
    return out


def parse_spoof_injector(stdout):
    out = {}
    m = re.search(r"Attack starts at t=([\d.]+)s", stdout)
    out["attack_start_t_s"] = float(m.group(1)) if m else None
    return out


def parse_detector(stdout):
    out = {}
    m = re.search(r"TP=(\d+) FP=(\d+) FN=(\d+) TN=(\d+)", stdout)
    if m:
        out["tp"], out["fp"], out["fn"], out["tn"] = (int(x) for x in m.groups())
    m = re.search(r"Precision:\s+([\d.]+)", stdout)
    out["precision"] = float(m.group(1)) if m else None
    m = re.search(r"Recall:\s+([\d.]+)", stdout)
    out["recall"] = float(m.group(1)) if m else None
    m = re.search(r"F1 score:\s+([\d.]+)", stdout)
    out["f1"] = float(m.group(1)) if m else None
    m = re.search(r"Identical to naive elapsed-time rule\? (\w+)", stdout)
    out["identical_to_naive_rule"] = (m.group(1) == "True") if m else None
    m = re.search(r"Detection latency: ([\d.]+) s", stdout)
    out["detection_latency_s"] = float(m.group(1)) if m else None
    if "Detector never caught the attack" in stdout:
        out["detection_latency_s"] = None
    m = re.search(r"False positives: none\.", stdout)
    if m:
        out["fp_note"] = "none"
    else:
        m = re.search(r"False positives: (\d+), at t-attack_start \(s\) = \[(-?[\d.]+) \.\. (-?[\d.]+)\]", stdout)
        if m:
            out["fp_note"] = f"{m.group(1)} FPs, offset range [{m.group(2)}..{m.group(3)}]s from attack start"
        else:
            out["fp_note"] = "unparsed"
    return out


# ---------------------------------------------------------------- main

def main():
    results = []
    trial_logs = []

    for vp in VELOCITY_PROFILES:
        for ap_name, ap in ATTACK_PROFILES.items():
            trial = f"{vp}_{ap_name}"
            log(f"=== TRIAL {trial} ===")

            flight_log_path, disp, duration = get_flight_with_retries(vp)
            if flight_log_path is None:
                log(f"  TRIAL {trial} FAILED: could not get real motion after retries. Skipping.")
                results.append({"velocity_profile": vp, "attack_profile": ap_name,
                                 "status": "FAILED_NO_MOTION"})
                continue

            trial_flight_log = os.path.join(MAVLINK_DIR, f"flight_log_{trial}.csv")
            shutil.copy(flight_log_path, trial_flight_log)
            log(f"  motion confirmed: {disp:.2f}m displacement, {duration:.1f}s flight -> saved {trial_flight_log}")

            dr_stdout, dr_ok = run_dead_reckoning(trial_flight_log)
            if not dr_ok:
                log(f"  dead_reckoning.py FAILED for {trial}:\n{dr_stdout}")
                results.append({"velocity_profile": vp, "attack_profile": ap_name,
                                 "status": "FAILED_DEAD_RECKONING"})
                continue
            dr_metrics = parse_dead_reckoning(dr_stdout)

            dr_out = os.path.join(MAVLINK_DIR, "dead_reckoning_output.csv")
            gt_out = os.path.join(MAVLINK_DIR, "gps_ground_truth.csv")
            trial_dr_out = os.path.join(MAVLINK_DIR, f"dead_reckoning_output_{trial}.csv")
            trial_gt_out = os.path.join(MAVLINK_DIR, f"gps_ground_truth_{trial}.csv")
            shutil.move(dr_out, trial_dr_out)
            shutil.move(gt_out, trial_gt_out)

            spoof_stdout, spoof_ok = run_spoof_injector(
                trial_gt_out, ap["start_fraction"], ap["drift_rate"]
            )
            if not spoof_ok:
                log(f"  spoof_injector.py FAILED for {trial}:\n{spoof_stdout}")
                results.append({"velocity_profile": vp, "attack_profile": ap_name,
                                 "status": "FAILED_SPOOF_INJECTOR"})
                continue
            spoof_metrics = parse_spoof_injector(spoof_stdout)

            spoofed_out = os.path.join(MAVLINK_DIR, "spoofed_gps.csv")
            trial_spoofed_out = os.path.join(MAVLINK_DIR, f"spoofed_gps_{trial}.csv")
            shutil.move(spoofed_out, trial_spoofed_out)

            det_stdout, det_ok = run_detector(trial_spoofed_out, trial_dr_out)
            if not det_ok:
                log(f"  detector.py FAILED for {trial}:\n{det_stdout}")
                results.append({"velocity_profile": vp, "attack_profile": ap_name,
                                 "status": "FAILED_DETECTOR"})
                continue
            det_metrics = parse_detector(det_stdout)

            det_out = os.path.join(MAVLINK_DIR, "detector_output.csv")
            trial_det_out = os.path.join(MAVLINK_DIR, f"detector_output_{trial}.csv")
            shutil.move(det_out, trial_det_out)

            row = {
                "velocity_profile": vp,
                "attack_profile": ap_name,
                "status": "OK",
                "flight_duration_s": round(duration, 2),
                "max_displacement_m": round(disp, 2),
                **dr_metrics,
                **spoof_metrics,
                **det_metrics,
            }
            results.append(row)
            trial_logs.append((trial, dr_stdout, spoof_stdout, det_stdout))
            log(f"  TRIAL {trial} DONE: P={row.get('precision')} R={row.get('recall')} "
                f"F1={row.get('f1')} FP={row.get('fp')} note={row.get('fp_note')}")

    # ---- write results_summary.csv
    fieldnames = [
        "velocity_profile", "attack_profile", "status", "flight_duration_s",
        "imu_sample_rate_hz", "dr_drift_mean_m", "dr_drift_max_m", "dr_drift_final_m",
        "attack_start_t_s", "tp", "fp", "fn", "tn", "precision", "recall", "f1",
        "detection_latency_s", "identical_to_naive_rule", "fp_note", "max_displacement_m",
    ]
    summary_path = os.path.join(MAVLINK_DIR, "results_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    log(f"Wrote {summary_path}")

    # ---- full stdout logs for later inspection
    with open(os.path.join(MAVLINK_DIR, "trial_stage_logs.txt"), "w") as f:
        for trial, dr_stdout, spoof_stdout, det_stdout in trial_logs:
            f.write(f"\n===== {trial} : dead_reckoning.py =====\n{dr_stdout}\n")
            f.write(f"\n===== {trial} : spoof_injector.py =====\n{spoof_stdout}\n")
            f.write(f"\n===== {trial} : detector.py =====\n{det_stdout}\n")

    log("ALL TRIALS COMPLETE")
    for row in results:
        log(str(row))


if __name__ == "__main__":
    main()
