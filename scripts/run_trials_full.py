"""
run_trials_full.py

Extends run_trials.py's pipeline (flight -> dead_reckoning -> spoof_injector
-> detector) with fusion.py, to the full 6 velocity profiles x 4 attack
types = 24 trial matrix. Launches SITL itself via the direct
arducopter-binary + daemon-mavproxy method (NOT sim_vehicle.py's own
orchestration, which was shown to fail in this sandboxed/headless
environment) and reuses it (with respawn-on-failure) across all 24 trials,
same as the original run_trials.py did for its 6.

All subprocesses are launched with an EXPLICIT interpreter/binary path
(ARDUPILOT_ENV/bin/...) because the bare `python3` / `mavproxy.py` on PATH
here resolve to a pyenv shim with no pymavlink/pandas/mavproxy installed.

Usage:
    /home/saad/ardupilot-env/bin/python3 run_trials_full.py
"""

import csv
import json
import os
import re
import shutil
import signal
import subprocess
import time

MAVLINK_DIR = os.path.dirname(os.path.abspath(__file__))
ARDUPILOT_ENV = "/home/saad/ardupilot-env"
PYTHON = os.path.join(ARDUPILOT_ENV, "bin", "python3")
MAVPROXY = os.path.join(ARDUPILOT_ENV, "bin", "mavproxy.py")
ARDUCOPTER_BIN = "/home/saad/ardupilot/build/sitl/bin/arducopter"

VELOCITY_PROFILES = ["gentle", "moderate", "aggressive",
                      "circular", "figure_eight", "sustained_cruise"]

# attack name -> (attack_start_fraction, drift_rate_mps, attack_type arg to spoof_injector.py)
# slow_drift/fast_drift reuse the exact params the original 6 trials used.
# sudden_jump/intermittent use spoof_injector.py's own defaults
# (ATTACK_START_FRACTION=0.4, DRIFT_RATE_MPS=0.6) explicitly, per task spec.
ATTACK_PROFILES = {
    "slow_drift":   (0.3, 0.3, "gradual"),
    "fast_drift":   (0.6, 1.2, "gradual"),
    "sudden_jump":  (0.4, 0.6, "sudden_jump"),
    "intermittent": (0.4, 0.6, "intermittent"),
}

MAX_FLIGHT_RETRIES = 3
MIN_DISPLACEMENT_M = 3.0
LANDING_POLL_TIMEOUT_S = 90
LANDING_SETTLE_S = 2.0
SITL_READY_TIMEOUT_S = 90


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- SITL health

def sitl_alive():
    for pattern in [ARDUCOPTER_BIN, "mavproxy.py"]:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        if r.returncode != 0 or not r.stdout.strip():
            return False
    return True


def wait_for_gps_fix(timeout_s=SITL_READY_TIMEOUT_S):
    """
    Runs mavlink_reader.py briefly against the live SITL/MAVProxy link and
    polls flight_log.csv for a GLOBAL_POSITION_INT row -- a much stronger
    readiness signal than "processes exist", since EKF/GPS init can lag
    process startup by many seconds.
    """
    probe_log = os.path.join(MAVLINK_DIR, "flight_log.csv")
    if os.path.exists(probe_log):
        os.remove(probe_log)
    reader_stdout = open(os.path.join(MAVLINK_DIR, "sitl_ready_probe.log"), "w")
    reader = subprocess.Popen(
        [PYTHON, "mavlink_reader.py"], cwd=MAVLINK_DIR,
        stdout=reader_stdout, stderr=subprocess.STDOUT,
    )
    start = time.time()
    got_fix = False
    try:
        while time.time() - start < timeout_s:
            if os.path.exists(probe_log):
                with open(probe_log, "r") as f:
                    for line in f:
                        parts = line.rstrip("\n").split(",")
                        if len(parts) >= 6 and parts[1] == "GLOBAL_POSITION_INT":
                            try:
                                lat = float(parts[2])
                                if lat != 0.0:
                                    got_fix = True
                                    break
                            except ValueError:
                                pass
            if got_fix:
                break
            time.sleep(2.0)
    finally:
        reader.send_signal(signal.SIGINT)
        try:
            reader.wait(timeout=10)
        except subprocess.TimeoutExpired:
            reader.kill()
            reader.wait(timeout=5)
        reader_stdout.close()
    return got_fix


def launch_sitl_fresh():
    log("Launching SITL fresh: direct arducopter binary + daemon mavproxy "
        "(sim_vehicle.py orchestration is NOT used -- confirmed broken in "
        "this environment last session).")
    for pattern in [ARDUCOPTER_BIN, "mavproxy.py", "sim_vehicle.py"]:
        subprocess.run(["pkill", "-f", pattern])
    time.sleep(3)

    arducopter_log = open(os.path.join(MAVLINK_DIR, "arducopter_manual.log"), "w")
    subprocess.Popen(
        [ARDUCOPTER_BIN, "--model", "+", "--speedup", "1", "--slave", "0",
         "--sim-address=127.0.0.1", "-I0"],
        cwd=MAVLINK_DIR, stdout=arducopter_log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(4)

    mavproxy_log = open(os.path.join(MAVLINK_DIR, "mavproxy_daemon.log"), "w")
    subprocess.Popen(
        [MAVPROXY, "--daemon", "--out", "127.0.0.1:14550",
         "--master", "tcp:127.0.0.1:5760", "--sitl", "127.0.0.1:5501"],
        cwd=MAVLINK_DIR, stdout=mavproxy_log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    start = time.time()
    while time.time() - start < 60:
        if sitl_alive():
            break
        time.sleep(2)
    else:
        log("FATAL: arducopter/mavproxy processes did not come up within 60s.")
        return False

    log("Processes up; waiting for a real GPS fix (GLOBAL_POSITION_INT with nonzero lat)...")
    if wait_for_gps_fix():
        log("SITL ready: got a live GPS fix.")
        return True
    log("FATAL: no GPS fix within timeout after process launch.")
    return False


def respawn_sitl():
    log("SITL appears unresponsive -- respawning.")
    return launch_sitl_fresh()


def ensure_sitl_ready():
    if sitl_alive():
        log("SITL processes already running -- verifying with a live GPS-fix probe before reuse.")
        if wait_for_gps_fix(timeout_s=30):
            log("Existing SITL confirmed healthy; reusing it.")
            return True
        log("Existing SITL processes present but not producing a GPS fix -- respawning.")
    return launch_sitl_fresh()


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
    log_path = os.path.join(MAVLINK_DIR, "flight_log.csv")
    if os.path.exists(log_path):
        os.remove(log_path)

    reader_stdout = open(os.path.join(MAVLINK_DIR, "reader_stdout.log"), "w")
    reader = subprocess.Popen(
        [PYTHON, "mavlink_reader.py"], cwd=MAVLINK_DIR,
        stdout=reader_stdout, stderr=subprocess.STDOUT,
    )

    time.sleep(5)

    try:
        driver = subprocess.run(
            [PYTHON, "flight_driver.py", velocity_profile],
            cwd=MAVLINK_DIR, capture_output=True, text=True, timeout=180,
        )
        driver_ok = driver.returncode == 0
        driver_stdout, driver_stderr = driver.stdout, driver.stderr
    except subprocess.TimeoutExpired as e:
        driver_ok = False
        driver_stdout = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        driver_stderr = f"TIMEOUT after {e.timeout}s"

    driver_log_path = os.path.join(MAVLINK_DIR, f"driver_stdout_{velocity_profile}.log")
    with open(driver_log_path, "w") as f:
        f.write(driver_stdout or "")
        f.write("\n--- stderr ---\n")
        f.write(driver_stderr or "")

    if not driver_ok:
        log(f"  flight_driver.py exited non-zero / timed out, see {driver_log_path}")

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

    time.sleep(1.0)
    reader.send_signal(signal.SIGINT)
    try:
        reader.wait(timeout=15)
    except subprocess.TimeoutExpired:
        reader.kill()
        reader.wait(timeout=5)
    reader_stdout.close()

    return driver_ok, log_path


def measure_max_displacement(flight_log_path):
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
            log("  flight_driver.py reported failure -- respawning SITL and retrying...")
            respawn_sitl()
            continue
        disp, duration = measure_max_displacement(log_path)
        log(f"  max displacement={disp:.2f}m over {duration:.1f}s")
        if disp >= MIN_DISPLACEMENT_M:
            return log_path, disp, duration
        log(f"  displacement below {MIN_DISPLACEMENT_M}m threshold -- treating as "
            f"hover-only / failed motion, respawning SITL and retrying...")
        respawn_sitl()
    return None, 0.0, 0.0


# ---------------------------------------------------------------- pipeline stages

def run_dead_reckoning(flight_log_path):
    r = subprocess.run(
        [PYTHON, "dead_reckoning.py", flight_log_path],
        cwd=MAVLINK_DIR, capture_output=True, text=True, timeout=120,
    )
    return r.stdout + "\n" + r.stderr, r.returncode == 0


def run_spoof_injector(gps_gt_path, start_fraction, drift_rate, attack_type):
    r = subprocess.run(
        [PYTHON, "spoof_injector.py", gps_gt_path, str(start_fraction),
         str(drift_rate), attack_type],
        cwd=MAVLINK_DIR, capture_output=True, text=True, timeout=60,
    )
    return r.stdout + "\n" + r.stderr, r.returncode == 0


def run_detector(spoofed_path, dr_output_path):
    r = subprocess.run(
        [PYTHON, "detector.py", spoofed_path, dr_output_path],
        cwd=MAVLINK_DIR, capture_output=True, text=True, timeout=60,
    )
    return r.stdout + "\n" + r.stderr, r.returncode == 0


def run_fusion_trial(vp, ap_name):
    r = subprocess.run(
        [PYTHON, "run_fusion_trial.py", vp, ap_name],
        cwd=MAVLINK_DIR, capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        return None, r.stdout + "\n" + r.stderr
    for line in r.stdout.splitlines():
        if line.startswith("FUSION_RESULT_JSON:"):
            return json.loads(line[len("FUSION_RESULT_JSON:"):]), r.stdout
    return None, r.stdout + "\n" + r.stderr


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
        m2 = re.search(r"False positives: (\d+), at t-attack_start \(s\) = \[(-?[\d.]+) \.\. (-?[\d.]+)\]", stdout)
        if m2:
            out["fp_note"] = f"{m2.group(1)} FPs, offset range [{m2.group(2)}..{m2.group(3)}]s from attack start"
        else:
            out["fp_note"] = "unparsed"
    m = re.search(r"Jump-detector latch fired this run\? (\w+)(?: \(first at t=([\d.]+)\))?", stdout)
    if m:
        out["jump_alarm_fired"] = (m.group(1) == "True")
        out["jump_alarm_first_t_s"] = float(m.group(2)) if m.group(2) else None
    else:
        out["jump_alarm_fired"] = None
        out["jump_alarm_first_t_s"] = None
    m = re.search(r"calib max jump_delta=([\d.]+) m, threshold=([\d.]+) m", stdout)
    if m:
        out["calib_jump_max_m"] = float(m.group(1))
        out["jump_threshold_m"] = float(m.group(2))
    m = re.search(r"baseline slope_gap=([\d.-]+) m/s, k=([\d.]+), h=([\d.]+)", stdout)
    if m:
        out["cusum_baseline_gap_mps"] = float(m.group(1))
        out["cusum_h"] = float(m.group(3))
    return out


# ---------------------------------------------------------------- main

def main():
    log("Ensuring SITL is up before starting the 24-trial matrix...")
    if not ensure_sitl_ready():
        log("FATAL: could not get SITL into a ready state. Aborting entire run.")
        return

    results = []
    trial_logs = []
    failed_trials = []

    for vp in VELOCITY_PROFILES:
        for ap_name, (start_fraction, drift_rate, attack_type) in ATTACK_PROFILES.items():
            trial = f"{vp}_{ap_name}"
            log(f"=== TRIAL {trial} ===")

            flight_log_path, disp, duration = get_flight_with_retries(vp)
            if flight_log_path is None:
                log(f"  TRIAL {trial} FAILED: could not get real motion after retries. Skipping.")
                results.append({"velocity_profile": vp, "attack_type": ap_name,
                                 "status": "FAILED_NO_MOTION"})
                failed_trials.append((trial, "FAILED_NO_MOTION"))
                continue

            trial_flight_log = os.path.join(MAVLINK_DIR, f"flight_log_{trial}.csv")
            shutil.copy(flight_log_path, trial_flight_log)
            log(f"  motion confirmed: {disp:.2f}m displacement, {duration:.1f}s flight -> saved {trial_flight_log}")

            dr_stdout, dr_ok = run_dead_reckoning(trial_flight_log)
            if not dr_ok:
                log(f"  dead_reckoning.py FAILED for {trial}:\n{dr_stdout}")
                results.append({"velocity_profile": vp, "attack_type": ap_name,
                                 "status": "FAILED_DEAD_RECKONING"})
                failed_trials.append((trial, "FAILED_DEAD_RECKONING"))
                continue
            dr_metrics = parse_dead_reckoning(dr_stdout)

            dr_out = os.path.join(MAVLINK_DIR, "dead_reckoning_output.csv")
            gt_out = os.path.join(MAVLINK_DIR, "gps_ground_truth.csv")
            trial_dr_out = os.path.join(MAVLINK_DIR, f"dead_reckoning_output_{trial}.csv")
            trial_gt_out = os.path.join(MAVLINK_DIR, f"gps_ground_truth_{trial}.csv")
            shutil.move(dr_out, trial_dr_out)
            shutil.move(gt_out, trial_gt_out)

            spoof_stdout, spoof_ok = run_spoof_injector(
                trial_gt_out, start_fraction, drift_rate, attack_type
            )
            if not spoof_ok:
                log(f"  spoof_injector.py FAILED for {trial}:\n{spoof_stdout}")
                results.append({"velocity_profile": vp, "attack_type": ap_name,
                                 "status": "FAILED_SPOOF_INJECTOR"})
                failed_trials.append((trial, "FAILED_SPOOF_INJECTOR"))
                continue
            spoof_metrics = parse_spoof_injector(spoof_stdout)

            spoofed_out = os.path.join(MAVLINK_DIR, "spoofed_gps.csv")
            trial_spoofed_out = os.path.join(MAVLINK_DIR, f"spoofed_gps_{trial}.csv")
            shutil.move(spoofed_out, trial_spoofed_out)

            det_stdout, det_ok = run_detector(trial_spoofed_out, trial_dr_out)
            if not det_ok:
                log(f"  detector.py FAILED for {trial}:\n{det_stdout}")
                results.append({"velocity_profile": vp, "attack_type": ap_name,
                                 "status": "FAILED_DETECTOR"})
                failed_trials.append((trial, "FAILED_DETECTOR"))
                continue
            det_metrics = parse_detector(det_stdout)

            det_out = os.path.join(MAVLINK_DIR, "detector_output.csv")
            trial_det_out = os.path.join(MAVLINK_DIR, f"detector_output_{trial}.csv")
            shutil.move(det_out, trial_det_out)

            fusion_row, fusion_stdout = run_fusion_trial(vp, ap_name)
            if fusion_row is None:
                log(f"  fusion run_trial FAILED for {trial}:\n{fusion_stdout}")
                results.append({"velocity_profile": vp, "attack_type": ap_name,
                                 "status": "FAILED_FUSION"})
                failed_trials.append((trial, "FAILED_FUSION"))
                continue

            row = {
                "velocity_profile": vp,
                "attack_type": ap_name,
                "status": "OK",
                "flight_duration_s": round(duration, 2),
                "max_displacement_m": round(disp, 2),
                **dr_metrics,
                **spoof_metrics,
                **det_metrics,
                "trust_weighted_error_mean_m": fusion_row["trust_weighted_error_mean"],
                "trust_weighted_error_max_m": fusion_row["trust_weighted_error_max"],
                "hard_switch_error_mean_m": fusion_row["hard_switch_error_mean"],
                "gps_only_error_mean_m": fusion_row["gps_only_error_mean"],
                "dr_only_error_mean_m": fusion_row["dr_only_error_mean"],
                "trust_weighted_max_jump_m": fusion_row["trust_weighted_max_jump_m"],
                "hard_switch_max_jump_m": fusion_row["hard_switch_max_jump_m"],
                "pre_attack_trust_weighted_error_mean": fusion_row["pre_attack_trust_weighted_error_mean"],
                "pre_attack_gps_only_error_mean": fusion_row["pre_attack_gps_only_error_mean"],
            }
            results.append(row)
            trial_logs.append((trial, dr_stdout, spoof_stdout, det_stdout, fusion_stdout))
            log(f"  TRIAL {trial} DONE: P={row.get('precision')} R={row.get('recall')} "
                f"F1={row.get('f1')} FP={row.get('fp')} jump_fired={row.get('jump_alarm_fired')} "
                f"tw_err_mean={row.get('trust_weighted_error_mean_m'):.3f}m "
                f"gps_err_mean={row.get('gps_only_error_mean_m'):.3f}m")

    # ---- write results_summary_full.csv
    fieldnames = [
        "velocity_profile", "attack_type", "status", "flight_duration_s",
        "imu_sample_rate_hz", "max_displacement_m", "dr_drift_mean_m", "dr_drift_max_m",
        "attack_start_t_s", "tp", "fp", "fn", "tn", "precision", "recall", "f1",
        "detection_latency_s", "jump_alarm_fired",
        "trust_weighted_error_mean_m", "hard_switch_error_mean_m",
        "gps_only_error_mean_m", "trust_weighted_max_jump_m",
        # extras kept for the analysis pass, beyond the task's minimum column list
        "dr_drift_final_m", "identical_to_naive_rule", "fp_note",
        "jump_alarm_first_t_s", "calib_jump_max_m", "jump_threshold_m",
        "cusum_baseline_gap_mps", "cusum_h",
        "trust_weighted_error_max_m", "hard_switch_max_jump_m", "dr_only_error_mean_m",
        "pre_attack_trust_weighted_error_mean", "pre_attack_gps_only_error_mean",
    ]
    summary_path = os.path.join(MAVLINK_DIR, "results_summary_full.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    log(f"Wrote {summary_path}")

    with open(os.path.join(MAVLINK_DIR, "trial_stage_logs_full.txt"), "w") as f:
        for trial, dr_stdout, spoof_stdout, det_stdout, fusion_stdout in trial_logs:
            f.write(f"\n===== {trial} : dead_reckoning.py =====\n{dr_stdout}\n")
            f.write(f"\n===== {trial} : spoof_injector.py =====\n{spoof_stdout}\n")
            f.write(f"\n===== {trial} : detector.py =====\n{det_stdout}\n")
            f.write(f"\n===== {trial} : fusion =====\n{fusion_stdout}\n")

    log("ALL 24 TRIALS ATTEMPTED.")
    log(f"Succeeded: {len(results) - len(failed_trials)} / {len(results)}")
    if failed_trials:
        log(f"FAILED TRIALS: {failed_trials}")
    for row in results:
        log(str(row))


if __name__ == "__main__":
    main()
