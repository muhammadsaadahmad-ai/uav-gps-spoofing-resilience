"""
run_fusion_trial.py

Thin CLI wrapper around fusion.py's run_trial(vp, ap) for one trial, so
run_trials_full.py can invoke it as a subprocess under the ardupilot-env
interpreter (which has pandas/numpy) without importing fusion/pandas into
the orchestrator's own process.

Usage: python3 run_fusion_trial.py <velocity_profile> <attack_name>
Prints a single line "FUSION_RESULT_JSON:<json>" with fusion.run_trial's
return dict, or exits nonzero with the exception on stderr on failure.
"""
import json
import sys

import fusion


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 run_fusion_trial.py <velocity_profile> <attack_name>")
        sys.exit(1)
    vp, ap = sys.argv[1], sys.argv[2]
    row = fusion.run_trial(vp, ap)
    print("FUSION_RESULT_JSON:" + json.dumps(row))


if __name__ == "__main__":
    main()
