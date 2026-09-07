"""
epv/log_report.py
------------------
CLI wrapper around epv.utils.log_quality_report: run this on any player log
BEFORE training to check how much of it actually lands inside the court.

Usage
-----
    python -m epv.log_report --log match0_log.json
"""

import argparse
import json

from epv.utils import load_frame_log, log_quality_report


def main():
    p = argparse.ArgumentParser(description="Quality report for a per-frame player log")
    p.add_argument("--log", required=True, help="path to a player log written by main.py --log-out")
    args = p.parse_args()

    rows = load_frame_log(args.log)
    report = log_quality_report(rows)
    print(json.dumps(report, indent=2))
    if report["warning"]:
        print("\n[ACTION NEEDED] Fix the homography/court-point mapping in main.py "
              "and re-run `main.py --log-out` before using this log for training.")


if __name__ == "__main__":
    main()
