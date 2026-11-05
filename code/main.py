"""Run the WHOLE project with no intervention:  python code/main.py

    python code/main.py            full run (20k subsample, 5-fold x 3 repeats)
    python code/main.py --quick    smoke test (small, ~minutes)
    python code/main.py --force    recompute even if cached results exist

Results are cached per experiment in results/<name>.json, so an interrupted
run resumes where it stopped.
"""
import os
import sys

if "--quick" in sys.argv:
    os.environ["PR_QUICK"] = "1"
if "--full" in sys.argv:
    os.environ["PR_FULL"] = "1"      # all 260k rows (slow models skipped)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import eda
import report
import track1_baselines
import track2_features
import track3_methods


def main():
    if "--force" in sys.argv:
        for p in common.RES.glob("t*.json"):
            p.unlink()
    common.log(f"START quick={common.QUICK} n={common.N_SUB} folds={common.N_SPLITS}x{common.N_REPEATS} jobs={common.N_JOBS}")
    eda.run()
    X, y = common.get_subsample()
    common.log("track 1: baselines")
    track1_baselines.run(X, y)
    common.log("track 2: feature extraction")
    track2_features.run(X, y)
    common.log("track 3: modified methods + MLP")
    track3_methods.run(X, y)
    report.run()
    common.log("DONE")


if __name__ == "__main__":
    main()
