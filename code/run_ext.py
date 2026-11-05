"""Extended experiments: architecture zoo (t4_) + nested-CV hyper-parameter tuning (t5_).

    python code/run_ext.py                 zoo, then tuning, then the combined report
    python code/run_ext.py --zoo           only the zoo
    python code/run_ext.py --tune          only tuning
    python code/run_ext.py --report        only rebuild tables/figures
    python code/run_ext.py --quick         smoke test (small data)
Results are cached in results/ext/*.json; logs in results/ext/progress.log.
"""
import os
import sys

if "--quick" in sys.argv:
    os.environ["PR_QUICK"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common

EXT = common.RES / "ext"
EXT.mkdir(exist_ok=True)
MAIN_RES = common.RES
common.RES = EXT                      # all caching + progress logging goes to results/ext

import report_ext
import tuning
import zoo


def main():
    do = [a for a in ("--zoo", "--tune", "--report") if a in sys.argv] or ["--zoo", "--tune", "--report"]
    common.log(f"EXT START quick={common.QUICK} n={common.N_SUB} folds={common.N_SPLITS}x{common.N_REPEATS} jobs={common.N_JOBS}")
    X, y = common.get_subsample()
    if "--zoo" in do:
        common.log("zoo: architectures with fixed hyper-parameters")
        common.run_all(zoo.jobs(), X, y)
    if "--tune" in do:
        common.log("tuning: nested CV grid searches")
        tuning.run(X, y)
    if "--report" in do:
        report_ext.run(MAIN_RES, EXT)
    common.log("EXT DONE")


if __name__ == "__main__":
    main()
