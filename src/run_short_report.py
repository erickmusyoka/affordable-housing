"""
Rebuild every notebook in this package from the cleaned tables in data/, in order (about 8 minutes).

    uv run src/run_short_report.py        run the notebooks (default)
    uv run src/run_short_report.py all    also re-run the cleaning scripts, which need the raw downloads described in DATA.md under data/raw/

The cleaned tables (data/interim, data/processed) are included, so the default needs no raw data. The notebooks write their tables to outputs/tables and
figures to outputs/figures. Run one copy at a time.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
CLEANERS = ["clean_nhpd.py", "clean_lihtc.py", "clean_chas.py", "clean_population_projections.py", "clean_acs_age.py", "clean_home_rent_limits.py", "clean_nlihc_gap.py"]
NOTEBOOKS = ["notebooks/01_eda/01_coverage.py", "notebooks/01_eda/02_signal_vs_noise.py", "notebooks/01_eda/03_trends.py", "notebooks/01_eda/04_need_today_ahni.py", "notebooks/01_eda/05_relationships.py", "notebooks/04_scenarios/01_stock_flow.py", "notebooks/05_preservation/01_preservation_model.py",
             "notebooks/05_preservation/02_lihtc_exit.py", "notebooks/05_preservation/03_rent_gap.py", "notebooks/05_preservation/04_mortgage_exit.py", "notebooks/05_preservation/06_price_growth.py", "notebooks/05_preservation/07_state_hindcast.py", "notebooks/05_preservation/08_rescue_drift.py", "notebooks/06_combined/01_units_at_stake.py", "notebooks/06_combined/02_uncertainty_and_sensitivity.py", "notebooks/06_combined/03_external_benchmarks.py",
             "notebooks/07_homelessness/01_ahar_context.py", "notebooks/08_actions/01_what_can_be_done.py"]


def run(cmd, label):
    start = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(f"{label:58s} {'ok' if r.returncode == 0 else 'FAILED':7s} {time.time() - start:6.1f}s", flush=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
        sys.exit(f"{label} failed")


def main():
    if not (ROOT / "data/processed/panel.csv").exists():
        sys.exit("data/processed/panel.csv is missing: it is included in the package and is built by the full project's build_panel.py")
    if "all" in sys.argv[1:]:
        for c in CLEANERS:
            run([sys.executable, str(HERE / c)], c)
    for nb in NOTEBOOKS:
        run([sys.executable, str(HERE / "notebook_tools.py"), nb], nb)


if __name__ == "__main__":
    main()
