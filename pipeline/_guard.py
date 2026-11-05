"""Run guard: modules that import this exit unless --allow-run is passed."""

import sys


def check_run_guard(script_name: str) -> None:
    """Abort with banner unless --allow-run is in argv."""
    if "--allow-run" not in sys.argv:
        banner = (
            "\n"
            f"  {script_name}: not run. Pass --allow-run to execute:\n"
            f"    python3 {script_name} --allow-run\n"
        )
        print(banner)
        sys.exit(0)
