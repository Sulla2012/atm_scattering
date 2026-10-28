"""Build NEATM lookup tables on a regular GHz grid (50-1500 GHz) for use with frequency-dependent emissivity."""

import argparse

import numpy as np

from atm.config import Config
from atm.functions.emissivity import ghzGridToLambda
from atm.models import NEATM


def main() -> None:
    """Parse the CLI and build the tables into the NEATM table directory."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fmin", type=float, default=50.0, help="GHz")
    ap.add_argument("--fmax", type=float, default=1500.0, help="GHz")
    ap.add_argument("--df", type=float, default=25.0, help="GHz")
    ap.add_argument("--tmin", type=float, default=100.0, help="K")
    ap.add_argument("--tmax", type=float, default=1200.0, help="K")
    ap.add_argument(
        "--tstep",
        type=float,
        default=10.0,
        help="K (coarser than the IR tables: ~1 s of CPU per integration)",
    )
    ap.add_argument(
        "--alpha-step",
        type=float,
        default=Config.tableParameterLimits["alpha"][1],
        help="rad",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="table directory (default: the NEATM table directory)",
    )
    ap.add_argument("--threads", type=int, default=Config.threads)
    ap.add_argument(
        "--debug", action="store_true", help="3 wavelengths, coarse T_ss/alpha grid"
    )
    a = ap.parse_args()
    if a.debug:
        a.df, a.tstep, a.alpha_step = (a.fmax - a.fmin) / 2, 50.0, np.pi / 8
    model = NEATM() if a.out is None else NEATM(tableDir=a.out)
    model.buildLambdaTables(
        ghzGridToLambda(a.fmin, a.fmax, a.df),
        tRange=[a.tmin, a.tmax],
        tStep=a.tstep,
        alphaStep=a.alpha_step,
        threads=a.threads,
        verbose=True,
    )


if __name__ == "__main__":
    main()
