"""HW3 Problem 2: Microcanonical, canonical, and grand canonical ensembles.

Run one part at a time (or all of them) from the command line, e.g.:

    python ensembles.py --part e
    python ensembles.py --part all

The simulations for parts (g) to (i) are written for you: chain.py is the
coupled-oscillator engine, laid out like md.py, and md.py is the HW2 solution
with a pressure measurement added.  Each simulation runs once and saves its
trajectory to data/; later calls read it back.  Fill in the TODOs below.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import chain
import md

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Part (g): coupled oscillators
CHAIN_N = 256              # oscillators
CHAIN_LAM = 0.2            # strength of the nonlinear coupling
CHAIN_DT = 0.05            # time step
CHAIN_T_MAX = 5.0e4        # how long to run for (under a minute)
CHAIN_SAMPLE_EVERY = 100   # record a frame after this many steps

# Parts (h) and (i): dilute gas
GAS_N = 100                # particles
GAS_L = 100.0              # side of the box
GAS_SPACING = 0.95         # initial lattice spacing, as in HW2
GAS_DT = 0.005             # time step
GAS_T_MAX = 2000.0         # how long to run for (about seven minutes)
GAS_SAMPLE_EVERY = 20      # record a frame after this many steps


def _ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def chain_run():
    """The part (g) trajectory: simulated the first time, read from disk after.

    All of the energy E = N starts in oscillator 0.  Returns a dict with t (S,),
    positions (S, N), velocities (S, N), K (S,), U (S,).
    """
    path = DATA_DIR / "chain.h5"
    if not path.exists():
        positions, velocities = chain.single_oscillator(CHAIN_N, float(CHAIN_N))
        print("Simulating the oscillator chain to t = %g" % CHAIN_T_MAX)
        run = chain.simulate(positions, velocities, CHAIN_LAM, CHAIN_DT,
                             int(round(CHAIN_T_MAX / CHAIN_DT)), CHAIN_SAMPLE_EVERY)
        chain.save_run(path, run)
    return chain.load_run(path)


def gas_run():
    """The part (h) and (i) trajectory: simulated the first time, read from disk after.

    Returns a dict with t (S,), positions (S, N, 2), velocities (S, N, 2),
    K (S,), U (S,), and P (S,), the pressure on the walls.
    """
    path = DATA_DIR / "gas.h5"
    if not path.exists():
        positions, velocities = md.compressed_lattice(GAS_N, GAS_L, GAS_SPACING)
        print("Simulating the gas to t = %g" % GAS_T_MAX)
        run = md.simulate(positions, velocities, GAS_L, GAS_DT,
                          int(round(GAS_T_MAX / GAS_DT)), GAS_SAMPLE_EVERY)
        md.save_run(path, run)
    return md.load_run(path)


def part_e():
    out_dir = _ensure_dir(BASE_DIR / "part_e")

    # TODO


def part_f():
    out_dir = _ensure_dir(BASE_DIR / "part_f")

    # TODO


def part_g():
    out_dir = _ensure_dir(BASE_DIR / "part_g")

    run = chain_run()
    t, q, p = run["t"], run["positions"], run["velocities"]

    # TODO


def part_h():
    out_dir = _ensure_dir(BASE_DIR / "part_h")

    run = gas_run()
    t, P = run["t"], run["P"]
    positions, velocities = run["positions"], run["velocities"]

    # TODO


def part_i():
    out_dir = _ensure_dir(BASE_DIR / "part_i")

    run = gas_run()
    t, P = run["t"], run["P"]
    positions, velocities = run["positions"], run["velocities"]

    # TODO


def main():
    """Command line entry: DO NOT MODIFY"""
    parser = argparse.ArgumentParser(
        description="HW3 Ensembles -- run one part, or all of them."
    )
    parser.add_argument(
        "--part", choices=["e", "f", "g", "h", "i", "all"], required=True,
        help="Which part of the problem to run.",
    )
    args = parser.parse_args()

    dispatch = {"e": part_e, "f": part_f, "g": part_g,
                "h": part_h, "i": part_i}

    if args.part == "all":
        for func in dispatch.values():
            func()
    else:
        dispatch[args.part]()


if __name__ == "__main__":
    main()
