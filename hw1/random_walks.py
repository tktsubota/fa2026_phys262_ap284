"""HW1: Random Walks -- starter code.

This script is based on exercise 2.5 from Sethna, Entropy, Order Parameters, and Complexity, 2014. Adapted from a Jupyter notebook by Jim Sethna modified by Vinny Manoharan.

Run one part at a time (or all of them) from the command line, e.g.:

    python random_walks.py --part a
    python random_walks.py --part all

Fill in the TODOs below.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent


def _ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def part_a():
    out_dir = _ensure_dir(BASE_DIR / "part_a")

    np.random.seed(101)

    # TODO: generate your 1D random walk trajectories.

    fig, ax = plt.subplots()

    # TODO: plot onto `ax` here.

    fig.savefig(out_dir / "part_a_1d_trajectories.png")
    plt.close(fig)

    # TODO: generate your 2D random walk trajectories.

    fig, ax = plt.subplots()

    # TODO: plot onto `ax` here.

    fig.savefig(out_dir / "part_a_2d_trajectories.png")
    plt.close(fig)

    # TODO: also answer -- does multiplying N by 100 roughly increase the
    # net distance by 10?


def part_b():
    out_dir = _ensure_dir(BASE_DIR / "part_b")

    # TODO


def part_c():
    out_dir = _ensure_dir(BASE_DIR / "part_c")

    # TODO


def main():
    """Command line entry: DO NOT MODIFY"""
    parser = argparse.ArgumentParser(
        description="HW1 Random Walks -- run one part, or all of them."
    )
    parser.add_argument(
        "--part", choices=["a", "b", "c", "all"], required=True,
        help="Which part of the problem to run.",
    )
    args = parser.parse_args()

    dispatch = {"a": part_a, "b": part_b, "c": part_c}

    if args.part == "all":
        for func in dispatch.values():
            func()
    else:
        dispatch[args.part]()


if __name__ == "__main__":
    main()
