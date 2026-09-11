"""Section 1: Monte Carlo

Two estimators: the probability of rolling a 7 with two dice, and the value of
pi.  In each case we compare with the exact answer.

Run one part at a time (or all of them) from the command line, e.g.:

    python monte_carlo.py --part a
    python monte_carlo.py --part all
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
    """Roll two dice N times and count how often they sum to 7."""
    out_dir = _ensure_dir(BASE_DIR / "part_a")
    np.random.seed(1234)

    N = 10000
    rolls = [] # TODO
    sevens = [] # TODO

    # Running estimate after each roll.
    running = [] # TODO

    fig, ax = plt.subplots()
    ax.plot(running)
    ax.axhline(1/6, color="k", ls="--", label="exact, $1/6$")
    ax.set_xlabel("number of two-dice rolls")
    ax.set_ylabel("fraction summing to 7")
    ax.set_title(f"$N={N}$: estimate {sevens.mean():.4f}")
    ax.legend()
    fig.savefig(out_dir / "part_a_running.png")
    plt.close(fig)


def part_b():
    """Estimate pi from the fraction of points landing in a quarter circle."""
    out_dir = _ensure_dir(BASE_DIR / "part_b") 
    np.random.seed(1234)

    N = 10000
    points = [] # TODO
    inside = [] # TODO

    fig, ax = plt.subplots()
    ax.plot(points[inside, 0], points[inside, 1], ".", markersize=2)
    ax.plot(points[~inside, 0], points[~inside, 1], ".", markersize=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"$N={N}$: estimate {4 * inside.mean():.4f}")
    fig.savefig(out_dir / "part_b_scatter.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Section 1 Monte Carlo -- run one part, or all of them."
    )
    parser.add_argument(
        "--part", choices=["a", "b", "all"], required=True,
        help="Which part of the problem to run.",
    )
    args = parser.parse_args()

    dispatch = {"a": part_a, "b": part_b}

    if args.part == "all":
        for func in dispatch.values():
            func()
    else:
        dispatch[args.part]()


if __name__ == "__main__":
    main()
