"""HW1: Random Walks

This script is based on exercise 2.5 from Sethna, Entropy, Order Parameters, and Complexity, 2014. Adapted from a Jupyter notebook by Jim Sethna modified by Vinny Manoharan.

Run one part at a time (or all of them) from the command line, e.g.:

    python random_walks.py --part a
    python random_walks.py --part all
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

LOW, HIGH = -0.5, 0.5
RMS_STEP_A = 1.0 / (2.0 * np.sqrt(3.0))

BASE_DIR = Path(__file__).resolve().parent


def random_walk(N, d):
    """Trajectory of an N-step walk in d dimensions, shape (N, d)."""
    steps = np.random.uniform(LOW, HIGH, size=(N, d))
    return np.cumsum(steps, axis=0)


def endpoints(W, N, d):
    """Endpoints of W independent N-step walks in d dimensions, shape (W, d)."""
    steps = np.random.uniform(LOW, HIGH, size=(N, W, d))
    return np.sum(steps, axis=0)


def _ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def part_a():
    """Trajectories (1D and 2D), plus the N vs. 100N distance-scaling check."""
    out_dir = _ensure_dir(BASE_DIR / "part_a")
    np.random.seed(101)

    N_1d, n_walks_1d = 10000, 10
    Ns_2d, n_walks_2d = [10, 1000, 100000], 5

    walks_1d = [random_walk(N_1d, 1) for _ in range(n_walks_1d)]
    fig, ax = plt.subplots()
    for walk in walks_1d:
        ax.plot(walk[:, 0])
    ax.set_xlabel("step")
    ax.set_ylabel("x")
    ax.set_title(f"1D random walks, $N={N_1d}$")
    fig.savefig(out_dir / "part_a_1d_trajectories.png")
    plt.close(fig)

    walks_2d = {N: [random_walk(N, 2) for _ in range(n_walks_2d)] for N in Ns_2d}
    fig, axes = plt.subplots(1, len(Ns_2d), figsize=(4 * len(Ns_2d), 4))
    for ax, N in zip(axes, Ns_2d):
        for walk in walks_2d[N]:
            ax.plot(walk[:, 0], walk[:, 1])
        ax.set_aspect("equal")
        ax.set_title(f"2D random walks, $N={N}$")
    fig.tight_layout()
    fig.savefig(out_dir / "part_a_2d_trajectories.png")
    plt.close(fig)

    # Does multiplying N by 100 roughly increase the net distance by 10?
    N_small, N_large = 1000, 100000
    endpoints_small = np.array([walk[-1] for walk in walks_2d[N_small]])
    endpoints_large = np.array([walk[-1] for walk in walks_2d[N_large]])

    dist_small = np.linalg.norm(endpoints_small, axis=1).mean()
    dist_large = np.linalg.norm(endpoints_large, axis=1).mean()
    ratio = dist_large / dist_small
    print(
        f"N={N_small} -> mean distance {dist_small:.3f}; "
        f"N={N_large} -> mean distance {dist_large:.3f}; "
        f"ratio = {ratio:.2f} (diffusive prediction: sqrt(100) = 10)"
    )


def part_b():
    """Scatter of endpoints for N=1 vs N=10, illustrating emergent circular symmetry."""
    out_dir = _ensure_dir(BASE_DIR / "part_b")
    np.random.seed(103)

    W, d = 10000, 2
    Ns = [1, 10]

    fig, ax = plt.subplots()
    for N in sorted(Ns, reverse=True):  # largest spread first, so smaller N draws on top
        x, y = endpoints(W, N, d).T
        ax.plot(x, y, ".", label=f"$N={N}$", markersize=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Endpoints of $W={W}$ 2D random walks")
    ax.legend()
    fig.savefig(out_dir / "part_b_scatter.png")
    plt.close(fig)


def part_c():
    """Numerical experiment demonstrating CLT convergence of the endpoint distribution.

    (This problem is open-ended in the assignment -- students choose their own
    N values, sample size, and comparison method. This solution uses histograms of
    W=10000 one-dimensional walks at N=1,2,3,5 overlaid with the predicted Gaussian,
    which is one reasonable way to do it.)
    """
    out_dir = _ensure_dir(BASE_DIR / "part_c")
    np.random.seed(105)

    W, d = 10000, 1
    Ns = [1, 2, 3, 5]

    for N in Ns:
        data = endpoints(W, N, d)[:, 0]
        sigma = np.sqrt(N) * RMS_STEP_A

        fig, ax = plt.subplots()
        ax.hist(data, bins=50, range=(-3 * sigma, 3 * sigma), density=True)
        x = np.linspace(-3 * sigma, 3 * sigma, 200)
        gauss = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-x**2 / (2 * sigma**2))
        ax.plot(x, gauss, "r")
        ax.set_title(f"$N={N}$")
        fig.savefig(out_dir / f"part_c_hist_N{N}.png")
        plt.close(fig)


def main():
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
