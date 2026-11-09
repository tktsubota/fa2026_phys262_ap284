"""Section 8: Metropolis Monte Carlo for an ideal gas under gravity

N particles in a unit square box, 0 <= x, z <= 1, with gravity pulling down
along z.  In units where the box height is L = 1, the Boltzmann weight of one
particle is exp(-eps z), with eps = beta m g L.  Linear response gives

    <z> - 1/2 = -eps/12 + O(eps^2),

and for an ideal gas the integral can be done exactly,

    <z> = 1/eps - 1/(e^eps - 1).

We sample this distribution with the Metropolis rule: pick a particle, propose
a random displacement, reject it outright if it leaves the box, and otherwise
accept it with probability min(1, exp(-eps dz)).

Part (a): runs from a uniform start at eps = 1, 4, 16, as a movie.  Each
column shows the box on top and the growing trace of z_cm below it, with the
exact and linear-response <z> marked.  The last frame is saved as a still for
the notes.

    python metropolis.py --part a

"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

BASE_DIR = Path(__file__).resolve().parent

plt.rcParams["font.size"] = 14

N = 100                 # particles
DELTA = 0.2             # proposed displacements are uniform in [-DELTA, DELTA]^2
N_SWEEPS = 300          # sweeps (N attempted moves each) per run
EPS_RUNS = [1, 4, 16]   # weak to strong gravity
FRAME_EVERY = 3         # one movie frame every FRAME_EVERY sweeps...
MOVIE_SECONDS = 20      # ...over about this many seconds


def linear_z(eps):
    """Linear response: <z> = 1/2 - eps/12."""
    return 0.5 - eps / 12


def exact_z(eps):
    """Exact ideal-gas average, <z> = 1/eps - 1/(e^eps - 1)."""
    return 1 / eps - 1 / np.expm1(eps)


def metropolis_step(r, rng, eps):
    """Attempt one Metropolis move.  Changes r in place."""
    a = rng.integers(N)
    new = r[a] + rng.uniform(-DELTA, DELTA, size=2)

    # Leaving the box costs infinite energy, so the acceptance probability is 0.
    if not np.all((new >= 0) & (new <= 1)):
        return
    dU = eps * (new[1] - r[a, 1])               # beta Delta U = eps dz
    if rng.random() < np.exp(-dU):
        r[a] = new


def run_sweeps(r0, rng, eps, n_sweeps=N_SWEEPS):
    """Run n_sweeps sweeps from r0.

    Returns z_cm after every sweep, and snapshots every FRAME_EVERY sweeps.
    """
    r = r0.copy()
    z_cm = np.empty(n_sweeps + 1)
    z_cm[0] = r[:, 1].mean()
    frames = [r.copy()]
    for s in range(n_sweeps):
        for _ in range(N):
            metropolis_step(r, rng, eps)
        z_cm[s + 1] = r[:, 1].mean()
        if (s + 1) % FRAME_EVERY == 0:
            frames.append(r.copy())
    return z_cm, frames


def _draw_box(ax):
    """The walls of the box and an arrow for gravity."""
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect("equal")
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k-", lw=1.5)
    ax.annotate("", xy=(1.18, 0.35), xytext=(1.18, 0.65),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(1.22, 0.5, "$g$", va="center")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")


def part_a():
    """Uniform start at eps = 1, 4, 16: movie of the box and z_cm trace."""
    out_dir = BASE_DIR / "part_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)

    runs = []
    for eps in EPS_RUNS:
        z_cm, frames = run_sweeps(rng.random((N, 2)), rng, eps)
        runs.append((frames, z_cm))
        print(f"eps = {eps:2d}: z_cm(final) = {z_cm[-1]:.4f}, "
              f"linear {linear_z(eps):+.4f}, exact {exact_z(eps):.4f}")

    fig, axes = plt.subplots(2, len(EPS_RUNS), figsize=(4 * len(EPS_RUNS), 7.5),
                             gridspec_kw=dict(height_ratios=[2, 1]))
    scatters, traces = [], []
    for col, eps in zip(axes.T, EPS_RUNS):
        _draw_box(col[0])
        col[0].set_title(f"$\\epsilon = {eps}$")
        scatters.append(col[0].scatter([], [], s=12, color="C0"))

        col[1].set_xlim(0, N_SWEEPS)
        col[1].set_ylim(min(0, linear_z(eps) - 0.05), 0.9)    # headroom for the legend
        col[1].axhline(exact_z(eps), color="k", ls="-", lw=1,
                       label=f"exact, {exact_z(eps):.3f}")
        col[1].axhline(linear_z(eps), color="k", ls="--", lw=1,
                       label=f"linear response, {linear_z(eps):.3f}")
        col[1].set_xlabel("sweep")
        col[1].set_ylabel("$z_{\\rm cm}$")
        col[1].legend(fontsize="x-small", loc="upper right")
        traces.append(col[1].plot([], [], lw=1, color="C0")[0])
    caption = fig.suptitle("")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def frame(k):
        sweep = k * FRAME_EVERY
        for sc, line, (frames, z_cm) in zip(scatters, traces, runs):
            sc.set_offsets(frames[k])
            line.set_data(np.arange(sweep + 1), z_cm[:sweep + 1])
        caption.set_text(f"$N = {N}$, uniform start, sweep {sweep}")
        return scatters + traces + [caption]

    n_frames = len(runs[0][0])
    fps = max(1, round(n_frames / MOVIE_SECONDS))
    anim = animation.FuncAnimation(fig, frame, frames=n_frames, interval=1000 // fps)
    anim.save(str(out_dir / "part_a_strong_gravity.gif"),
              writer=animation.PillowWriter(fps=fps))

    frame(n_frames - 1)                     # the last frame, as a still for the notes
    fig.savefig(out_dir / "part_a_strong_gravity.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Section 8 Metropolis.")
    parser.add_argument("--part", choices=["a"], required=True,
                        help="Which part of the problem to run.")
    args = parser.parse_args()
    {"a": part_a}[args.part]()


if __name__ == "__main__":
    main()
