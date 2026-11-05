"""HW5 Problem 3: Ising model (numerical).

Run one part at a time (or all of them) from the command line, e.g.:

    python ising.py --part c
    python ising.py --part all

Fill in the TODOs: the Metropolis step in `update`, and the plots in `part_c`
and `part_d`.  The functions that set up the lattice (`init_state`), run many
sweeps while recording E and M after each one (`measure`), and make the movies
(`get_movie_and_measure`) are already written.
"""

import argparse
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"

L = 50                              # side of the lattice
N = L * L                           # number of spins
N_SWEEPS = 1000                     # sweeps per run (about ~30 seconds per run)
TEMPERATURES = [2.0, 2.269, 3.0]    # temperatures to run
INITIAL_CONDITIONS = ["random", "allup", "alldown"]   # starting configurations for part (d)
SEED = 0                            # for random number generator

# Movie parameters for part (b): N_FRAMES frames, T_GAP sweeps apart, so a
# movie covers N_FRAMES * T_GAP = N_SWEEPS sweeps.
N_FRAMES = 50
T_GAP = N_SWEEPS // N_FRAMES
FPS = 10                            # frames per second of the GIF

# ---------------------------------------------------------------------------
# 1.  The state of the system.
# ---------------------------------------------------------------------------

class State:
    def __init__(self, spins, E, m):
        self.spins = spins # L x L array of spin values
        self.E = E   # total energy
        self.m = m   # total magnetization


def init_state(L, init="random", rng=None):
    """An L x L lattice of Ising spins, each +1 or -1.

    `init` is "random", "allup", or "alldown". 
    """
    if rng is None:
        rng = np.random.default_rng()
    if init == "random":
        spins = rng.choice([-1, 1], size=(L, L))
    elif init == "allup":
        spins = np.ones((L, L), dtype=int)
    elif init == "alldown":
        spins = -np.ones((L, L), dtype=int)
    else:
        raise ValueError("Unknown init")
    state = State(spins, 0, 0)
    state.E = compute_energy(state)
    state.m = compute_magnetization(state)
    return state


# ---------------------------------------------------------------------------
# 2.  The Metropolis algorithm: one attempted flip, then one sweep.
# ---------------------------------------------------------------------------

def update(state, L, T, rng):
    x, y = rng.integers(low=0, high=L, size=2)
    s = state.spins[x, y]

    # TODO - Metropolis step
    
    # TODO - make sure to update the change in total energy to state.E and the change in total magnetization in state.m

    return state


def update_sweep(state, L, T, rng):
    """One sweep: L^2 attempted flips, so on average each spin is tried once."""
    for _ in range(L * L):
        state = update(state, L, T, rng)
    return state


# ---------------------------------------------------------------------------
# 3.  Measuring the energy and the magnetization.
# ---------------------------------------------------------------------------

def compute_energy(state):
    """Total energy, E = -sum_<ij> s_i s_j, with J = 1.
    """
    E = 0
    L = state.spins.shape[0]
    for x in range(L):
        for y in range(L):
            E += -state.spins[x, y] * (state.spins[(x + 1) % L, y] + state.spins[x, (y + 1) % L])
    return E


def compute_magnetization(state):
    """Total magnetization, m = sum_i s_i, between -L^2 and L^2."""
    L = state.spins.shape[0]
    return np.sum(state.spins)


def get_movie_and_measure(state, L, T, rng, t_gap, n_frame, in_fps, out_name, dpi=120):
    """Run the simulation while making a movie of it, saved as a GIF at `out_name`.

    Records n_frame snapshots, spaced by t_gap sweeps, in addition to the
    original input spin configuration, so the movie covers n_frame * t_gap
    sweeps.  Also records the total energy and magnetization at each frame;
    returns them as an array `data` of shape (n_frame + 1, 2), with E in
    column 0 and M in column 1.  The state is updated in place.
    """
    total_frames = int(n_frame) + 1
    data = np.empty((total_frames, 2), dtype=float)

    # A 660 x 600 pixel figure: the extra width leaves room for the colorbar
    # beside the square lattice, and constrained_layout keeps the two-line
    # title clear of both.
    fig, ax = plt.subplots(figsize=(660 / dpi, 600 / dpi), dpi=dpi, constrained_layout=True)

    im = ax.imshow(state.spins, cmap="bwr", vmin=-1, vmax=1,
                   origin="lower", extent=[1, L, 1, L],
                   interpolation="nearest", aspect="equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(im, ax=ax, ticks=[-1, 1], shrink=0.8)
    cbar.set_label("spin")
    ttl = ax.set_title("", fontsize=10)
    t_gap_int = int(t_gap)

    frames = []
    for k in range(total_frames):
        if k > 0:
            for _ in range(t_gap_int):
                update_sweep(state, L, T, rng)

        # sweeps completed so far (k = 0 is the initial frame)
        sweeps_done = k * t_gap_int

        # measure: `update` keeps state.E and state.m current
        data[k, 0] = state.E
        data[k, 1] = state.m

        # update image + title, then render the figure into a frame
        im.set_data(state.spins)
        ttl.set_text("T = %g, sweep %d\nE/N = %.2f, M/N = %+.3f"
                     % (T, sweeps_done, data[k, 0] / L**2, data[k, 1] / L**2))
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("P", palette=Image.ADAPTIVE))
    plt.close(fig)

    # disposal=2 makes every frame replace the previous one completely.
    # Without it, Pillow stores only what changed between frames, and the old
    # title text shows through the new one.
    frames[0].save(str(out_name), save_all=True, append_images=frames[1:],
                   duration=int(round(1000 / in_fps)), loop=0, disposal=2)
    return data


def measure(state, L, T, n_sweeps, rng):
    """Run n_sweeps sweeps at temperature T, recording the total E and M after each one.

    Like get_movie_and_measure without the movie, and measuring after every
    sweep.  Returns (E, M), arrays of shape (n_sweeps + 1,); entry 0 is the
    starting configuration and entry k is the configuration after k sweeps.
    The state is updated in place.
    """
    E = np.zeros(n_sweeps + 1)
    M = np.zeros(n_sweeps + 1)
    E[0], M[0] = state.E, state.m
    for k in range(1, n_sweeps + 1):
        state = update_sweep(state, L, T, rng)
        E[k], M[k] = state.E, state.m      # kept current by `update`
    return E, M


# ---------------------------------------------------------------------------
# 4.  The parts of the problem to plot.
# ---------------------------------------------------------------------------

def part_b():
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(1, len(TEMPERATURES), figsize=(3.4 * len(TEMPERATURES), 3.6))
    for ax, T in zip(axes, TEMPERATURES):
        print("T = %g: running %d sweeps and making a movie..." % (T, N_FRAMES * T_GAP), flush=True)
        state = init_state(L, init="random", rng=rng)
        gif = OUT_DIR / ("part_b_movie_T%g.gif" % T)
        get_movie_and_measure(state, L, T, rng, T_GAP, N_FRAMES, FPS, gif)
        print("  final E/N = %.3f, M/N = %.3f  (movie: output/%s)"
              % (state.E / N, state.m / N, gif.name))
        ax.imshow(state.spins, cmap="bwr", vmin=-1, vmax=1, origin="lower")  # same orientation as the movie
        ax.set_title("$T = %g$" % T)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "part_b_snapshots.png", dpi=150)
    plt.close(fig)


def part_c():
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    # TODO


def part_d():
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    # TODO


def main():
    """Command line entry: DO NOT MODIFY"""
    parser = argparse.ArgumentParser(
        description="HW5 Ising model -- run one part, or all of them."
    )
    parser.add_argument(
        "--part", choices=["b", "c", "d", "all"], required=True,
        help="Which part of the problem to run.",
    )
    args = parser.parse_args()

    dispatch = {"b": part_b, "c": part_c, "d": part_d}

    if args.part == "all":
        for func in dispatch.values():
            func()
    else:
        dispatch[args.part]()


if __name__ == "__main__":
    main()
