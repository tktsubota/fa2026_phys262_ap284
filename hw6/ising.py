"""HW6 Problem 4: Ising model in a field (numerical).

Run one part at a time (or all of them) from the command line, e.g.:

    python ising.py --part a
    python ising.py --part all

The simulation is written for you: it is the HW5 solution with a uniform
field H added, so that the energy is

    E = -sum_<ij> s_i s_j - H sum_i s_i      (J = k_B = 1).

Every function that runs the simulation (init_state, update, update_sweep,
measure, get_movie_and_measure) takes the field as a keyword argument H,
which defaults to 0, and state.E includes the field term.  Fill in the TODOs
below: the runs and plots for parts (a), (b), (c) and (f).  Save your figures
and movies to output/.
"""

import argparse
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"

L = 128                             # side of the lattice
N = L * L                           # number of spins
N_SWEEPS = 500                      # sweeps per run (about 2-3 minutes per run at L = 128)
T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))   # critical temperature at H = 0, about 2.269
SEED = 0                            # for random number generator

# Default movie parameters for get_movie_and_measure: N_FRAMES frames, T_GAP
# sweeps apart, so a movie covers N_FRAMES * T_GAP = N_SWEEPS sweeps.
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


def init_state(L, init="random", rng=None, H=0.0):
    """An L x L lattice of Ising spins, each +1 or -1, in a field H.

    `init` is "random", "allup", or "alldown".  The field only enters the
    stored energy state.E; use the same H here as in the runs that follow.
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
    state.E = compute_energy(state, H)
    state.m = compute_magnetization(state)
    return state


# ---------------------------------------------------------------------------
# 2.  The Metropolis algorithm: one attempted flip, then one sweep.
# ---------------------------------------------------------------------------

def update(state, L, T, rng, H=0.0):
    """Attempt to flip one randomly chosen spin in a field H, with periodic
    boundary conditions.
    """
    x, y = rng.integers(low=0, high=L, size=2)
    s = state.spins[x, y]

    # sum of the four nearest neighbors, wrapping around the edges
    neighbors = (state.spins[(x + 1) % L, y] + state.spins[(x - 1) % L, y]
                 + state.spins[x, (y + 1) % L] + state.spins[x, (y - 1) % L])

    # Metropolis rule.  Flipping s reverses -J s * neighbors - H s (J = 1), so
    # the energy changes by 2 s (neighbors + H).
    cost = 2 * s * (neighbors + H)
    if cost <= 0 or rng.random() < np.exp(-cost / T):
        state.spins[x, y] = -s
        state.E += cost                 # the flip raised the energy by cost
        state.m -= 2 * s                # and changed sum_i s_i by -2s

    return state


def update_sweep(state, L, T, rng, H=0.0):
    """One sweep: L^2 attempted flips, so on average each spin is tried once."""
    for _ in range(L * L):
        state = update(state, L, T, rng, H)
    return state


# ---------------------------------------------------------------------------
# 3.  Measuring the energy and the magnetization.
# ---------------------------------------------------------------------------

def compute_energy(state, H=0.0):
    """Total energy, E = -sum_<ij> s_i s_j - H sum_i s_i, with J = 1.

    Every bond joins a site to the neighbor on its right or the one above it,
    so summing only those two bonds for every site counts each bond exactly
    once.  A fully polarized lattice gives E = -2 L^2 at H = 0.
    """
    E = 0
    L = state.spins.shape[0]
    for x in range(L):
        for y in range(L):
            E += -state.spins[x, y] * (state.spins[(x + 1) % L, y] + state.spins[x, (y + 1) % L])
            E += -H * state.spins[x, y]
    return E


def compute_magnetization(state):
    """Total magnetization, m = sum_i s_i, between -L^2 and L^2."""
    L = state.spins.shape[0]
    return np.sum(state.spins)


def get_movie_and_measure(state, L, T, rng, t_gap, n_frame, in_fps, out_name, H=0.0, dpi=120):
    """Run the simulation while making a movie of it, saved as a GIF at `out_name`.

    Records n_frame snapshots, spaced by t_gap sweeps, in addition to the
    original input spin configuration, so the movie covers n_frame * t_gap
    sweeps.  Also records the total energy and magnetization at each frame;
    returns them as an array `data` of shape (n_frame + 1, 2), with E in
    column 0 and M in column 1.  The state is updated in place.  The last
    frame is also saved as a PNG, next to the GIF with the same name.
    """
    total_frames = int(n_frame) + 1
    data = np.empty((total_frames, 2), dtype=float)

    # A 660 x 600 pixel figure: the extra width leaves room for the colorbar
    # beside the square lattice.  The margins are fixed, so the axes stay put
    # from frame to frame while the title text changes.
    fig, ax = plt.subplots(figsize=(660 / dpi, 600 / dpi), dpi=dpi)
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.88)

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
                update_sweep(state, L, T, rng, H)

        # sweeps completed so far (k = 0 is the initial frame)
        sweeps_done = k * t_gap_int

        # measure: `update` keeps state.E and state.m current
        data[k, 0] = state.E
        data[k, 1] = state.m

        print("\r  %3d%%" % (100 * k // n_frame), end="", flush=True)   # progress

        # update image + title, then render the figure into a frame
        im.set_data(state.spins)
        ttl.set_text("T = %g, H = %g, sweep %d\nE/N = %.2f, M/N = %+.3f"
                     % (T, H, sweeps_done, data[k, 0] / L**2, data[k, 1] / L**2))
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("P", palette=Image.ADAPTIVE))
    fig.savefig(Path(out_name).with_suffix(".png"), dpi=dpi)   # the last frame
    plt.close(fig)
    print()

    # disposal=2 makes every frame replace the previous one completely.
    # Without it, Pillow stores only what changed between frames, and the old
    # title text shows through the new one.
    frames[0].save(str(out_name), save_all=True, append_images=frames[1:],
                   duration=int(round(1000 / in_fps)), loop=0, disposal=2)
    return data


def measure(state, L, T, n_sweeps, rng, H=0.0):
    """Run n_sweeps sweeps at temperature T and field H, recording the total E and M after each one.

    Like get_movie_and_measure without the movie, and measuring after every
    sweep.  Returns (E, M), arrays of shape (n_sweeps + 1,); entry 0 is the
    starting configuration and entry k is the configuration after k sweeps.
    The state is updated in place.
    """
    E = np.zeros(n_sweeps + 1)
    M = np.zeros(n_sweeps + 1)
    E[0], M[0] = state.E, state.m
    for k in range(1, n_sweeps + 1):
        state = update_sweep(state, L, T, rng, H)
        E[k], M[k] = state.E, state.m      # kept current by `update`
        print("\r  %3d%%" % (100 * k // n_sweeps), end="", flush=True)   # progress
    print()
    return E, M


# ---------------------------------------------------------------------------
# 4.  The parts of the problem to plot.
# ---------------------------------------------------------------------------

def part_a():
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    # TODO


def part_b():
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    # TODO


def part_c():
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    # TODO


def part_f():
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    # TODO


def main():
    """Command line entry: DO NOT MODIFY"""
    parser = argparse.ArgumentParser(
        description="HW6 Ising model in a field -- run one part, or all of them."
    )
    parser.add_argument(
        "--part", choices=["a", "b", "c", "f", "all"], required=True,
        help="Which part of the problem to run.",
    )
    args = parser.parse_args()

    dispatch = {"a": part_a, "b": part_b, "c": part_c, "f": part_f}

    if args.part == "all":
        for func in dispatch.values():
            func()
    else:
        dispatch[args.part]()


if __name__ == "__main__":
    main()
