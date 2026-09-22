"""HW2 Problem 2: Molecular dynamics of a two-dimensional Lennard-Jones gas.

Run one part at a time (or all of them) from the command line, e.g.:

    python molecular_dynamics.py --part c
    python molecular_dynamics.py --part all

The plotting helpers and part (c) are written for you; the work in part (c) is
implementing md.py.  Once that is done, `--part c` writes the trajectory to
data/run.h5, and parts (d) and (e) read it back rather than simulating again, so
run part (c) first.  Fill in the TODOs in parts (d) to (g).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import md

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RUN_FILE = DATA_DIR / "run.h5"

# Parameters and initial condition
N = 100                    # particles
L = 20.0                   # side of the box
SPACING = 0.95             # initial lattice spacing
DT = 0.005                 # time step
T_MAX = 50.0               # how long to run for
SAMPLE_EVERY = 20          # record frame after this many steps


def _ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def marker_area(ax, diameter=1.0):
    """Marker area for `ax.scatter` that draws a circle `diameter` wide in data units.

    Matplotlib sizes scatter markers in points squared, which has nothing to do
    with the axis scale, so a fixed size draws the particles far smaller than
    they really are and collisions look like point particles missing each other.
    The Lennard-Jones diameter is sigma = 1, so that is the default.
    """
    ax.figure.canvas.draw()                # the axes need a size before we measure it
    width_in_points = ax.get_window_extent().width * 72.0 / ax.figure.dpi
    x_lo, x_hi = ax.get_xlim()
    return (diameter * width_in_points / (x_hi - x_lo))**2


def save_movie(run, L, path, t_end=40.0, fps=25, flip_at=None):
    """Animate the particles up to time t_end, colored by speed, and write `path`.

    Writes an animated GIF, which needs nothing beyond matplotlib.  `flip_at` is
    only for part (g): pass the time at which the momenta were reversed and the
    title says so from then on, which makes it obvious which half of the movie
    you are watching.
    """
    keep = run["t"] <= t_end
    t, r, v = run["t"][keep], run["positions"][keep], run["velocities"][keep]
    speeds = np.linalg.norm(v, axis=2)

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.set_xlim(-1.0, L + 1.0)
    ax.set_ylim(-1.0, L + 1.0)
    ax.set_aspect("equal")
    for edge in (0.0, L):
        ax.axvline(edge, color="gray", lw=1)
        ax.axhline(edge, color="gray", lw=1)
    scatter = ax.scatter(r[0, :, 0], r[0, :, 1], s=marker_area(ax), c=speeds[0],
                         cmap="inferno", vmin=0.0, vmax=speeds.max())
    title = ax.set_title("$t=0$")

    def frame(k):
        scatter.set_offsets(r[k])
        scatter.set_array(speeds[k])
        label = "$t=%.1f$" % t[k]
        if flip_at is not None and t[k] > flip_at:
            label += "  (running backwards)"
        title.set_text(label)
        return scatter, title

    anim = animation.FuncAnimation(fig, frame, frames=len(t),
                                   interval=1000 // fps, blit=False)
    anim.save(str(path), writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print("%s: %d frames covering t = 0 to %g" % (path.name, len(t), t[-1]))


def snapshot_grid(run, L, times, path):
    """A strip of configuration snapshots at the requested times."""
    fig, axes = plt.subplots(1, len(times), figsize=(3.2 * len(times), 3.4))
    for ax, t_snap in zip(axes, times):
        r = run["positions"][np.argmin(np.abs(run["t"] - t_snap))]
        ax.set_xlim(-1.0, L + 1.0)
        ax.set_ylim(-1.0, L + 1.0)
        ax.set_aspect("equal")
        ax.scatter(r[:, 0], r[:, 1], s=marker_area(ax))
        ax.set_title("$t=%g$" % t_snap)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def part_c():
    out_dir = _ensure_dir(BASE_DIR / "part_c")

    positions, velocities = md.compressed_lattice(N, L, SPACING)
    n_steps = int(round(T_MAX / DT))

    run = md.simulate(positions, velocities, L, DT, n_steps, SAMPLE_EVERY)
    md.save_run(RUN_FILE, run)

    snapshot_grid(run, L, [0.0, 1.0, 5.0, 10.0, 15.0, 30.0],
                  out_dir / "part_c_snapshots.png")
    save_movie(run, L, out_dir / "part_c_movie.gif")


def part_d():
    out_dir = _ensure_dir(BASE_DIR / "part_d")

    run = md.load_run(RUN_FILE)

    # TODO


def part_e():
    out_dir = _ensure_dir(BASE_DIR / "part_e")

    run = md.load_run(RUN_FILE)

    # TODO


def part_f():
    out_dir = _ensure_dir(BASE_DIR / "part_f")

    # TODO


def part_g():
    out_dir = _ensure_dir(BASE_DIR / "part_g")

    # TODO


def main():
    """Command line entry: DO NOT MODIFY"""
    parser = argparse.ArgumentParser(
        description="HW2 Molecular Dynamics -- run one part, or all of them."
    )
    parser.add_argument(
        "--part", choices=["c", "d", "e", "f", "g", "all"], required=True,
        help="Which part of the problem to run.",
    )
    args = parser.parse_args()

    dispatch = {"c": part_c, "d": part_d, "e": part_e,
                "f": part_f, "g": part_g}

    if args.part == "all":
        for func in dispatch.values():
            func()
    else:
        dispatch[args.part]()


if __name__ == "__main__":
    main()
