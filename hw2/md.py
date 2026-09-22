"""Molecular dynamics engine: a two-dimensional Lennard-Jones gas in a box.

Based on Daniel V. Schroeder, "Interactive molecular dynamics", Am. J. Phys. 83,
210 (2015).

Fill in the TODOs.
"""

import numpy as np
import h5py

R_CUT = 3.0
U_CUT = 4.0 * (R_CUT**-12.0 - R_CUT**-6.0)
K_WALL = 50.0


def pair_potential(r):
    # TODO
    pass


def pair_force(r):
    # TODO
    pass


def separations(positions):
    # TODO
    pass


def pair_forces(positions):
    # TODO
    pass


def wall_depth(positions, L):
    # TODO
    pass


def wall_forces(positions, L):
    # TODO
    pass


def forces(positions, L):
    # TODO
    pass


def pair_energy(positions):
    # TODO
    pass


def wall_energy(positions, L):
    # TODO
    pass


def potential_energy(positions, L):
    # TODO
    pass


def kinetic_energy(velocities):
    # TODO
    pass


def velocity_verlet_step(positions, velocities, accelerations, dt, L):
    # TODO
    pass


def simulate(positions, velocities, L, dt, n_steps, sample_every):
    """Step the system forward n_steps times, recording it every `sample_every`.

    Returns a dict of arrays with S = n_steps // sample_every + 1 samples, the
    first one at t = 0: t (S,), positions (S, N, 2), velocities (S, N, 2),
    K (S,), U (S,).
    """
    n_samples = n_steps // sample_every + 1
    history = {
        "t": np.zeros(n_samples),
        "positions": np.zeros((n_samples, len(positions), 2)),
        "velocities": np.zeros((n_samples, len(positions), 2)),
        "K": np.zeros(n_samples),
        "U": np.zeros(n_samples),
    }

    # TODO: step the system forward, filling in `history` every sample_every
    # steps (including at step 0), and return it.
    raise NotImplementedError


def compressed_lattice(N, L, spacing):
    """N particles at rest on a square lattice in the middle of the box."""
    side = int(np.ceil(np.sqrt(N)))            # smallest square grid holding N sites
    x, y = np.meshgrid(np.arange(side) * spacing, np.arange(side) * spacing)
    lattice = np.column_stack([x.ravel(), y.ravel()])[:N]
    centered = lattice - lattice.mean(axis=0) + 0.5 * L
    return centered, np.zeros((N, 2))


def save_run(path, history):
    """Write the arrays of a run to `path`, one dataset per entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        for name, array in history.items():
            f.create_dataset(name, data=array)


def load_run(path):
    """Read back a run written by save_run, as the same dict of arrays."""
    with h5py.File(str(path), "r") as f:
        return {name: f[name][...] for name in f}
