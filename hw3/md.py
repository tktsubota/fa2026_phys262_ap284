"""Molecular dynamics engine: a two-dimensional Lennard-Jones gas in a box.

Based on Daniel V. Schroeder, "Interactive molecular dynamics", Am. J. Phys. 83,
210 (2015).
"""

import numpy as np
import h5py

R_CUT = 3.0                                    # pair interactions are cut off here
U_CUT = 4.0 * (R_CUT**-12.0 - R_CUT**-6.0)     # ...and shifted by this, so u(R_CUT) = 0
K_WALL = 50.0                                  # stiffness of the soft walls


# ---------------------------------------------------------------------------
# 1.  One pair of particles a distance r apart.
# ---------------------------------------------------------------------------

def pair_potential(r):
    """Lennard-Jones potential energy u(r) of a pair of particles a distance r apart.

    The potential is cut off at R_CUT and shifted down by U_CUT so that it goes
    continuously to zero there; without the shift, a pair crossing the cutoff
    would dump a small jump of energy into the system.
    """
    return np.where(r < R_CUT, 4.0 * (r**-12.0 - r**-6.0) - U_CUT, 0.0)


def pair_force(r):
    """Force -du/dr between a pair of particles a distance r apart.

    Positive means repulsive, pushing the pair apart.  It vanishes at the
    minimum of u(r), at r = 2^(1/6), and, like the potential, beyond R_CUT.
    """
    return np.where(r < R_CUT, 24.0 * (2.0 * r**-13.0 - r**-7.0), 0.0)


# ---------------------------------------------------------------------------
# 2.  N particles.  `positions` is an (N, 2) array, one row per particle, and
#     the box is the square [0, L] x [0, L].
# ---------------------------------------------------------------------------

def separations(positions):
    """Displacement vectors and distances for every pair: (N, N, 2) and (N, N).

    The first return value points from particle j to particle i.  The diagonal
    of the second is set to infinity rather than zero, so that a particle
    contributes nothing to its own force or energy -- both pair_force and
    pair_potential vanish beyond the cutoff.
    """
    dr = positions[:, None, :] - positions[None, :, :]
    r = np.sqrt(np.sum(dr**2, axis=-1))
    np.fill_diagonal(r, np.inf)
    return dr, r


def pair_forces(positions):
    """Lennard-Jones force on each particle from all the others, shape (N, 2)."""
    dr, r = separations(positions)
    direction = dr / r[:, :, None]             # unit vector pointing from j to i
    return np.sum(pair_force(r)[:, :, None] * direction, axis=1)


def wall_depth(positions, L):
    """How far each coordinate has strayed outside the box, shape (N, 2).

    Zero for a particle inside the box, negative if it has gone past the wall at
    0 and positive if past the wall at L.  Each wall is a one-sided quadratic
    potential, so this is the displacement of that spring.
    """
    return (np.where(positions < 0.0, positions, 0.0)
            + np.where(positions > L, positions - L, 0.0))


def wall_forces(positions, L):
    """Force on each particle from the soft walls, shape (N, 2)."""
    return -K_WALL * wall_depth(positions, L)


def forces(positions, L):
    """Total force on each particle, shape (N, 2): the pair forces plus the walls."""
    return pair_forces(positions) + wall_forces(positions, L)


# ---------------------------------------------------------------------------
# 3.  Energies.
# ---------------------------------------------------------------------------

def pair_energy(positions):
    """Total Lennard-Jones energy.

    The sum runs over ordered pairs and so counts each pair twice; the factor of
    one half undoes that.
    """
    return 0.5 * np.sum(pair_potential(separations(positions)[1]))


def wall_energy(positions, L):
    """Total energy stored in the soft walls."""
    return 0.5 * K_WALL * np.sum(wall_depth(positions, L)**2)


def potential_energy(positions, L):
    """Lennard-Jones energy plus wall energy."""
    return pair_energy(positions) + wall_energy(positions, L)


def kinetic_energy(velocities):
    """Sum of m v^2 / 2 over the particles, with m = 1."""
    return 0.5 * np.sum(velocities**2)


def wall_pressure(positions, L):
    """Instantaneous pressure: total outward force on the walls per unit length.

    Each compressed wall spring pushes outward on its wall with force
    K_WALL * |depth|.  Summing over all four walls and dividing by the perimeter
    4L gives the pressure, as in Schroeder's paper.
    """
    return K_WALL * np.sum(np.abs(wall_depth(positions, L))) / (4.0 * L)



# ---------------------------------------------------------------------------
# 4.  One time step, and then many.
# ---------------------------------------------------------------------------

def velocity_verlet_step(positions, velocities, accelerations, dt, L):
    """One velocity-Verlet step: half kick, drift, half kick.

    This is the composition derived in part (b), written out for this system.
    `accelerations` is the acceleration at the incoming positions and supplies
    the opening half kick; the acceleration at the new positions is returned
    along with them, so that the next step does not have to work it out again
    and each step costs only one force evaluation.
    """
    velocities = velocities + 0.5 * dt * accelerations
    positions = positions + dt * velocities
    accelerations = forces(positions, L)        # the mass is 1, so F = a
    return positions, velocities + 0.5 * dt * accelerations, accelerations


def simulate(positions, velocities, L, dt, n_steps, sample_every):
    """Step the system forward n_steps times, recording it every `sample_every`.

    Returns a dict of arrays with S = n_steps // sample_every + 1 samples, the
    first one at t = 0: t (S,), positions (S, N, 2), velocities (S, N, 2),
    K (S,), U (S,), P (S,).

    Wall collisions are brief, so a snapshot of the wall force every
    `sample_every` steps would miss most of them.  P is therefore the wall
    pressure averaged over every step since the previous sample (the first entry
    is just the instantaneous value at t = 0).  Progress is printed every 5%.
    """
    n_samples = n_steps // sample_every + 1
    history = {
        "t": np.zeros(n_samples),
        "positions": np.zeros((n_samples, len(positions), 2)),
        "velocities": np.zeros((n_samples, len(positions), 2)),
        "K": np.zeros(n_samples),
        "U": np.zeros(n_samples),
        "P": np.zeros(n_samples),
    }

    accelerations = forces(positions, L)        # the mass is 1, so F = a
    pressure_sum = wall_pressure(positions, L) * sample_every
    next_report = 5
    for step in range(n_steps + 1):
        if step % sample_every == 0:
            sample = step // sample_every
            history["t"][sample] = step * dt
            history["positions"][sample] = positions
            history["velocities"][sample] = velocities
            history["K"][sample] = kinetic_energy(velocities)
            history["U"][sample] = potential_energy(positions, L)
            history["P"][sample] = pressure_sum / sample_every
            pressure_sum = 0.0
        if step < n_steps:
            positions, velocities, accelerations = velocity_verlet_step(
                positions, velocities, accelerations, dt, L)
            pressure_sum += wall_pressure(positions, L)
        if n_steps > 0 and 100 * step >= next_report * n_steps:
            print("  %3d%%  (t = %g)" % (next_report, step * dt), flush=True)
            next_report += 5
    return history


# ---------------------------------------------------------------------------
# 5.  The initial condition, and saving a run to disk.
# ---------------------------------------------------------------------------

def compressed_lattice(N, L, spacing):
    """N particles at rest on a square lattice in the middle of the box.

    Returns (positions, velocities).

    With `spacing` below the minimum of u(r), at r = 2^(1/6), every neighboring
    pair sits on the repulsive wall of the potential, so the clump starts with a
    large potential energy and will fly apart as soon as it is released.
    Nothing random happens here, so these three numbers fix the state, and hence
    the whole trajectory, completely.
    """
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
