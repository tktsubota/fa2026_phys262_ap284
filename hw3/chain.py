"""Molecular dynamics engine: a ring of coupled harmonic oscillators.

This is laid out like md.py from HW2, with the Lennard-Jones gas swapped for
N oscillators on a ring,

    H = sum_i [ p_i^2 / 2 + q_i^2 / 2 ] + (lam / 4) sum_i (q_{i+1} - q_i)^4,

with m = omega = 1 and q_{N+1} = q_1.  Each oscillator moves in one dimension,
so `positions` and `velocities` are arrays of shape (N,).
"""

import numpy as np
import h5py


# ---------------------------------------------------------------------------
# 1.  One bond, between neighbors a distance d = q_{i+1} - q_i apart.
# ---------------------------------------------------------------------------

def bond_potential(d, lam):
    """Coupling energy lam d^4 / 4 of one bond."""
    return 0.25 * lam * d**4


def bond_tension(d, lam):
    """du/dd for one bond.

    Positive means the bond is stretched and pulls its two ends together.
    """
    return lam * d**3


# ---------------------------------------------------------------------------
# 2.  N oscillators.  `positions` is an (N,) array, and oscillator N-1 is
#     bonded back to oscillator 0.
# ---------------------------------------------------------------------------

def bond_lengths(positions):
    """d_i = q_{i+1} - q_i for every bond i, shape (N,), wrapping around the ring."""
    return np.roll(positions, -1) - positions


def coupling_forces(positions, lam):
    """Force on each oscillator from its two bonds, shape (N,).

    Bond i pulls oscillator i forward and oscillator i+1 back.
    """
    tension = bond_tension(bond_lengths(positions), lam)
    return tension - np.roll(tension, 1)


def onsite_forces(positions):
    """Force on each oscillator from its own harmonic spring, shape (N,)."""
    return -positions


def forces(positions, lam):
    """Total force on each oscillator, shape (N,): its own spring plus the bonds."""
    return onsite_forces(positions) + coupling_forces(positions, lam)


# ---------------------------------------------------------------------------
# 3.  Energies.
# ---------------------------------------------------------------------------

def onsite_energy(positions):
    """Total energy stored in the harmonic springs, sum of q_i^2 / 2."""
    return 0.5 * np.sum(positions**2)


def coupling_energy(positions, lam):
    """Total energy stored in the bonds.  Each bond is counted once."""
    return np.sum(bond_potential(bond_lengths(positions), lam))


def potential_energy(positions, lam):
    """Harmonic energy plus coupling energy."""
    return onsite_energy(positions) + coupling_energy(positions, lam)


def kinetic_energy(velocities):
    """Sum of m v^2 / 2 over the oscillators, with m = 1."""
    return 0.5 * np.sum(velocities**2)


# ---------------------------------------------------------------------------
# 4.  One time step, and then many.
# ---------------------------------------------------------------------------

def velocity_verlet_step(positions, velocities, accelerations, dt, lam):
    """One velocity-Verlet step: half kick, drift, half kick.

    Identical to md.py, except that the forces come from this Hamiltonian.
    """
    velocities = velocities + 0.5 * dt * accelerations
    positions = positions + dt * velocities
    accelerations = forces(positions, lam)      # the mass is 1, so F = a
    return positions, velocities + 0.5 * dt * accelerations, accelerations


def simulate(positions, velocities, lam, dt, n_steps, sample_every):
    """Step the system forward n_steps times, recording it every `sample_every`.

    Returns a dict of arrays with S = n_steps // sample_every + 1 samples, the
    first one at t = 0: t (S,), positions (S, N), velocities (S, N), K (S,),
    U (S,).  Progress is printed every 5%.
    """
    n_samples = n_steps // sample_every + 1
    history = {
        "t": np.zeros(n_samples),
        "positions": np.zeros((n_samples, len(positions))),
        "velocities": np.zeros((n_samples, len(positions))),
        "K": np.zeros(n_samples),
        "U": np.zeros(n_samples),
    }

    accelerations = forces(positions, lam)      # the mass is 1, so F = a
    next_report = 5
    for step in range(n_steps + 1):
        if step % sample_every == 0:
            sample = step // sample_every
            history["t"][sample] = step * dt
            history["positions"][sample] = positions
            history["velocities"][sample] = velocities
            history["K"][sample] = kinetic_energy(velocities)
            history["U"][sample] = potential_energy(positions, lam)
        if step < n_steps:
            positions, velocities, accelerations = velocity_verlet_step(
                positions, velocities, accelerations, dt, lam)
        if n_steps > 0 and 100 * step >= next_report * n_steps:
            print("  %3d%%  (t = %g)" % (next_report, step * dt), flush=True)
            next_report += 5
    return history


# ---------------------------------------------------------------------------
# 5.  The initial condition, and saving a run to disk.
# ---------------------------------------------------------------------------

def single_oscillator(N, E):
    """All of the energy E in oscillator 0, as kinetic energy; the rest at rest.

    Returns (positions, velocities).  Every coupling term vanishes at t = 0, so
    the total energy is exactly E.
    """
    positions = np.zeros(N)
    velocities = np.zeros(N)
    velocities[0] = np.sqrt(2.0 * E)
    return positions, velocities


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
