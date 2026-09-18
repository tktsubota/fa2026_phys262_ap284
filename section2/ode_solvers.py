"""Section 2: Numerical solution of differential equations

We integrate the harmonic oscillator, x'' = -omega^2 x, with omega = 1.  We
know the exact solution, x(t) = cos t, so we can grade every numerical answer
against the truth.  Three parts: Euler's method at two step sizes, the order of
accuracy read off a log-log plot of the error, and the same for RK4.

Run one part at a time (or all of them) from the command line, e.g.:

    python ode_solvers.py --part a
    python ode_solvers.py --part all
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

OMEGA = 1.0                        # unit frequency, so the period is 2 pi
Y0 = np.array([1.0, 0.0])          # start at rest, unit displacement
T = 6 * np.pi                      # integrate for three periods

DTS = [0.1, 0.01]                  # the two step sizes drawn in parts (a) and (c)

# Step sizes for the two convergence studies.  Euler needs Delta t below about
# 0.01 before its error settles onto a power law; RK4 stops at 0.001, since
# below that its error is already down at the level of roundoff.
DTS_EULER = np.logspace(-2, -4, 7)
DTS_RK4 = np.logspace(-1, -3, 7)


def rhs(y, omega=OMEGA):
    """Right-hand side of y' = f(y) for the state y = (x, xdot)."""
    x, xdot = y
    return np.array([xdot, -omega**2 * x])


def exact(t, omega=OMEGA, y0=Y0):
    """Exact trajectory at times t, shape (len(t), 2)."""
    x0, xdot0 = y0
    x = x0 * np.cos(omega * t) + (xdot0 / omega) * np.sin(omega * t)
    xdot = -x0 * omega * np.sin(omega * t) + xdot0 * np.cos(omega * t)
    return np.stack([x, xdot], axis=-1)


def euler_step(y, dt):
    """One step of Euler's method: y_{n+1} = y_n + dt f(y_n)."""
    return y + dt * rhs(y)


def rk4_step(y, dt):
    """One step of the classic fourth-order Runge-Kutta method."""
    k1 = rhs(y)
    k2 = rhs(y + 0.5 * dt * k1)
    k3 = rhs(y + 0.5 * dt * k2)
    k4 = rhs(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(step, dt, t_final=T):
    """Step from 0 to t_final; return the times and the states."""
    n_steps = int(round(t_final / dt))
    ys = np.empty((n_steps + 1, 2))
    ys[0] = Y0
    for n in range(n_steps):
        ys[n + 1] = step(ys[n], dt)
    return dt * np.arange(n_steps + 1), ys


def max_error(step, dt, t_final=T):
    """Largest deviation of the numerical trajectory from the exact one."""
    t, ys = integrate(step, dt, t_final)
    return np.abs(ys - exact(t)).max()


def convergence(step, dts):
    """Error at each step size, and the slope of the log-log fit."""
    errors = np.array([max_error(step, dt) for dt in dts])
    order, log_C = np.polyfit(np.log(dts), np.log(errors), 1)
    return errors, order, np.exp(log_C)


def _ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plot_method(step, name, filename):
    """Trajectory and phase-space portrait at each step size in DTS."""
    t_fine = np.linspace(0, T, 2000)
    ys_fine = exact(t_fine)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for dt in DTS:
        t, ys = integrate(step, dt)
        label = f"$\\Delta t = {dt}$"
        axes[0].plot(t, ys[:, 0], lw=1, label=label)
        axes[1].plot(ys[:, 0], ys[:, 1], lw=1, label=label)
        print(f"{name:5s}, dt={dt:5.3f}: max error = {max_error(step, dt):.3e}")

    axes[0].plot(t_fine, ys_fine[:, 0], "k--", lw=1, label="exact, $\\cos t$")
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel("$x$")
    axes[0].set_title("Trajectory")
    axes[0].legend(fontsize=9)

    axes[1].plot(ys_fine[:, 0], ys_fine[:, 1], "k--", lw=1, label="exact")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$\\dot{x}$")
    axes[1].set_title("Phase space")
    axes[1].legend(fontsize=9, loc="lower right")

    fig.suptitle(f"{name}, $\\omega = 1$, three periods")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def part_a():
    """Euler's method on the oscillator, at two step sizes."""
    out_dir = _ensure_dir(BASE_DIR / "part_a")
    _plot_method(euler_step, "Euler", out_dir / "part_a_euler.png")


def part_b():
    """Order of accuracy of Euler: error vs. step size on a log-log plot."""
    out_dir = _ensure_dir(BASE_DIR / "part_b")

    errors, order, C = convergence(euler_step, DTS_EULER)

    fig, ax = plt.subplots()
    ax.loglog(DTS_EULER, errors, "o", label="Euler")
    ax.loglog(DTS_EULER, C * DTS_EULER**order, "-", label=f"fit, slope $= {order:.2f}$")
    ax.set_xlabel("$\\Delta t$")
    ax.set_ylabel("max error over three periods")
    ax.set_title("Euler's method is first order")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "part_b_euler_convergence.png", dpi=150)
    plt.close(fig)

    for dt, err in zip(DTS_EULER, errors):
        print(f"Euler, dt={dt:8.5f}: max error = {err:.3e}")
    print(f"Euler: fitted order = {order:.2f}")


def part_c():
    """RK4 at the same two step sizes, and its order of accuracy."""
    out_dir = _ensure_dir(BASE_DIR / "part_c")
    _plot_method(rk4_step, "RK4", out_dir / "part_c_rk4.png")

    errors, order, C = convergence(rk4_step, DTS_RK4)

    fig, ax = plt.subplots()
    ax.loglog(DTS_RK4, errors, "o", label="RK4")
    ax.loglog(DTS_RK4, C * DTS_RK4**order, "-", label=f"fit, slope $= {order:.2f}$")
    ax.set_xlabel("$\\Delta t$")
    ax.set_ylabel("max error over three periods")
    ax.set_title("RK4 is fourth order")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "part_c_convergence.png", dpi=150)
    plt.close(fig)

    for dt, err in zip(DTS_RK4, errors):
        print(f"RK4,   dt={dt:8.5f}: max error = {err:.3e}")
    print(f"RK4: fitted order = {order:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Section 2 ODE solvers -- run one part, or all of them."
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
