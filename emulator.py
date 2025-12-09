import os
from pathlib import Path
from functools import lru_cache

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quiet TF C++ logs

import tensorflow as tf
from tensorflow.keras.models import load_model

tf.get_logger().setLevel("ERROR")  # quiet TF Python logs

ROOT = Path(__file__).resolve().parent

font = {"size": 16}
matplotlib.rc("font", **font)

@lru_cache(maxsize=2)
def _load_params(grid: str, base_dir: str | os.PathLike | None = None):
    """
    Load pre-exported emulator parameters for the given grid.

    Parameters
    ----------
    grid : {"MESA", "YREC"}
    base_dir : optional base directory (defaults to this file's directory)
    """
    base_dir = Path(base_dir) if base_dir is not None else ROOT

    if grid == "MESA":
        npz_path = base_dir / "data" / "models" / "rotated_mesa_run3_params.npz"
    elif grid == "YREC":
        npz_path = base_dir / "data" / "models" / "model_4_params.npz"
    else:
        raise ValueError("grid must be 'MESA' or 'YREC'")

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Parameter file not found: {npz_path}\n"
            "Did you run export_emulator_params.py?"
        )

    data = np.load(npz_path, allow_pickle=True)

    # w and b are stored as object arrays; convert back to Python lists of ndarrays
    w = [np.array(arr) for arr in data["w"]]
    b = [np.array(arr) for arr in data["b"]]

    params = {
        "input_offset": np.array(data["input_offset"]),
        "input_scale": np.array(data["input_scale"]),
        "output_offset": np.array(data["output_offset"]),
        "output_scale": np.array(data["output_scale"]),
        "w": w,
        "b": b,
    }
    return params


def _elu(x: np.ndarray) -> np.ndarray:
    """
    Numerically safe ELU with alpha=1.

    f(x) = x        if x >= 0
           exp(x)-1 if x < 0
    """
    x = np.asarray(x, dtype=np.float32)
    out = x.copy()
    neg_mask = out < 0
    out[neg_mask] = np.exp(out[neg_mask]) - 1.0
    return out


def emulate(
    inputs: np.ndarray,
    grid: str,
    base_dir: str | os.PathLike | None = None,
) -> np.ndarray:
    """
    Forward pass through the emulator network using pure NumPy.

    Parameters
    ----------
    inputs : array-like, shape (n_samples, n_features)
        e.g. for MESA: [log10(age), M, feh, Y, alpha, fk, rocrit]
    grid   : {"MESA", "YREC"}
    base_dir : optional base directory for model paths.

    Returns
    -------
    outputs : np.ndarray, shape (n_samples, n_outputs)
    """
    params = _load_params(grid, base_dir=base_dir)

    x = np.asarray(inputs, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]

    # Input normalization
    x = (x - params["input_offset"]) / params["input_scale"]

    w_list = params["w"]
    b_list = params["b"]

    # Hidden layers with ELU
    for wi, bi in zip(w_list[:-1], b_list[:-1]):
        z = x @ wi + bi
        x = _elu(z)

    # Final linear layer
    outputs = x @ w_list[-1] + b_list[-1]

    # Output scaling
    outputs = params["output_offset"] + outputs * params["output_scale"]
    return outputs


def bound_normal(mean=0, sd=1, low=0, upp=10, size=1):
    return truncnorm.rvs(
        (low - mean) / sd,
        (upp - mean) / sd,
        loc=mean,
        size=size,
        scale=sd,
    )


def create_MESA_population(
    nstars=10000,
    fk=7.5,
    rocrit=1.6,
    plot=True,
    save_dataframe=False,
    df_path="MESA_sim_track.csv",
    include_uncertainties=False,
    age_bounds=[5.0, 5.0, 0.0, 10.0],
    M_bounds=[1.0, 0.1, 0.8, 1.2],
    feh_bounds=[0.0, 0.2, -0.3, 0.3],
    Y_bounds=[0.26, 0.2, 0.22, 0.28],
    alpha_bounds=[1.6, 0.2, 1.4, 2.0],
    base_dir: str | os.PathLike | None = None,
):

    age_sample = bound_normal(*age_bounds, nstars)
    M_sample = bound_normal(*M_bounds, nstars)
    feh_sample = bound_normal(*feh_bounds, nstars)
    Y_sample = bound_normal(*Y_bounds, nstars)
    alpha_sample = bound_normal(*alpha_bounds, nstars)
    fk_sample = np.ones(nstars) * fk
    rocrit_sample = np.ones(nstars) * rocrit

    sim_inputs = np.column_stack(
        [
            np.log10(age_sample),
            M_sample,
            feh_sample,
            Y_sample,
            alpha_sample,
            fk_sample,
            rocrit_sample,
        ]
    )

    sim_y = emulate(sim_inputs, "MESA", base_dir=base_dir)

    Teffs = 10 ** sim_y.T[0]
    Rs = 10 ** sim_y.T[1]
    surface_fehs = sim_y.T[2]
    Prots = 10 ** sim_y.T[3]

    Ls = (Rs**2) * ((Teffs / 5777.0) ** 4)

    if include_uncertainties:
        err_Age = np.ones(nstars) * 0.2
        err_P = np.ones(nstars) * 5.0
        err_Teff = np.ones(nstars) * 100.0
        err_M = np.ones(nstars) * 0.02
        err_L = np.ones(nstars) * 0.02
        err_alpha = np.ones(nstars) * 0.07
        efeh = np.ones(nstars) * 0.1
        err_R = np.ones(nstars) * 0.01
        err_Yini = np.ones(nstars) * 0.02

        sim_df = pd.DataFrame(
            np.array(
                [
                    age_sample,
                    err_Age,
                    M_sample,
                    err_M,
                    Ls,
                    err_L,
                    feh_sample,
                    efeh,
                    Y_sample,
                    err_Yini,
                    alpha_sample,
                    err_alpha,
                    fk_sample,
                    rocrit_sample,
                    Teffs,
                    err_Teff,
                    Rs,
                    err_R,
                    surface_fehs,
                    Prots,
                    err_P,
                ]
            ).T,
            columns=[
                "Age",
                "err_Age",
                "M",
                "err_M",
                "L",
                "err_L",
                "feh",
                "efeh",
                "Yini",
                "err_Yini",
                "alpha",
                "err_alpha",
                "fk",
                "rocrit",
                "Teff",
                "err_Teff",
                "R",
                "err_R",
                "feh_surf",
                "P",
                "err_P",
            ],
        )

    else:
        sim_df = pd.DataFrame(
            np.array(
                [
                    age_sample,
                    M_sample,
                    Ls,
                    feh_sample,
                    Y_sample,
                    alpha_sample,
                    fk_sample,
                    rocrit_sample,
                    Teffs,
                    Rs,
                    surface_fehs,
                    Prots,
                ]
            ).T,
            columns=[
                "Age",
                "M",
                "L",
                "feh",
                "Yini",
                "alpha",
                "fk",
                "rocrit",
                "Teff",
                "R",
                "feh_surf",
                "P",
            ],
        )

    if save_dataframe:
        sim_df.to_csv(df_path, index=False)

    if plot:
        plot_sample(sim_df, M_bounds=M_bounds)

    return sim_df


def create_YREC_population(
    nstars=10000,
    fk=7.5,
    rocrit=1.6,
    plot=True,
    save_dataframe=False,
    df_path="YREC_sim_track.csv",
    include_uncertainties=False,
    age_bounds=[5.0, 5.0, 0.0, 10.0],
    M_bounds=[1.0, 0.1, 0.8, 1.2],
    feh_bounds=[0.0, 0.2, -0.3, 0.3],
    alpha_bounds=[1.6, 0.2, 1.4, 2.0],
    base_dir: str | os.PathLike | None = None,
):

    age_sample = bound_normal(*age_bounds, nstars)
    M_sample = bound_normal(*M_bounds, nstars)
    feh_sample = bound_normal(*feh_bounds, nstars)
    alpha_sample = bound_normal(*alpha_bounds, nstars)
    fk_sample = np.ones(nstars) * fk
    rocrit_sample = np.ones(nstars) * rocrit

    sim_inputs = np.column_stack(
        [
            np.log10(age_sample),
            M_sample,
            feh_sample,
            alpha_sample,
            fk_sample,
            rocrit_sample,
        ]
    )

    sim_y = emulate(sim_inputs, "YREC", base_dir=base_dir)

    Teffs = 10 ** sim_y.T[0]
    Rs = 10 ** sim_y.T[1]
    surface_fehs = 10 ** sim_y.T[2]
    Prots = 10 ** sim_y.T[3]

    Ls = (Rs**2) * ((Teffs / 5777.0) ** 4)

    if include_uncertainties:
        err_Age = np.ones(nstars) * 0.2
        err_P = np.ones(nstars) * 5.0
        err_Teff = np.ones(nstars) * 100.0
        err_M = np.ones(nstars) * 0.02
        err_L = np.ones(nstars) * 0.02
        err_alpha = np.ones(nstars) * 0.07
        efeh = np.ones(nstars) * 0.1
        err_R = np.ones(nstars) * 0.01

        sim_df = pd.DataFrame(
            np.array(
                [
                    age_sample,
                    err_Age,
                    M_sample,
                    err_M,
                    Ls,
                    err_L,
                    feh_sample,
                    efeh,
                    alpha_sample,
                    err_alpha,
                    fk_sample,
                    rocrit_sample,
                    Teffs,
                    err_Teff,
                    Rs,
                    err_R,
                    surface_fehs,
                    Prots,
                    err_P,
                ]
            ).T,
            columns=[
                "Age",
                "err_Age",
                "M",
                "err_M",
                "L",
                "err_L",
                "feh",
                "efeh",
                "alpha",
                "err_alpha",
                "fk",
                "rocrit",
                "Teff",
                "err_Teff",
                "R",
                "err_R",
                "feh_surf",
                "P",
                "err_P",
            ],
        )

    else:
        sim_df = pd.DataFrame(
            np.array(
                [
                    age_sample,
                    M_sample,
                    Ls,
                    feh_sample,
                    alpha_sample,
                    fk_sample,
                    rocrit_sample,
                    Teffs,
                    Rs,
                    surface_fehs,
                    Prots,
                ]
            ).T,
            columns=[
                "Age",
                "M",
                "L",
                "feh",
                "alpha",
                "fk",
                "rocrit",
                "Teff",
                "R",
                "feh_surf",
                "P",
            ],
        )

    if save_dataframe:
        sim_df.to_csv(df_path, index=False)

    if plot:
        plot_sample(sim_df, M_bounds=M_bounds)

    return sim_df


def create_MESA_track(
    npoints=10000,
    min_age=0.01,
    max_age=10.0,
    plot=True,
    save_dataframe=False,
    df_path="MESA_sim_track.csv",
    M=1.0,
    feh=0.0,
    Y=0.26,
    alpha=1.6,
    fk=7.6,
    rocrit=1.6,
    base_dir: str | os.PathLike | None = None,
):

    age_sample = np.linspace(min_age, max_age, npoints)
    M_sample = np.ones(npoints) * M
    feh_sample = np.ones(npoints) * feh
    Y_sample = np.ones(npoints) * Y
    alpha_sample = np.ones(npoints) * alpha
    fk_sample = np.ones(npoints) * fk
    rocrit_sample = np.ones(npoints) * rocrit

    sim_inputs = np.column_stack(
        [
            np.log10(age_sample),
            M_sample,
            feh_sample,
            Y_sample,
            alpha_sample,
            fk_sample,
            rocrit_sample,
        ]
    )

    sim_y = emulate(sim_inputs, "MESA", base_dir=base_dir)

    Teffs = 10 ** sim_y.T[0]
    Rs = 10 ** sim_y.T[1]
    surface_fehs = 10 ** sim_y.T[2]
    Prots = 10 ** sim_y.T[3]

    Ls = (Rs**2) * ((Teffs / 5777.0) ** 4)

    sim_df = pd.DataFrame(
        np.array(
            [
                age_sample,
                M_sample,
                Ls,
                feh_sample,
                Y_sample,
                alpha_sample,
                fk_sample,
                rocrit_sample,
                Teffs,
                Rs,
                surface_fehs,
                Prots,
            ]
        ).T,
        columns=[
            "Age",
            "M",
            "L",
            "feh",
            "Yini",
            "alpha",
            "fk",
            "rocrit",
            "Teff",
            "R",
            "feh_surf",
            "P",
        ],
    )

    if save_dataframe:
        sim_df.to_csv(df_path, index=False)

    if plot:
        plot_star(sim_df)

    return sim_df


def create_YREC_track(
    npoints=10000,
    min_age=0.01,
    max_age=10.0,
    plot=True,
    save_dataframe=False,
    df_path="YREC_sim_track.csv",
    M=1.0,
    feh=0.0,
    alpha=1.6,
    fk=7.6,
    rocrit=1.6,
    base_dir: str | os.PathLike | None = None,
):

    age_sample = np.linspace(min_age, max_age, npoints)
    M_sample = np.ones(npoints) * M
    feh_sample = np.ones(npoints) * feh
    alpha_sample = np.ones(npoints) * alpha
    fk_sample = np.ones(npoints) * fk
    rocrit_sample = np.ones(npoints) * rocrit

    sim_inputs = np.column_stack(
        [
            np.log10(age_sample),
            M_sample,
            feh_sample,
            alpha_sample,
            fk_sample,
            rocrit_sample,
        ]
    )

    sim_y = emulate(sim_inputs, "YREC", base_dir=base_dir)

    Teffs = 10 ** sim_y.T[0]
    Rs = 10 ** sim_y.T[1]
    surface_fehs = 10 ** sim_y.T[2]
    Prots = 10 ** sim_y.T[3]

    Ls = (Rs**2) * ((Teffs / 5777.0) ** 4)

    sim_df = pd.DataFrame(
        np.array(
            [
                age_sample,
                M_sample,
                Ls,
                feh_sample,
                alpha_sample,
                fk_sample,
                rocrit_sample,
                Teffs,
                Rs,
                surface_fehs,
                Prots,
            ]
        ).T,
        columns=[
            "Age",
            "M",
            "L",
            "feh",
            "alpha",
            "fk",
            "rocrit",
            "Teff",
            "R",
            "feh_surf",
            "P",
        ],
    )

    if save_dataframe:
        sim_df.to_csv(df_path, index=False)

    if plot:
        plot_star(sim_df)

    return sim_df


def plot_star(df):

    benchmarks = [1e-3, 1e-2, 1e-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    dims = (4, 10)
    plt.figure(dpi=120)
    ax = plt.subplot2grid(dims, (0, 0), colspan=4, rowspan=2)

    ax.plot(df["Age"], df["P"], c="cornflowerblue")
    ax.set_xlabel("Age [Gyr]")
    ax.set_ylabel("P [days]")

    ax = plt.subplot2grid(dims, (2, 0), colspan=4, rowspan=2)

    ax.plot(df["Teff"], np.log10(df["P"]), c="k", lw=1)
    for b in benchmarks:
        if (b >= df["Age"].iloc[0]) and (b <= df["Age"].iloc[-1]):
            idx = np.argmin(np.abs(b - df["Age"]))
            ax.scatter(
                df["Teff"][idx],
                np.log10(df["P"][idx]),
                c=df["Age"][idx],
                zorder=1000,
                vmin=df["Age"].iloc[0],
                vmax=df["Age"].iloc[-1],
                s=75,
            )
    ax.set_xlim(np.max(df["Teff"]) + 100, np.min(df["Teff"]) - 100)
    ax.set_xlabel(r"T$_{\rm eff}$ [K]")
    ax.set_ylabel("log(P) [days]")

    ax = plt.subplot2grid(dims, (0, 4), colspan=5, rowspan=4)
    ax.plot(df["Teff"], np.log10(df["L"]), c="k", lw=1)
    for b in benchmarks:
        if (b >= df["Age"].iloc[0]) and (b <= df["Age"].iloc[-1]):
            idx = np.argmin(np.abs(b - df["Age"]))
            ax.scatter(
                df["Teff"][idx],
                np.log10(df["L"][idx]),
                c=df["Age"][idx],
                zorder=1000,
                vmin=df["Age"].iloc[0],
                vmax=df["Age"].iloc[-1],
                s=75,
                label=f"{b} Gyr",
            )
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", fontsize=10)
    ax.set_xlim(np.max(df["Teff"]) + 100, np.min(df["Teff"]) - 100)
    ax.set_ylim(
        np.min(np.log10(df["L"])) - 0.025,
        np.max(np.log10(df["L"])) + 0.025,
    )
    ax.set_xlabel(r"T$_{\rm eff}$ [K]")
    ax.set_ylabel(r"log(L) [L$_\odot$]")

    fig = plt.gcf()
    fig.patch.set_facecolor("white")
    fig.set_size_inches([d + 5 for d in dims[::-1]])
    fig.tight_layout()
    plt.show()


def plot_sample(df, M_bounds=[1.0, 0.5, 0.8, 1.2]):

    age_sample = np.array(df["Age"])
    M_sample = np.array(df["M"])
    Prots = np.array(df["P"])
    Teffs = np.array(df["Teff"])
    Ls = np.array(df["L"])

    dims = (4, 10)
    plt.figure(dpi=120)
    ax = plt.subplot2grid(dims, (0, 0), colspan=4, rowspan=2)

    ax.scatter(
        age_sample,
        Prots,
        edgecolor="None",
        c=M_sample,
        alpha=0.5,
        s=10,
        vmin=M_bounds[2],
        vmax=M_bounds[3],
    )
    ax.set_ylim(0, 50)
    ax.set_xlabel("Age [Gyr]")
    ax.set_ylabel("P [days]")

    ax = plt.subplot2grid(dims, (2, 0), colspan=4, rowspan=2)

    ax.scatter(
        Teffs,
        np.log10(Prots),
        edgecolor="None",
        c=M_sample,
        alpha=0.5,
        s=10,
        vmin=M_bounds[2],
        vmax=M_bounds[3],
    )
    ax.set_xlim(7000, 4000)
    ax.set_ylim(0.2, 2)
    ax.set_xlabel(r"T$_{\rm eff}$ [K]")
    ax.set_ylabel("log(P) [days]")

    ax = plt.subplot2grid(dims, (0, 4), colspan=5, rowspan=4)

    ax.scatter(
        Teffs,
        np.log10(Ls),
        edgecolor="None",
        c=M_sample,
        alpha=0.5,
        s=10,
        vmin=M_bounds[2],
        vmax=M_bounds[3],
    )
    ax.set_xlim(7000, 4000)
    ax.set_ylim(-1, 2)
    ax.set_xlabel(r"T$_{\rm eff}$ [K]")
    ax.set_ylabel(r"log(L) [L$_\odot$]")
    cax = ax.scatter(1, 1, c=1, vmin=M_bounds[2], vmax=M_bounds[3])
    plt.colorbar(cax, ax=ax, label=r"M [M$_\odot$]")

    fig = plt.gcf()
    fig.patch.set_facecolor("white")
    fig.set_size_inches([d + 5 for d in dims[::-1]])
    fig.tight_layout()
    plt.show()