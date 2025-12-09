# app.py
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for Streamlit
import matplotlib.pyplot as plt
import streamlit as st

from emulator import (
    create_MESA_population,
    create_YREC_population,
    create_MESA_track,
    create_YREC_track,
)

st.set_page_config(page_title="Stelevator", layout="wide")

st.markdown(
    """
    <style>
    .stelevator-title {
        font-size: 2.5rem;
        font-weight: 400;        /* normal weight for the whole title */
        margin: 0 0 1rem 0;
    }
    .stelevator-title b {
        font-weight: 700;        /* extra bold only for the marked bits */
    }
    </style>
    <div class="stelevator-title">
      <u><b>stelevator</b></u>: <u>stel</u>lar <u>ev</u>olution emul<u>ator</u>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Configuration")

grid = st.sidebar.selectbox("Stellar grid", ["MESA", "YREC"])
mode = st.sidebar.selectbox("Mode", ["Population", "Track"])

# ---------------------------------------------------------------------
# Figure helpers that mimic emulator.py's plot_star / plot_sample
# ---------------------------------------------------------------------

def make_star_figure(df):
    """
    Match emulator.plot_star: 3 panels, Age–P, Teff–logP, Teff–logL with
    benchmark ages highlighted.
    """
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
    return fig


def make_sample_figure(df, M_bounds):
    """
    Match emulator.plot_sample: 3 panels with points colored by mass.
    """
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
    return fig

# ---------------------------------------------------------------------
# UI controls
# ---------------------------------------------------------------------

if mode == "Population":
    st.subheader(f"{grid} population")

    nstars = st.number_input(
        "Number of stars", min_value=100, max_value=200_000, value=10_000, step=1000
    )
    fk = st.number_input("fk", value=7.5)
    rocrit = st.number_input("rocrit", value=1.6)
    include_uncertainties = st.checkbox("Include uncertainties", value=False)

    st.markdown("### Parameter bounds")

    cols = st.columns(4)
    with cols[0]:
        age_mu = st.number_input("Age mean [Gyr]", value=5.0)
    with cols[1]:
        age_sigma = st.number_input("Age sigma [Gyr]", value=5.0)
    with cols[2]:
        age_min = st.number_input("Age min [Gyr]", value=0.0)
    with cols[3]:
        age_max = st.number_input("Age max [Gyr]", value=10.0)

    cols = st.columns(4)
    with cols[0]:
        M_mu = st.number_input("Mass mean [M☉]", value=1.0)
    with cols[1]:
        M_sigma = st.number_input("Mass sigma [M☉]", value=0.1)
    with cols[2]:
        M_min = st.number_input("Mass min [M☉]", value=0.8)
    with cols[3]:
        M_max = st.number_input("Mass max [M☉]", value=1.2)

    cols = st.columns(4)
    with cols[0]:
        feh_mu = st.number_input("[Fe/H] mean", value=0.0)
    with cols[1]:
        feh_sigma = st.number_input("[Fe/H] sigma", value=0.2)
    with cols[2]:
        feh_min = st.number_input("[Fe/H] min", value=-0.3)
    with cols[3]:
        feh_max = st.number_input("[Fe/H] max", value=0.3)

    if grid == "MESA":
        cols = st.columns(4)
        with cols[0]:
            Y_mu = st.number_input("Y_ini mean", value=0.26)
        with cols[1]:
            Y_sigma = st.number_input("Y_ini sigma", value=0.2)
        with cols[2]:
            Y_min = st.number_input("Y_ini min", value=0.22)
        with cols[3]:
            Y_max = st.number_input("Y_ini max", value=0.28)

    cols = st.columns(4)
    with cols[0]:
        alpha_mu = st.number_input("α_MLT mean", value=1.6)
    with cols[1]:
        alpha_sigma = st.number_input("α_MLT sigma", value=0.2)
    with cols[2]:
        alpha_min = st.number_input("α_MLT min", value=1.4)
    with cols[3]:
        alpha_max = st.number_input("α_MLT max", value=2.0)

else:
    st.subheader(f"{grid} track")

    npoints = st.number_input(
        "Number of points", min_value=10, max_value=50_000, value=10_000, step=100
    )
    min_age = st.number_input("Min age [Gyr]", value=0.01)
    max_age = st.number_input("Max age [Gyr]", value=10.0)

    M = st.number_input("Mass [M☉]", value=1.0)
    feh = st.number_input("[Fe/H]", value=0.0)
    if grid == "MESA":
        Y = st.number_input("Y_ini", value=0.26)
    alpha = st.number_input("α_MLT", value=1.6)
    fk = st.number_input("fk", value=7.6)
    rocrit = st.number_input("rocrit", value=1.6)

# ---------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------

run = st.button("Run emulator")

if run:
    with st.spinner("Running emulator..."):
        if mode == "Population":
            age_bounds = [age_mu, age_sigma, age_min, age_max]
            M_bounds = [M_mu, M_sigma, M_min, M_max]
            feh_bounds = [feh_mu, feh_sigma, feh_min, feh_max]
            alpha_bounds = [alpha_mu, alpha_sigma, alpha_min, alpha_max]

            if grid == "MESA":
                Y_bounds = [Y_mu, Y_sigma, Y_min, Y_max]
                df = create_MESA_population(
                    nstars=nstars,
                    fk=fk,
                    rocrit=rocrit,
                    plot=False,
                    save_dataframe=False,
                    include_uncertainties=include_uncertainties,
                    age_bounds=age_bounds,
                    M_bounds=M_bounds,
                    feh_bounds=feh_bounds,
                    Y_bounds=Y_bounds,
                    alpha_bounds=alpha_bounds,
                )
            else:
                df = create_YREC_population(
                    nstars=nstars,
                    fk=fk,
                    rocrit=rocrit,
                    plot=False,
                    save_dataframe=False,
                    include_uncertainties=include_uncertainties,
                    age_bounds=age_bounds,
                    M_bounds=M_bounds,
                    feh_bounds=feh_bounds,
                    alpha_bounds=alpha_bounds,
                )
        else:
            if grid == "MESA":
                df = create_MESA_track(
                    npoints=npoints,
                    min_age=min_age,
                    max_age=max_age,
                    plot=False,
                    save_dataframe=False,
                    M=M,
                    feh=feh,
                    Y=Y,
                    alpha=alpha,
                    fk=fk,
                    rocrit=rocrit,
                )
            else:
                df = create_YREC_track(
                    npoints=npoints,
                    min_age=min_age,
                    max_age=max_age,
                    plot=False,
                    save_dataframe=False,
                    M=M,
                    feh=feh,
                    alpha=alpha,
                    fk=fk,
                    rocrit=rocrit,
                )

    st.success("Done!")

    # --- Plots matching emulator.py style ---
    st.subheader("Plots")

    if mode == "Population":
        fig = make_sample_figure(df, M_bounds)
    else:
        fig = make_star_figure(df)

    st.pyplot(fig)
    plt.close(fig)

    # --- Table + download ---
    st.subheader("First few rows of output")
    st.dataframe(df.head())

    st.subheader("Download CSV")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    filename = f"{grid}_{mode.lower()}_output.csv"
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )
else:
    st.info("Set parameters in the sidebar and click **Run emulator**.")
