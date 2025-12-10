from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import emulator as em

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="stelevator", layout="wide")

# ---------------------------------------------------------------------
# Custom title styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stelevator-title {
        font-size: 2.5rem;
        font-weight: 400;
        margin: 0 0 1rem 0;
    }
    .stelevator-title b, {
        font-weight: 700;
    }
    </style>
    <div class="stelevator-title">
      <u><b>stelevator</b></u>: a <u>stel</u>lar <u>ev</u>olution emul<u>ator</u>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("***↑ Going up?***")

st.markdown(
    "Grid-based stellar evolution emulator for MESA / YREC models. "
    "Use the sidebar to choose a model grid and mode."
)

st.markdown(
    "Developed by Nicholas Saunders ([GitHub](https://github.com/nksaunders), [Personal Website](https://nksaunders.space)).  \n"
    "If you use this software in a publication, please cite "
    "[Saunders et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024ApJ...962..138S/abstract)."
)

st.markdown("---")

st.subheader("Read me before using")

st.markdown("#### About stelevator")

st.markdown(
    "**Stelevator** is an artificial neural network trained to emulate a limited set of outputs "
    "from detailed stellar evolution models. It is designed to rapidly generate large populations "
    "of stars, or individual evolutionary tracks, for use in stellar and exoplanetary studies. "
    "**Stelevator** currently supports two model grids: MESA and YREC."
)

st.markdown(
    "This tool was developed to examine the behavior of stellar rotation over time, "
    "and includes a simple braking law to model angular momentum loss via magnetized stellar winds. "
    "Because of this, **stelevator** takes two additional parameters not typically seen in stellar models:  \n"
    " - $f_K$, the braking law strength  \n"
    r" - $\textrm{Ro}_\textsf{crit}$, the critical Rossby number "    
)

st.markdown(
    "Please see the [paper](https://ui.adsabs.harvard.edu/abs/2024ApJ...962..138S/abstract) for details about "
    "the braking law."
)

st.markdown("#### Model limitations")

st.markdown(
    "The neural networks were trained on models with a limited range of input parameters, "
    "so care should be taken when using **stelevator** to ensure that inputs remain within "
    "the training bounds. See the [paper](https://ui.adsabs.harvard.edu/abs/2024ApJ...962..138S/abstract) "
    "for full details on the model training and performance. Parameter boundaries are listed in Table 1. "
)

st.markdown(
    "Stellar tracks used for training were only computed up to the subgiant phase, and emulated outputs "
    "should not be trusted beyond this point. "
    "While the current parameter ranges are relatively limited in scope, we hope to expand the model grids "
    "in future releases."
)

st.markdown("⚠ **We strongly caution that extrapolation beyond the training bounds may lead to unphysical results.**")



st.markdown("---")

# ---------------------------------------------------------------------
# Helper: per-parameter sampling radio
# ---------------------------------------------------------------------
def sampling_radio(label: str, key: str) -> str:
    choice = st.radio(
        label,
        ["Uniform", "Truncated normal"],
        horizontal=True,
        key=key,
    )
    return "truncnorm" if choice == "Truncated normal" else "uniform"


# ---------------------------------------------------------------------
# Helper: validation for truncated normal parameters
# ---------------------------------------------------------------------
def check_truncnorm(name, mu, sigma, low, high, units=""):
    msgs = []
    if not (low <= mu <= high):
        msgs.append(
            f"{name} mean{units} must lie within [{low:.3g}, {high:.3g}] "
            f"(currently {mu:.3g})."
        )
    if sigma < 0:
        msgs.append(f"{name} sigma{units} must be non-negative.")
    elif sigma > (high - low):
        msgs.append(
            f"{name} sigma{units} should be ≤ range width ({high - low:.3g}); "
            f"currently {sigma:.3g}."
        )
    return msgs


# ---------------------------------------------------------------------
# Sidebar: global configuration
# ---------------------------------------------------------------------
st.sidebar.header("Configuration")
grid = st.sidebar.radio("Model grid", ["MESA", "YREC"])
mode = st.sidebar.radio("Mode", ["Population", "Track"])


# ---------------------------------------------------------------------
# Population mode
# ---------------------------------------------------------------------
if mode == "Population":
    st.subheader(f"{grid} population")

    nstars = st.number_input(
        "Number of stars",
        min_value=100,
        max_value=500_000,
        value=10_000,
        step=100,
        format="%d",
        help="More stars give smoother distributions but take longer to compute.",
    )
    fk = st.number_input(r"$f_K$: braking law strength", value=7.5, min_value=4.0, max_value=11.0, step=0.1)
    rocrit = st.number_input(r"$\textrm{Ro}_\textsf{crit}$: critical Rossby number", value=1.6, min_value=1.0, max_value=4.5, step=0.1)
    st.markdown("ℹ Note: if you do not want to model the effects of weakened magnetic braking (WMB), "
                r"set $\textrm{Ro}_\textsf{crit}$ to its maximum value of 4.5.")
    include_uncertainties = st.checkbox("Return estimated uncertainties in output table", value=False)

    st.markdown("### Parameter bounds & sampling")

    # --- Age [Gyr] ---
    sampling_age = sampling_radio("Age sampling", key="sampling_age")
    age_min, age_max = st.slider(
        "Age range [Gyr]",
        min_value=0.0,
        max_value=12.0,
        value=(0.0, 8.0),
        step=0.1,
    )
    if sampling_age == "truncnorm":
        cols = st.columns(2)
        with cols[0]:
            age_mu = st.number_input("Age mean [Gyr]", value=5.0, step=0.1)
        with cols[1]:
            age_sigma = st.number_input("Age sigma [Gyr]", value=5.0, min_value=0.0, step=0.1)
    else:
        age_mu, age_sigma = None, None

    # --- Mass [M_sun] ---
    sampling_M = sampling_radio("Mass sampling", key="sampling_M")
    M_min, M_max = st.slider(
        r"Mass range [M$_☉$]",
        min_value=0.8,
        max_value=1.2,
        value=(0.8, 1.2),
        step=0.01,
    )
    if sampling_M == "truncnorm":
        cols = st.columns(2)
        with cols[0]:
            M_mu = st.number_input(r"Mass mean [M$_☉$]", value=1.0, step=0.01)
        with cols[1]:
            M_sigma = st.number_input(r"Mass sigma [M$_☉$]", value=0.1, min_value=0.0, step=0.01)
    else:
        M_mu, M_sigma = None, None

    # --- [Fe/H] ---
    sampling_feh = sampling_radio("[Fe/H] sampling", key="sampling_feh")
    feh_min, feh_max = st.slider(
        "[Fe/H] range",
        min_value=-0.3,
        max_value=0.3,
        value=(-0.3, 0.3),
        step=0.05,
    )
    if sampling_feh == "truncnorm":
        cols = st.columns(2)
        with cols[0]:
            feh_mu = st.number_input("[Fe/H] mean", value=0.0, step=0.01)
        with cols[1]:
            feh_sigma = st.number_input("[Fe/H] sigma", value=0.2, min_value=0.0, step=0.01)
    else:
        feh_mu, feh_sigma = None, None

    # --- α_MLT ---
    sampling_alpha = sampling_radio(r"α$_\textsf{MLT}$ sampling", key="sampling_alpha")
    alpha_min, alpha_max = st.slider(
        r"α$_\textsf{MLT}$ range",
        min_value=1.4,
        max_value=2.0,
        value=(1.4, 2.0),
        step=0.05,
    )
    if sampling_alpha == "truncnorm":
        cols = st.columns(2)
        with cols[0]:
            alpha_mu = st.number_input(r"α$_\textsf{MLT}$ mean", value=1.6, step=0.01)
        with cols[1]:
            alpha_sigma = st.number_input(
                r"α$_\textsf{MLT}$ sigma", value=0.2, min_value=0.0, step=0.01
            )
    else:
        alpha_mu, alpha_sigma = None, None

    # --- Y_ini (MESA only) ---
    if grid == "MESA":
        sampling_Y = sampling_radio(r"Y$_\textsf{init}$ sampling", key="sampling_Y")
        Y_min, Y_max = st.slider(
            r"Y$_\textsf{init}$ range",
            min_value=0.22,
            max_value=0.28,
            value=(0.22, 0.28),
            step=0.005,
        )
        if sampling_Y == "truncnorm":
            cols = st.columns(2)
            with cols[0]:
                Y_mu = st.number_input(r"Y$_\textsf{init}$ mean", value=0.26, step=0.005)
            with cols[1]:
                Y_sigma = st.number_input(
                r"Y$_\textsf{init}$ sigma", value=0.02, min_value=0.0, step=0.005
            )
        else:
            Y_mu, Y_sigma = None, None
    else:
        Y_min, Y_max, Y_mu, Y_sigma, sampling_Y = None, None, None, None, None

    # Build bounds arrays
    age_bounds = [age_mu, age_sigma, age_min, age_max]
    M_bounds = [M_mu, M_sigma, M_min, M_max]
    feh_bounds = [feh_mu, feh_sigma, feh_min, feh_max]
    if grid == "MESA":
        Y_bounds = [Y_mu, Y_sigma, Y_min, Y_max]
    alpha_bounds = [alpha_mu, alpha_sigma, alpha_min, alpha_max]

    # Validation: only for parameters using truncated normal
    invalid_msgs: list[str] = []

    if sampling_age == "truncnorm":
        invalid_msgs += check_truncnorm(
            "Age", age_mu, age_sigma, age_min, age_max, " [Gyr]"
        )
    if sampling_M == "truncnorm":
        invalid_msgs += check_truncnorm(
            "Mass", M_mu, M_sigma, M_min, M_max, " [M☉]"
        )
    if sampling_feh == "truncnorm":
        invalid_msgs += check_truncnorm(
            "[Fe/H]", feh_mu, feh_sigma, feh_min, feh_max
        )
    if grid == "MESA" and sampling_Y == "truncnorm":
        invalid_msgs += check_truncnorm(
            "Y_ini", Y_mu, Y_sigma, Y_min, Y_max
        )
    if sampling_alpha == "truncnorm":
        invalid_msgs += check_truncnorm(
            "α_MLT", alpha_mu, alpha_sigma, alpha_min, alpha_max
        )

    if invalid_msgs:
        st.error("Some parameter settings are invalid:\n\n- " + "\n- ".join(invalid_msgs))

    params_valid = len(invalid_msgs) == 0
    run = st.button("Run emulator", disabled=not params_valid)

    if run and params_valid:
        try:
            if grid == "MESA":
                df = em.create_MESA_population(
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
                    sampling_age=sampling_age,
                    sampling_M=sampling_M,
                    sampling_feh=sampling_feh,
                    sampling_Y=sampling_Y,
                    sampling_alpha=sampling_alpha,
                )
            else:
                df = em.create_YREC_population(
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
                    sampling_age=sampling_age,
                    sampling_M=sampling_M,
                    sampling_feh=sampling_feh,
                    sampling_alpha=sampling_alpha,
                )

            st.success("Population generated.")

            # Plot in the original style
            fig = em.plot_sample(df, M_bounds=M_bounds, show=False)
            st.pyplot(fig)

            # --- Table + download ---
            st.subheader("First few rows of output")
            st.dataframe(df.head())

            # Download CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download population CSV",
                csv,
                file_name=f"{grid}_population.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error running emulator: {e}")


# ---------------------------------------------------------------------
# Track mode
# ---------------------------------------------------------------------
else:
    st.subheader(f"{grid} evolutionary track")

    npoints = st.number_input(
        "Number of points along track",
        min_value=100,
        max_value=100_000,
        value=10_000,
        step=100,
        format="%d",
    )

    age_min, age_max = st.slider(
        "Age range [Gyr]",
        min_value=0.0,
        max_value=14.0,
        value=(0.01, 10.0),
        step=0.01,
    )

    fk = st.number_input(r"$f_k$: braking law strength", value=7.6, min_value=4.0, max_value=11.0, step=0.1)
    rocrit = st.number_input(r"$\textrm{Ro}_\textsf{crit}$: critical Rossby number", value=1.6, min_value=1.0, max_value=4.5, step=0.1)
    st.markdown("ℹ Note: if you do not want to model the effects of weakened magnetic braking (WMB), "
                r"set $\textrm{Ro}_\textsf{crit}$ to its maximum value of 4.5.")
    
    M = st.number_input(r"Mass [M$_☉$]", value=1.0, step=0.01, min_value=0.8, max_value=1.2)
    feh = st.number_input("[Fe/H]", value=0.0, step=0.01, min_value=-0.3, max_value=0.3)
    alpha = st.number_input(r"α$_\textsf{MLT}$", value=1.6, step=0.01, min_value=1.4, max_value=2.0)

    if grid == "MESA":
        Y = st.number_input(r"Y$_\textsf{init}$", value=0.26, step=0.005, min_value=0.22, max_value=0.28)
    else:
        Y = None

    run = st.button("Run emulator")

    if run:
        try:
            if grid == "MESA":
                df = em.create_MESA_track(
                    npoints=npoints,
                    min_age=age_min,
                    max_age=age_max,
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
                df = em.create_YREC_track(
                    npoints=npoints,
                    min_age=age_min,
                    max_age=age_max,
                    plot=False,
                    save_dataframe=False,
                    M=M,
                    feh=feh,
                    alpha=alpha,
                    fk=fk,
                    rocrit=rocrit,
                )

            st.success("Track generated.")

            fig = em.plot_star(df, show=False)
            st.pyplot(fig)

            # --- Table + download ---
            st.subheader("First few rows of output")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download track CSV",
                csv,
                file_name=f"{grid}_track.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error running emulator: {e}")
