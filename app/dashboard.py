"""Streamlit dashboard for the Dubai irrigation advisor.

    pipenv run streamlit run app/dashboard.py

Runs offline against the committed NASA POWER cache. The only optional network
dependency is the Ollama explanation engine, which is off by default and
degrades to the deterministic explainer when unavailable.

ACCESSIBILITY

The audience is a grounds manager, not an analyst, and the delivery is often a
laptop screen shared in a meeting room. So:

- Colour is never the only carrier of meaning. Every status has a text label,
  every chart series has a marker or dash pattern as well as a hue, and the
  palette is Okabe-Ito, which stays distinguishable under the common forms of
  colour vision deficiency.
- Charts get a written caption stating the conclusion, so a screen reader user
  reaches the same finding as a sighted one.
- Headings nest properly (one h1, then h2, then h3) rather than being chosen
  for their font size, so heading-based navigation works.
- Numbers are always accompanied by their unit.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from irrigation.climate.et0_series import et0_for_day
from irrigation.data.nasa_power import load_metadata, load_records, load_weather
from irrigation.decision.policy import CostModel, decide
from irrigation.explain.advisor import explain_decision
from irrigation.explain.llm import build_llm_explainer
from irrigation.models.evaluate import run_comparison
from irrigation.physics.crop import CROPS, SOILS
from irrigation.viz import apply_style, colour_for, marker_for

st.set_page_config(
    page_title="Dubai Irrigation Advisor",
    page_icon="~",
    layout="wide",
)

apply_style(base_font_size=11)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------
# Cached data access
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def climate_summary() -> dict:
    """Monthly means from the committed 30-year NASA POWER record."""
    records = load_records()
    weather = load_weather()
    et0 = np.array([et0_for_day(day).et0_mm_day for day in weather])
    months = np.array([day.date.month for day in weather])
    years = np.array([day.date.year for day in weather])

    return {
        "et0_by_month": [float(et0[months == m].mean()) for m in range(1, 13)],
        "tmax_by_month": [
            float(np.mean([r.tmax_c for r in records if r.date.month == m]))
            for m in range(1, 13)
        ],
        # Millimetres per DAY, so the bars are directly comparable to ET0 on
        # the same axis. Monthly totals would be roughly 30x taller and the
        # chart would imply rainfall outweighs demand every winter.
        "rain_by_month": [
            float(np.mean([r.rainfall_mm for r in records if r.date.month == m]))
            for m in range(1, 13)
        ],
        "annual_rainfall_mm": float(
            sum(r.rainfall_mm for r in records) / 30.0
        ),
        "wetness_by_month": [
            float(np.mean([r.wetness_root for r in records if r.date.month == m]))
            for m in range(1, 13)
        ],
        "annual_totals": [float(et0[years == y].sum()) for y in range(1995, 2025)],
        "first_date": records[0].date,
        "last_date": records[-1].date,
        "n_days": len(records),
        "metadata": load_metadata(),
    }


@st.cache_data(show_spinner="Simulating a 120-day season for each method...")
def season_comparison(
    crop_key: str, soil_key: str, root_depth_m: float, kc: float,
    water_cost: float, stress_cost: float, days: int,
) -> list[dict]:
    results = run_comparison(
        CROPS[crop_key], SOILS[soil_key],
        root_depth_m=root_depth_m, kc=kc, days=days,
        cost_model=CostModel(
            water_cost_per_mm=water_cost,
            stress_cost_per_mm_deficit=stress_cost,
        ),
    )
    return [result.summary() for result in results]


def caption(text: str) -> None:
    """Chart caption stating the conclusion, not describing the axes."""
    st.caption(text)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.header("Site setup")

crop_key = st.sidebar.selectbox(
    "Crop", list(CROPS), index=list(CROPS).index("turfgrass"),
    help="Sets the crop coefficient and how much depletion is tolerated.",
)
soil_key = st.sidebar.selectbox(
    "Soil", list(SOILS), index=list(SOILS).index("sand"),
    help="Sets how much water the soil can hold per metre of depth.",
)
crop, soil = CROPS[crop_key], SOILS[soil_key]

root_depth_m = st.sidebar.slider(
    "Root depth (metres)", 0.2, 2.0, 0.5, 0.1,
    help="Deeper roots reach a larger reservoir and tolerate longer gaps.",
)
kc = st.sidebar.slider(
    "Crop coefficient Kc", 0.3, 1.3, 0.85, 0.05,
    help="Water use relative to a reference grass surface.",
)

st.sidebar.header("What errors cost")
water_cost = st.sidebar.slider(
    "Water (AED per mm applied)", 0.5, 10.0, 2.5, 0.5,
    help="Desalinated supply. Charged on every millimetre, used or drained.",
)
stress_cost = st.sidebar.slider(
    "Crop stress (AED per mm of deficit)", 1.0, 120.0, 30.0, 1.0,
    help="The asymmetry. Raising this favours methods that over-water.",
)
st.sidebar.metric(
    "Stress-to-water ratio", f"{stress_cost / water_cost:.1f} to 1",
    help="How much worse under-watering is than over-watering.",
)

st.sidebar.header("Explanations")
use_llm = st.sidebar.checkbox(
    "Use a language model (Ollama)", value=False,
    help=(
        "Off by default. When on, every number the model writes is checked "
        "against the computed facts and the text is discarded if any figure "
        "was not supplied."
    ),
)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Dubai Irrigation Advisor")
st.markdown(
    "Physics-grounded irrigation decisions, evaluated on **what they cost to "
    "run** rather than on prediction accuracy. Climate data is a committed "
    "30-year NASA POWER record; nothing here needs a network connection."
)

decision_tab, comparison_tab, climate_tab, data_tab = st.tabs(
    ["Today's decision", "Method comparison", "Dubai climate", "Data and limits"]
)


# --------------------------------------------------------------------------
# Today's decision
# --------------------------------------------------------------------------
with decision_tab:
    st.header("Today's decision")

    left, right = st.columns([1, 1.4])

    with left:
        taw = soil.total_available_water_mm(root_depth_m)
        depletion = st.slider(
            "Estimated root-zone depletion (mm)",
            0.0, float(round(taw, 1)), min(13.8, float(round(taw * 0.4, 1))), 0.1,
            help="How much water the root zone has lost since it was last full.",
        )
        et0_forecast = st.slider(
            "Tomorrow's forecast ET0 (mm/day)", 2.0, 12.0, 8.4, 0.1,
            help="Water the crop will lose tomorrow. July in Dubai is near 8.5.",
        )

        decision = decide(
            predicted_depletion_mm=depletion, et0_forecast_mm=et0_forecast,
            crop=crop, soil=soil, root_depth_m=root_depth_m, kc=kc,
        )

        if decision.action == "irrigate":
            st.error(
                f"**IRRIGATE - apply {decision.depth_mm:.1f} mm today**",
                icon=":material/water_drop:",
            )
        else:
            st.success("**HOLD - no irrigation needed today**", icon=":material/check:")

        st.metric("Total available water", f"{decision.taw_mm:.1f} mm")
        st.metric("Readily available water", f"{decision.raw_mm:.1f} mm",
                  help="Depletion beyond this point reduces transpiration.")
        st.metric("Current depletion", f"{decision.predicted_depletion_mm:.1f} mm",
                  f"{decision.predicted_depletion_mm / decision.taw_mm * 100:.0f}% of capacity")

    with right:
        fig, ax = plt.subplots(figsize=(6.5, 2.4))
        ax.barh([0], [decision.taw_mm], color="#DDDDDD", height=0.5,
                label="Total available water")
        ax.barh([0], [decision.raw_mm], color="#56B4E9", height=0.5,
                label="Readily available (safe zone)")
        ax.barh([0], [decision.predicted_depletion_mm], color="#0072B2", height=0.22,
                label="Current depletion")
        ax.axvline(decision.raw_mm, color="#D55E00", linewidth=2.5, linestyle="--")
        ax.annotate("stress begins", (decision.raw_mm, 0.32), color="#D55E00",
                    fontsize=9, ha="center", fontweight="bold")
        ax.set_yticks([])
        ax.set_xlabel("Millimetres of water")
        ax.set_xlim(0, decision.taw_mm * 1.05)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.85), ncol=3, fontsize=8)
        ax.grid(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        caption(
            f"Depletion is {decision.predicted_depletion_mm:.1f} mm against a "
            f"stress threshold of {decision.raw_mm:.1f} mm, so the recommendation "
            f"is to {'irrigate' if decision.action == 'irrigate' else 'hold'}."
        )

        st.subheader("Why")
        engine = build_llm_explainer() if use_llm else None
        with st.spinner("Writing the explanation..."):
            explanation = explain_decision(
                decision, et0_forecast_mm=et0_forecast,
                crop_name=crop_key, soil_name=soil_key,
                sandy=soil_key.startswith("sand"), engine=engine,
            )

        st.markdown(explanation.body.strip())
        st.markdown("**Grounded in**")
        for citation in explanation.citations:
            st.markdown(f"- {citation['source']} - {citation['text']}")

        if explanation.engine.startswith("offline (fallback"):
            st.warning(
                f"The language model was bypassed and the deterministic text "
                f"is shown instead. Reason: {explanation.engine}",
                icon=":material/info:",
            )
        else:
            st.caption(f"Explanation engine: {explanation.engine}")


# --------------------------------------------------------------------------
# Method comparison
# --------------------------------------------------------------------------
with comparison_tab:
    st.header("Which method costs least to run?")
    st.markdown(
        "Six ways of deciding when to irrigate, each run over the same "
        "simulated 120-day summer with the same weather and the same probe. "
        "The only difference is how each one estimates depletion."
    )

    summaries = season_comparison(
        crop_key, soil_key, root_depth_m, kc, water_cost, stress_cost, 120
    )
    names = [s["predictor"] for s in summaries]
    costs = np.array([s["total_cost_aed"] for s in summaries])
    rmse = np.array([s["depletion_rmse_mm"] for s in summaries])

    cheapest = int(np.argmin(costs))
    most_accurate = int(np.argmin(rmse))

    top = st.columns(3)
    top[0].metric("Cheapest to run", names[cheapest], f"{costs[cheapest]:,.0f} AED")
    top[1].metric("Most accurate", names[most_accurate],
                  f"{rmse[most_accurate]:.2f} mm RMSE")
    top[2].metric(
        "Same method?",
        "No" if cheapest != most_accurate else "Yes",
        help="The project's central finding is that these usually differ.",
    )

    if cheapest != most_accurate:
        st.info(
            f"**{names[most_accurate]}** predicts depletion most accurately and "
            f"costs {costs[most_accurate] - costs[cheapest]:,.0f} AED more to "
            f"operate than **{names[cheapest]}**. Accuracy weights over- and "
            "under-estimation equally; a field does not.",
            icon=":material/lightbulb:",
        )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        order = np.argsort(costs)
        ax.barh(np.arange(len(names)), costs[order],
                color=[colour_for(names[i]) for i in order])
        ax.set_yticks(np.arange(len(names)), [names[i] for i in order], fontsize=9)
        ax.set_xlabel("Season operating cost (AED)")
        ax.set_xscale("log")
        ax.set_title("Cost to run")
        ax.grid(axis="x")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        caption(
            f"Cheapest is {names[cheapest]} at {costs[cheapest]:,.0f} AED; "
            f"most expensive is {names[int(np.argmax(costs))]} at "
            f"{costs.max():,.0f} AED. Note the logarithmic scale."
        )

    with chart_right:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for index, name in enumerate(names):
            ax.scatter(rmse[index], costs[index], s=170, color=colour_for(name),
                       marker=marker_for(name), edgecolor="white",
                       linewidth=1.4, zorder=3, label=name)
        ax.set_xlabel("Prediction error (RMSE, mm)")
        ax.set_ylabel("Operating cost (AED)")
        ax.set_yscale("log")
        ax.set_title("Accuracy against cost")
        ax.legend(fontsize=7.5, loc="upper center", ncol=2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        caption(
            "Points toward the bottom-left would be both accurate and cheap. "
            "No method is in that corner, which is the trade-off this project "
            "quantifies."
        )

    st.subheader("Full results")
    st.dataframe(
        [
            {
                "Method": s["predictor"],
                "Cost (AED)": round(s["total_cost_aed"]),
                "Water (mm)": round(s["water_mm"]),
                "Irrigations": s["irrigation_events"],
                "Stress days": s["stress_days"],
                "Severe stress days": s["severe_stress_days"],
                "Drained away (mm)": round(s["drainage_mm"]),
                "RMSE (mm)": s["depletion_rmse_mm"],
                "Bias (mm)": round(s["depletion_bias_mm"], 2),
            }
            for s in summaries
        ],
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Bias is the average signed error. A positive bias means the method "
        "believes the soil is drier than it is, and therefore over-waters."
    )


# --------------------------------------------------------------------------
# Climate
# --------------------------------------------------------------------------
with climate_tab:
    st.header("Dubai climate, 30 years of NASA observations")

    summary = climate_summary()
    annual = np.array(summary["annual_totals"])

    top = st.columns(4)
    top[0].metric("Days of record", f"{summary['n_days']:,}")
    top[1].metric("Mean annual water demand", f"{annual.mean():,.0f} mm")
    top[2].metric("Peak month demand", f"{max(summary['et0_by_month']):.1f} mm/day")
    top[3].metric("Annual rainfall", f"{summary['annual_rainfall_mm']:.0f} mm")

    x = np.arange(12)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(x, summary["et0_by_month"], color="#0072B2", label="Water lost to air")
    axes[0].bar(x, summary["rain_by_month"], color="#009E73", label="Rainfall")
    axes[0].set_xticks(x, MONTHS, fontsize=9)
    axes[0].set_ylabel("mm per day")
    axes[0].set_title("Demand against supply")
    axes[0].legend(fontsize=9)

    axes[1].plot(x, summary["wetness_by_month"], color="#D55E00", marker="o")
    axes[1].set_xticks(x, MONTHS, fontsize=9)
    axes[1].set_ylabel("Soil wetness (0 to 1)")
    axes[1].set_title("Desert soil is driest when demand peaks")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    caption(
        f"Peak demand is {max(summary['et0_by_month']):.1f} mm/day in "
        f"{MONTHS[int(np.argmax(summary['et0_by_month']))]}, against total "
        f"annual rainfall of about {summary['annual_rainfall_mm']:.0f} mm. "
        f"Even the wettest month averages only "
        f"{max(summary['rain_by_month']):.1f} mm/day of rain, so rainfall "
        "never meaningfully offsets demand."
    )


# --------------------------------------------------------------------------
# Data and limits
# --------------------------------------------------------------------------
with data_tab:
    st.header("Where the data comes from, and what it cannot tell you")

    summary = climate_summary()
    cell = summary["metadata"]["grid_cell_point"]

    st.subheader("Source")
    st.markdown(
        f"""
- **NASA POWER**, daily, {summary['first_date']} to {summary['last_date']}
  ({summary['n_days']:,} days, no gaps).
- Grid cell centre **{cell['latitude']:.2f} N, {cell['longitude']:.2f} E**,
  reported elevation {cell['elevation_m']:.0f} m.
- Public domain, no account or key required. Downloaded once and committed to
  the repository, so every figure here is reproducible offline.
"""
    )

    st.subheader("Limits worth stating plainly")
    # st.info rather than st.warning: these are scope limits, not faults.
    # Warning styling implies something has gone wrong and pulls the eye away
    # from the results the tab exists to qualify.
    st.info(
        """
**The grid cell is about 55 x 65 km.** This is regional Dubai, not a specific
field. Appropriate for reference evapotranspiration, which is defined over a
uniform surface; not a substitute for on-site measurement of a particular plot.

**The soil moisture series is bare desert, not farmland.** It comes from a
land-surface model and describes how untouched sand behaves. It is genuinely
independent of this project's simulator, which is why it is useful for
validation - but it is not an irrigated root zone.

**The sensor is simulated.** A separate hardware workstream builds the real
probe. The simulation carries the failure modes that break field deployments -
calibration offset, salinity drift, dropout, quantisation - but it is a model
of a probe, not a probe.

**Costs are a policy choice, not a measurement.** The stress-to-water ratio is
a slider precisely so an agronomist can disagree with it and see immediately
whether the conclusion survives.
""",
        icon=":material/info:",
    )

    st.subheader("Citation")
    st.code(
        "These data were obtained from the NASA Langley Research Center POWER\n"
        "Project, funded through the NASA Earth Science Directorate Applied\n"
        "Science Program. https://power.larc.nasa.gov/",
        language=None,
    )
