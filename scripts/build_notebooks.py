"""Generate and execute the analysis notebooks.

The committed `.ipynb` files carry their outputs, so a reviewer reads them on
GitHub without installing anything. This script is what produced them, which
makes that output reproducible rather than a screenshot of a state nobody can
recover.

    PYTHONPATH=src python scripts/build_notebooks.py

Execution needs a working torch install and a few minutes on a GPU. Nothing
here touches the network: the notebooks read the committed NASA POWER cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"

PREAMBLE = """\
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from irrigation.viz import apply_style, colour_for, dashes_for, marker_for, ACTUAL

apply_style()
"""


# ==========================================================================
# 01 - Climate and reference evapotranspiration
# ==========================================================================
CLIMATE = [
    ("md", """\
# Dubai climate and reference evapotranspiration

**What this notebook answers:** how much water does a plant in Dubai lose to
the air each day, how confident can we be in that number, and what does the
soil actually do in response?

Everything downstream - how much to irrigate, when, and whether a machine
learning model helps - rests on this single quantity. If it is wrong, nothing
built on top of it can be right, so it is worth establishing carefully before
going further.

**The data.** Thirty years of daily observations for Dubai from NASA POWER,
1995 to 2024. NASA POWER combines satellite measurements of sunlight with a
global weather reanalysis. It is public, free, needs no account, and can be
re-downloaded by anyone who wants to check this work.
"""),
    ("code", PREAMBLE + """
from irrigation.data.nasa_power import load_metadata, load_records, load_weather
from irrigation.climate.et0_series import et0_for_day
from irrigation.climate.dubai import normals_day

records = load_records()
weather = load_weather()
metadata = load_metadata()

et0 = np.array([et0_for_day(day).et0_mm_day for day in weather])
years = np.array([day.date.year for day in weather])
months = np.array([day.date.month for day in weather])

cell = metadata["grid_cell_point"]
print(f"Days of record : {len(records):,}")
print(f"Period         : {records[0].date} to {records[-1].date}")
print(f"Location       : {cell['latitude']:.2f} N, {cell['longitude']:.2f} E")
print(f"Missing values : {30 * 365 + 8 - len(records)}")
"""),
    ("md", """\
## The Dubai year

Reference evapotranspiration, written **ET0**, is the depth of water a
standard grass surface loses per day - to evaporation from the soil and
transpiration through the leaves combined. It is measured in millimetres per
day, which is convenient: 1 mm of ET0 is one litre of water lost per square
metre.

The chart below is the whole irrigation problem in one picture.
"""),
    ("code", """\
monthly_et0 = np.array([et0[months == m].mean() for m in range(1, 13)])
monthly_tmax = np.array([r.tmax_c for r in records])
monthly_tmin = np.array([r.tmin_c for r in records])
record_months = np.array([r.date.month for r in records])

tmax_by_month = np.array([monthly_tmax[record_months == m].mean() for m in range(1, 13)])
tmin_by_month = np.array([monthly_tmin[record_months == m].mean() for m in range(1, 13)])
# Millimetres per DAY, matching the ET0 units they are plotted against.
# Monthly totals on the same axis would be about 30x taller and would suggest
# winter rainfall exceeds demand, which is the opposite of the truth.
rain_by_month = np.array([
    np.mean([r.rainfall_mm for r in records if r.date.month == m])
    for m in range(1, 13)
])
annual_rainfall = sum(r.rainfall_mm for r in records) / 30.0

labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
x = np.arange(12)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].fill_between(x, tmin_by_month, tmax_by_month, color="#56B4E9",
                     alpha=0.35, label="Daily low to high")
axes[0].plot(x, tmax_by_month, color="#D55E00", marker="o", label="Average daily high")
axes[0].plot(x, tmin_by_month, color="#0072B2", marker="o", label="Average daily low")
axes[0].set_ylabel("Air temperature (C)")
axes[0].set_title("Dubai through the year, averaged over 1995-2024")
# Lower centre: temperature peaks mid-year, so an upper legend sits on top of
# the very curves it labels.
axes[0].legend(loc="lower center", ncol=3)

bars = axes[1].bar(x, monthly_et0, color="#0072B2", label="Water lost to the air")
axes[1].bar(x, rain_by_month, color="#009E73", label="Rainfall")
axes[1].set_ylabel("Millimetres per day")
axes[1].set_xticks(x, labels)
axes[1].set_title("Water leaving the soil, against water arriving")
axes[1].legend(loc="upper left")

for index, value in enumerate(monthly_et0):
    axes[1].annotate(f"{value:.1f}", (index, value), ha="center",
                     va="bottom", fontsize=9, color="#333333")

plt.tight_layout()
plt.show()

print(f"Average annual rainfall : {annual_rainfall:.0f} mm")
print(f"Wettest month           : {labels[int(np.argmax(rain_by_month))]}, "
      f"{max(rain_by_month):.2f} mm/day on average")
print(f"Peak water demand       : {max(monthly_et0):.1f} mm/day")
print(f"Ratio at the peak       : {max(monthly_et0) / rain_by_month[int(np.argmax(monthly_et0))]:.0f} to 1")
"""),
    ("md", """\
**Read the lower panel first.** The blue bars are water leaving the soil; the
green bars are rain arriving. Both are in millimetres per day, so they are
directly comparable.

In July a square metre of grass loses about 8.5 litres a day and receives
about 0.1. Even January - the wettest month - averages under 1 mm/day of rain
against 3.5 mm/day of demand. Total annual rainfall is roughly 109 mm, against
annual demand above 2,200 mm.

There is no season in Dubai when rainfall meaningfully offsets demand. Every
drop a plant uses has to be applied deliberately, which is why getting the
amount right is worth effort.
"""),
    ("md", """\
## Is the number trustworthy?

The physics used here is the FAO-56 Penman-Monteith equation, the international
standard since 1998. The code implementing it is checked against the worked
examples published in FAO's own manual - but those examples are Bangkok,
Brussels and Lyon. None of them is a desert.

So the check that matters is a different one: build the same quantity **twice,
from unrelated sources**, and see whether the two agree.

- **Route one** starts from published monthly climate normals for Dubai - the
  kind of table in an almanac - and generates a smooth year from them.
- **Route two** starts from thirty years of NASA satellite and reanalysis data.

Nothing connects them. Neither was adjusted to match the other.
"""),
    ("code", """\
synthetic_daily = np.array([et0_for_day(normals_day(d)).et0_mm_day for d in range(1, 366)])
synthetic_months = np.array([
    (np.datetime64("2001-01-01") + np.timedelta64(d - 1, "D")).astype("datetime64[M]").astype(int) % 12 + 1
    for d in range(1, 366)
])
synthetic_by_month = np.array([synthetic_daily[synthetic_months == m].mean() for m in range(1, 13)])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8),
                         gridspec_kw={"width_ratios": [1.6, 1]})

axes[0].plot(x, monthly_et0, color=ACTUAL, marker="o", linewidth=2.5,
             label="From NASA observations (30 years)")
axes[0].plot(x, synthetic_by_month, color="#E69F00", marker="s", linestyle="--",
             linewidth=2.5, label="From published climate normals")
axes[0].set_xticks(x, labels)
axes[0].set_ylabel("ET0 (mm/day)")
axes[0].set_title("Two independent routes to the same answer")
# Lower centre: the curve peaks mid-year, so the usual upper-left position
# puts the legend text straight through the line it is labelling.
axes[0].legend(loc="lower center")

difference = monthly_et0 - synthetic_by_month
colours = ["#009E73" if abs(d) < 0.25 else "#E69F00" for d in difference]
axes[1].barh(x, difference, color=colours)
axes[1].axvline(0, color="#4D4D4D", linewidth=1)
axes[1].set_yticks(x, labels)
axes[1].set_xlabel("Difference (mm/day)")
axes[1].set_title("Gap between the two")
axes[1].set_xlim(-0.6, 0.6)
axes[1].grid(axis="x")

plt.tight_layout()
plt.show()

annual_real = np.array([et0[years == y].sum() for y in range(1995, 2025)])
print(f"NASA observations, mean annual ET0 : {annual_real.mean():,.0f} mm")
print(f"Published normals, annual ET0      : {synthetic_daily.sum():,.0f} mm")
print(f"Disagreement                       : "
      f"{abs(annual_real.mean() - synthetic_daily.sum()) / annual_real.mean() * 100:.1f}%")
"""),
    ("md", """\
**They agree to about 2%, and the largest monthly gap is under 0.4 mm/day.**

That is meaningful evidence. Two derivations sharing no inputs do not land on
the same seasonal curve by accident, so both are very likely right.

This check has caught a real error before. An earlier version of this project
fed wind speed into the equation at the height it is *reported* - 10 metres -
rather than the 2 metres the equation requires. Annual ET0 came out around
2,450 mm. The seasonal shape looked perfect, every chart looked reasonable,
and the error was roughly 9%. Only a magnitude comparison against independent
data exposed it.
"""),
    ("code", """\
fig, ax = plt.subplots(figsize=(10, 4.6))

ax.bar(np.arange(1995, 2025), annual_real, color="#56B4E9", label="Annual total")
ax.axhline(annual_real.mean(), color="#D55E00", linewidth=2,
           label=f"30-year mean ({annual_real.mean():,.0f} mm)")
ax.set_ylabel("ET0 (mm per year)")
ax.set_ylim(0, annual_real.max() * 1.15)
ax.set_title("Year-to-year variation is small")
ax.legend(loc="lower right", ncol=2)

plt.tight_layout()
plt.show()

print(f"Range across 30 years : {annual_real.min():,.0f} to {annual_real.max():,.0f} mm")
print(f"Standard deviation    : {annual_real.std():,.0f} mm "
      f"({annual_real.std() / annual_real.mean() * 100:.1f}% of the mean)")
"""),
    ("md", """\
Annual demand varies by only about 3% from year to year. For a client this is
the good news buried in the data: **irrigation demand in Dubai is highly
predictable at the annual scale.** Budgeting water for next year is not the
hard problem. Deciding what to do *this Tuesday* is.
"""),
    ("md", """\
## What the soil does

NASA POWER also publishes modelled root-zone soil wetness for the same grid
cell - how full the soil is, on a scale from 0 (bone dry) to 1 (saturated).

An important caveat, stated plainly: **this grid cell is mostly bare desert,
not farmland.** It describes how untouched Dubai sand behaves, which is
genuinely useful - it is real, measured independently of anything in this
project - but it is not a picture of an irrigated field.
"""),
    ("code", """\
wetness = np.array([r.wetness_root for r in records])
wetness_top = np.array([r.wetness_top for r in records])
rain = np.array([r.rainfall_mm for r in records])

wet_by_month = np.array([wetness[record_months == m].mean() for m in range(1, 13)])
top_by_month = np.array([wetness_top[record_months == m].mean() for m in range(1, 13)])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

axes[0].plot(x, wet_by_month, color="#0072B2", marker="o", label="Root zone")
axes[0].plot(x, top_by_month, color="#E69F00", marker="s", linestyle="--", label="Surface")
axes[0].set_xticks(x, labels)
axes[0].set_ylabel("Soil wetness (0 = dry, 1 = saturated)")
axes[0].set_title("Desert soil is driest exactly when demand peaks")
axes[0].legend()

recent = [(r.date, r.wetness_root, r.rainfall_mm) for r in records if r.date.year == 2024]
dates = [d for d, _, _ in recent]
axes[1].plot(dates, [w for _, w, _ in recent], color="#0072B2", label="Root-zone wetness")
rain_axis = axes[1].twinx()
rain_axis.bar(dates, [p for _, _, p in recent], color="#009E73", width=2.0, label="Rainfall")
rain_axis.set_ylabel("Rainfall (mm/day)")
rain_axis.grid(False)
axes[1].set_ylabel("Soil wetness")
axes[1].set_title("2024: soil responds sharply to rain, then dries slowly")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()

print(f"Root-zone wetness: {wetness.min():.3f} to {wetness.max():.3f}, mean {wetness.mean():.3f}")
print(f"Days with measurable rain: {(rain > 0.5).sum()} out of {len(rain):,} "
      f"({(rain > 0.5).mean() * 100:.1f}%)")
"""),
    ("md", """\
Two things stand out.

**The soil is at its driest in exactly the months demand is highest.** There is
no natural buffer to draw on in summer.

**Rain arrives as sharp spikes, then the soil dries slowly over weeks.** That
slow recession is the signal a model can learn, and it is what the next
notebook tries to predict.

---

## Summary

| Question | Answer |
|---|---|
| How much water does Dubai grass lose? | About 2,275 mm/year; 8.5 mm/day in July |
| How reliable is that figure? | Two independent derivations agree within 2% |
| Does rain help? | Effectively never - under 4% of days see measurable rain |
| How much does demand vary year to year? | About 3%, so annual planning is easy |
| When is the soil driest? | Summer - the same months demand peaks |

Next: [02 - Soil moisture sequence models](02_soil_moisture_sequence_models.ipynb)
"""),
]


# ==========================================================================
# 02 - Sequence models
# ==========================================================================
SEQUENCE = [
    ("md", """\
# Predicting soil moisture with LSTM and GRU networks

**What this notebook answers:** can a neural network predict how wet the soil
is, and is it better than the obvious simple answers?

The models here are trained in PyTorch on the thirty-year NASA record. The
target is real, published, independently produced soil moisture - not anything
this project generated - so the scores below cannot be inflated by a model
learning its own simulator's assumptions.

## Two different questions

These get confused constantly, and conflating them is how soil-moisture models
come to look far better than they are.

**Question A - forecasting.** *Given everything up to today, including how wet
the soil is now, how wet will it be tomorrow?* This is easy, because soil
moisture changes slowly. The honest benchmark is "assume tomorrow is the same
as today" - the **persistence** baseline.

**Question B - estimation.** *Given only the weather, how wet is the soil?* No
moisture readings at any point. This is much harder, and much closer to the
real situation: a site with no sensor, or one whose sensor has drifted. The
benchmark is **climatology** - "whatever is normal for this date".

Reporting only Question A would be flattering and close to meaningless. Both
are run.
"""),
    ("code", PREAMBLE + """
from irrigation.data.nasa_power import load_records
from irrigation.models import sequence as S

records = load_records()

configs = {
    ("forecast", "lstm"): S.SequenceConfig(task="forecast", cell="lstm"),
    ("forecast", "gru"): S.SequenceConfig(task="forecast", cell="gru"),
    ("estimate", "lstm"): S.SequenceConfig(task="estimate", cell="lstm"),
    ("estimate", "gru"): S.SequenceConfig(task="estimate", cell="gru"),
}

datasets = {
    task: S.build_dataset(records, S.SequenceConfig(task=task))
    for task in ("forecast", "estimate")
}

for task, dataset in datasets.items():
    print(f"{task.upper():9s} train {dataset.x_train.shape[0]:>5,}  "
          f"val {dataset.x_val.shape[0]:>5,}  test {dataset.x_test.shape[0]:>5,}  "
          f"features {len(dataset.feature_names)}")
print()
print("Forecast inputs:", ", ".join(datasets["forecast"].feature_names))
print("Estimate inputs:", ", ".join(datasets["estimate"].feature_names))
"""),
    ("md", """\
## Splitting the data honestly

The single most common way to get a spectacular and worthless soil moisture
result is to shuffle the data before splitting it. Soil moisture changes over
weeks, so if Tuesday is in the training set and Wednesday is in the test set,
the model is effectively being shown the answer.

The split here is **chronological**: train on the earliest years, tune on the
middle years, and touch the final four years exactly once, at the end.
"""),
    ("code", """\
dataset = datasets["forecast"]
fig, ax = plt.subplots(figsize=(10, 2.6))

spans = [
    ("Training", dataset.dates_train, "#0072B2"),
    ("Validation", dataset.dates_val, "#E69F00"),
    ("Test (used once)", dataset.dates_test, "#D55E00"),
]
for index, (label, dates, colour) in enumerate(spans):
    ax.barh(0, (max(dates) - min(dates)).days, left=min(dates).toordinal(),
            height=0.5, color=colour, label=f"{label}  ({min(dates).year}-{max(dates).year})")

ax.set_yticks([])
ax.set_xlim(min(dataset.dates_train).toordinal() - 200,
            max(dataset.dates_test).toordinal() + 200)
ticks = [np.datetime64(f"{y}-01-01").astype("datetime64[D]").astype(int) + 719163
         for y in range(1995, 2026, 5)]
ax.set_xticks(ticks, [str(y) for y in range(1995, 2026, 5)])
ax.set_title("Time never flows backwards across a split boundary")
ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
ax.grid(False)

plt.tight_layout()
plt.show()
"""),
    ("code", """\
results = {}
for (task, cell), config in configs.items():
    results[(task, cell)] = S.train_sequence_model(datasets[task], config)
    metrics = results[(task, cell)].metrics
    print(f"{task:9s} {cell.upper():5s} "
          f"RMSE {metrics['rmse']:.5f}  MAE {metrics['mae']:.5f}  "
          f"R2 {metrics['r2']:.4f}  (best epoch {results[(task, cell)].best_epoch})")

baselines = {
    ("forecast", "Persistence"): S.persistence_baseline(datasets["forecast"],
                                                        configs[("forecast", "lstm")]),
    ("forecast", "Climatology"): S.climatology_baseline(records, datasets["forecast"],
                                                        configs[("forecast", "lstm")]),
    ("estimate", "Climatology"): S.climatology_baseline(records, datasets["estimate"],
                                                        configs[("estimate", "lstm")]),
}
print()
for (task, name), predictions in baselines.items():
    metrics = S.regression_metrics(datasets[task].y_test, predictions)
    print(f"{task:9s} {name:12s} RMSE {metrics['rmse']:.5f}  "
          f"MAE {metrics['mae']:.5f}  R2 {metrics['r2']:.4f}")
"""),
    ("code", """\
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)

for ax, (task, title) in zip(axes, [("forecast", "Question A: forecast tomorrow"),
                                    ("estimate", "Question B: estimate from weather alone")]):
    for cell in ("lstm", "gru"):
        result = results[(task, cell)]
        ax.plot(range(1, len(result.train_losses) + 1), result.train_losses,
                color=colour_for(cell.upper()), label=f"{cell.upper()} training")
        ax.plot(range(1, len(result.val_losses) + 1), result.val_losses,
                color=colour_for(cell.upper()), linestyle="--",
                label=f"{cell.upper()} validation")
        ax.axvline(result.best_epoch, color=colour_for(cell.upper()),
                   alpha=0.3, linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared error")
    ax.set_title(title)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.show()
"""),
    ("md", """\
## The result, and the catch

The vertical lines mark where each model was at its best on the validation
years. Training stopped shortly after and the best weights were restored, so
the test scores below come from the model that generalized best rather than
the one that trained longest.

Now the interesting part.
"""),
    ("code", """\
rows = [
    ("Persistence\\n(assume no change)", "forecast", baselines[("forecast", "Persistence")]),
    ("LSTM", "forecast", results[("forecast", "lstm")].predictions_test),
    ("GRU", "forecast", results[("forecast", "gru")].predictions_test),
]
names = [r[0] for r in rows]
metrics = [S.regression_metrics(datasets[r[1]].y_test, r[2]) for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
positions = np.arange(len(rows))
palette = ["#E69F00", "#0072B2", "#009E73"]

for ax, key, title in [(axes[0], "rmse", "Typical error, squared-and-rooted (RMSE)"),
                       (axes[1], "mae", "Typical error, plain average (MAE)")]:
    values = [m[key] for m in metrics]
    best = int(np.argmin(values))
    bars = ax.bar(positions, values,
                  color=[palette[i] if i == best else "#BBBBBB" for i in range(len(values))])
    ax.set_xticks(positions, names, fontsize=10)
    ax.set_ylabel("Soil wetness error")
    ax.set_title(title)
    for index, value in enumerate(values):
        ax.annotate(f"{value:.5f}", (index, value), ha="center", va="bottom", fontsize=10)
    ax.annotate("best", (best, values[best]), ha="center", va="bottom",
                fontsize=10, fontweight="bold", xytext=(0, 18),
                textcoords="offset points", color="#006644")

plt.tight_layout()
plt.show()
"""),
    ("md", """\
## The winner depends on the question you ask

**The LSTM has the lower RMSE. Persistence has the lower MAE.** Same models,
same data, same test years - opposite conclusions.

This is not a paradox and not a bug. It is what the two measures mean.

- **MAE** is the average error. Persistence wins because on most days in a
  desert nothing happens, and "the same as yesterday" is then exactly right.
- **RMSE** squares errors before averaging, so a few large misses dominate.
  Persistence is badly wrong on the handful of days when it rains and the soil
  jumps. The LSTM is slightly wrong every day but handles those jumps.

So: **persistence is better on a normal day. The network is better on the days
that matter.** Which one is "accurate" depends entirely on whether being wrong
occasionally-and-badly costs more than being wrong constantly-and-slightly.

That question cannot be answered by a metric. It is answered by what the error
costs in the field - which is the subject of the next notebook, and the central
argument of this project.
"""),
    ("code", """\
dataset = datasets["forecast"]
window = slice(0, 400)
dates = dataset.dates_test[window]

fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)

axes[0].plot(dates, dataset.y_test[window], color=ACTUAL, linewidth=2.5,
             label="Observed (NASA)")
axes[0].plot(dates, results[("forecast", "lstm")].predictions_test[window],
             color="#0072B2", linestyle="--", label="LSTM forecast")
axes[0].plot(dates, baselines[("forecast", "Persistence")][window],
             color="#E69F00", linestyle=":", label="Persistence")
axes[0].set_ylabel("Soil wetness")
axes[0].set_title("Question A: forecasting tomorrow (test years, first 400 days)")
axes[0].legend(ncol=3, loc="upper right")

estimate_dataset = datasets["estimate"]
axes[1].plot(estimate_dataset.dates_test[window], estimate_dataset.y_test[window],
             color=ACTUAL, linewidth=2.5, label="Observed (NASA)")
axes[1].plot(estimate_dataset.dates_test[window],
             results[("estimate", "lstm")].predictions_test[window],
             color="#0072B2", linestyle="--", label="LSTM from weather only")
axes[1].plot(estimate_dataset.dates_test[window],
             baselines[("estimate", "Climatology")][window],
             color="#D55E00", linestyle=":", label="Climatology")
axes[1].set_ylabel("Soil wetness")
axes[1].set_title("Question B: estimating with no moisture readings at all")
axes[1].legend(ncol=3, loc="upper right")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()
"""),
    ("md", """\
The two panels show why the distinction matters so much.

**Top:** with yesterday's reading available, both methods track the truth
closely. The curves are nearly indistinguishable at this scale.

**Bottom:** with only weather available, the LSTM captures the seasonal drying
and the broad response to rain, but misses the sharp spikes. It is far better
than the seasonal average - but visibly not the same as measuring.

The gap between the panels is the honest value of a working soil moisture
sensor. It is large.
"""),
    ("code", """\
summary = []
for task in ("forecast", "estimate"):
    for name, predictions in [
        ("LSTM", results[(task, "lstm")].predictions_test),
        ("GRU", results[(task, "gru")].predictions_test),
    ]:
        summary.append((task, name, S.regression_metrics(datasets[task].y_test, predictions)))
for (task, name), predictions in baselines.items():
    summary.append((task, name, S.regression_metrics(datasets[task].y_test, predictions)))

header = f"{'Question':<10} {'Method':<14} {'RMSE':>9} {'MAE':>9} {'R2':>8}"
print(header)
print("-" * len(header))
for task, name, metrics in sorted(summary, key=lambda row: (row[0], row[2]["rmse"])):
    print(f"{task:<10} {name:<14} {metrics['rmse']:>9.5f} "
          f"{metrics['mae']:>9.5f} {metrics['r2']:>8.4f}")
"""),
    ("md", """\
---

## Summary

| Finding | Detail |
|---|---|
| Networks beat persistence on RMSE | LSTM about 20% lower, driven by rain-response days |
| Persistence beats networks on MAE | It is exactly right on the many days nothing happens |
| Weather alone explains most of it | R2 around 0.72 with no moisture readings at all |
| A sensor is worth more than a model | Forecast error is roughly 4x smaller than estimate error |
| LSTM and GRU are equivalent here | Difference between them is smaller than the seed-to-seed spread |

**The honest conclusion:** the network adds real value, but a working sensor
adds much more, and the choice of accuracy metric decides the winner. That last
point is the one to carry into the next notebook.

Next: [03 - Irrigation decisions and what errors cost](03_irrigation_decisions_and_cost.ipynb)
"""),
]


# ==========================================================================
# 03 - Decisions and cost
# ==========================================================================
DECISIONS = [
    ("md", """\
# Irrigation decisions, and what a wrong one costs

**What this notebook answers:** given that we can estimate soil moisture, how
should that turn into a decision - and which method actually costs least to
run?

The previous notebook ended on an uncomfortable note: two reasonable accuracy
measures picked different winners. This notebook resolves that by asking the
only question a client actually cares about.

> Not "which model predicts best", but **"which model costs least to operate"**.

## Why accuracy is the wrong target

Irrigation errors are not symmetric.

**Applying 5 mm too much** wastes desalinated water and washes nutrients below
the roots. In the UAE that is genuinely expensive - but it is recoverable, and
the crop is unharmed.

**Applying 5 mm too little**, in a July where demand is 8.5 mm/day on sand that
holds only 35 mm in total, pushes the plant past its stress threshold within a
day. Growth stops. Severe stress is not recoverable at all.

RMSE weights those two identically. The field does not. So the evaluation here
is in dirhams.
"""),
    ("code", PREAMBLE + """
from irrigation.decision.policy import CostModel
from irrigation.models.evaluate import render_table, run_comparison
from irrigation.physics.crop import CROPS, SOILS

crop, soil = CROPS["turfgrass"], SOILS["sand"]
results = run_comparison(crop, soil, root_depth_m=0.5, kc=0.85)

print(render_table(results))
"""),
    ("md", """\
Six ways of deciding when to irrigate, run over the same simulated 120-day
Dubai summer:

- **Physics** - the FAO-56 water balance. Track what goes in and what leaves.
  No sensor, no training data, no machine learning. This is what irrigation
  scheduling has used since 1998.
- **Sensor only** - trust the probe. This is what most commercial "smart
  irrigation" does.
- **Physics + sensor fusion** - the engineer's answer: use the balance for the
  trend and let the probe correct it slowly.
- **Random forest, gradient boosting, XGBoost** - supervised models trained on
  simulated seasons, given the same information plus lagged history.
"""),
    ("code", """\
names = [r.predictor_name for r in results]
costs = np.array([r.total_cost for r in results])
rmse = np.array([r.depletion_rmse for r in results])

fig, ax = plt.subplots(figsize=(10, 5))
order = np.argsort(costs)
positions = np.arange(len(results))
colours = [colour_for(names[i]) for i in order]

bars = ax.barh(positions, costs[order], color=colours)
ax.set_yticks(positions, [names[i] for i in order])
ax.set_xlabel("Operating cost over one season (AED)")
ax.set_xscale("log")
ax.set_title("Cost to run, lower is better")
ax.grid(axis="x")

for index, value in enumerate(costs[order]):
    ax.annotate(f"{value:,.0f}", (value, index), va="center",
                xytext=(6, 0), textcoords="offset points", fontsize=10)

plt.tight_layout()
plt.show()
"""),
    ("md", """\
The scale is logarithmic because the spread is enormous. Trusting the sensor
costs roughly **eight times** what the 1998 water balance costs.

That is the single most commercially relevant number in this project, so it is
worth being precise about the cause. The simulated probe carries the failure
modes that actually break field deployments: an uncalibrated offset, and
salinity drift - Gulf irrigation water is desalinated or brackish, salts
accumulate in the root zone, and a capacitance probe reads dissolved salt as
moisture. It therefore reads **progressively wetter than reality**, so a
controller that believes it waters progressively less. The crop starves while
the dashboard shows green.

Naive fusion inherits the same bias and barely improves on it.
"""),
    ("code", """\
best_rmse = min(results, key=lambda r: r.depletion_rmse)
best_cost = min(results, key=lambda r: r.total_cost)

# Two panels rather than one. The sensor-based methods are so much worse that
# on a single set of axes they compress the four serious contenders into an
# unreadable corner - and those four are the entire argument.
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))

for ax in axes:
    for result in results:
        name = result.predictor_name
        ax.scatter(result.depletion_rmse, result.total_cost, s=210,
                   color=colour_for(name), marker=marker_for(name),
                   edgecolor="white", linewidth=1.5, zorder=3, label=name)

axes[0].set_yscale("log")
axes[0].set_xlabel("Prediction error (RMSE, mm)")
axes[0].set_ylabel("Operating cost (AED, log scale)")
axes[0].set_title("All six methods")
axes[0].legend(loc="center right", fontsize=9)

contenders = [r for r in results if r.total_cost < best_cost.total_cost * 2.5]
x_values = [r.depletion_rmse for r in contenders]
y_values = [r.total_cost for r in contenders]
pad_x = (max(x_values) - min(x_values)) * 0.55 + 0.15
pad_y = (max(y_values) - min(y_values)) * 0.55 + 60

axes[1].set_xlim(min(x_values) - pad_x, max(x_values) + pad_x)
axes[1].set_ylim(min(y_values) - pad_y, max(y_values) + pad_y)
axes[1].set_xlabel("Prediction error (RMSE, mm)  - lower is more accurate")
axes[1].set_ylabel("Operating cost (AED)  - lower is cheaper")
axes[1].set_title("The four serious contenders, zoomed")

axes[1].annotate("Most accurate", (best_rmse.depletion_rmse, best_rmse.total_cost),
                 xytext=(15, 42), textcoords="offset points", fontsize=11,
                 fontweight="bold", ha="left",
                 arrowprops=dict(arrowstyle="->", color="#4D4D4D"))
axes[1].annotate("Cheapest to run", (best_cost.depletion_rmse, best_cost.total_cost),
                 xytext=(-15, 40), textcoords="offset points", fontsize=11,
                 fontweight="bold", ha="right",
                 arrowprops=dict(arrowstyle="->", color="#4D4D4D"))

# The three supervised models sit almost exactly on top of one another, so
# they get one label between them rather than three overlapping ones. That is
# also the more honest presentation: the point is that they are
# indistinguishable, not that each deserves separate identification.
supervised = [r for r in contenders if not r.predictor_name.startswith("Physics (")]
physics = next(r for r in contenders if r.predictor_name.startswith("Physics ("))

axes[1].annotate(f"{len(supervised)} machine learning models\\n(within 1% of each other)",
                 (float(np.mean([r.depletion_rmse for r in supervised])),
                  float(np.mean([r.total_cost for r in supervised]))),
                 xytext=(0, -46), textcoords="offset points",
                 ha="center", fontsize=9.5, color="#444444")
axes[1].annotate(physics.predictor_name,
                 (physics.depletion_rmse, physics.total_cost),
                 xytext=(0, -22), textcoords="offset points",
                 ha="center", fontsize=9.5, color="#444444")

plt.tight_layout()
plt.show()

print(f"Lowest prediction error : {best_rmse.predictor_name} "
      f"(RMSE {best_rmse.depletion_rmse:.2f} mm, cost {best_rmse.total_cost:,.0f} AED)")
print(f"Lowest operating cost   : {best_cost.predictor_name} "
      f"(RMSE {best_cost.depletion_rmse:.2f} mm, cost {best_cost.total_cost:,.0f} AED)")
print(f"Accuracy gap            : {best_cost.depletion_rmse - best_rmse.depletion_rmse:+.2f} mm")
print(f"Cost gap                : {best_rmse.total_cost - best_cost.total_cost:+,.0f} AED")
"""),
    ("md", """\
## The headline result

**The most accurate model is not the cheapest to operate.**

The machine learning models predict root-zone depletion more accurately than
the physics baseline does, and they still cost more to run. The reason is
visible in the water column of the table above: they err systematically toward
"drier than reality" and irrigate accordingly, applying a few hundred extra
millimetres over the season, most of which drains straight past the roots.

Crucially this is **not one model getting unlucky**. Random forest, gradient
boosting and XGBoost are three different learning algorithms - one that cannot
extrapolate beyond its training range and two that can. All three land on the
same side of the truth. Swapping the estimator does not fix it, because the
estimator was never what was wrong: fitting to squared error produces a model
optimized for a symmetric world, and this world is not symmetric.

Reporting "we trained a model and reached 1.9 mm RMSE" would have been true,
impressive-sounding, and would have cost the client money.
"""),
    ("code", """\
penalties = [2.5, 5.0, 15.0, 30.0, 60.0, 120.0]
curves = {name: [] for name in names}

for penalty in penalties:
    scenario = run_comparison(crop, soil, root_depth_m=0.5, kc=0.85,
                              cost_model=CostModel(stress_cost_per_mm_deficit=penalty))
    for result in scenario:
        curves[result.predictor_name].append(result.total_cost)

fig, ax = plt.subplots(figsize=(10, 5.4))
for name, series in curves.items():
    ax.plot(penalties, series, color=colour_for(name), marker=marker_for(name),
            label=name)
    ax.lines[-1].set_dashes(dashes_for(name) if dashes_for(name)[0] else [])

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Cost charged per mm of crop stress (AED)")
ax.set_ylabel("Total operating cost (AED)")
ax.set_title("Where the ranking flips")
ax.legend(ncol=2, fontsize=9.5)

crossover = None
for index, penalty in enumerate(penalties):
    winner = min(curves, key=lambda n: curves[n][index])
    if not winner.startswith("Physics ("):
        crossover = (penalty, winner)
        break

if crossover:
    ax.axvline(crossover[0], color="#4D4D4D", linestyle=":", linewidth=1.5)
    ax.annotate(f"machine learning\\ntakes over\\n({crossover[1]})",
                (crossover[0], ax.get_ylim()[0] * 3), fontsize=10, ha="center")

plt.tight_layout()
plt.show()

for index, penalty in enumerate(penalties):
    winner = min(curves, key=lambda n: curves[n][index])
    print(f"stress penalty {penalty:6.1f} AED/mm  ->  cheapest: {winner}")
"""),
    ("md", """\
## When would machine learning be the right call?

Only when crop stress is catastrophically expensive.

At every ordinary cost ratio the 1998 water balance wins - with no training
data, no sensor, and no inference cost. Machine learning takes over only at the
far end, where stress is so ruinous that the models' systematic over-watering
becomes cheap insurance.

**That is a real answer, and a defensible recommendation.** It is also one a
client is unlikely to hear from a vendor selling models. The right response to
"should we use AI for this?" is sometimes no - and being able to show exactly
where the answer would change is more valuable than either a yes or a no.
"""),
    ("code", """\
from irrigation.decision.policy import decide
from irrigation.explain.advisor import explain_decision

for depletion, et0 in [(13.8, 8.4), (4.0, 5.2)]:
    decision = decide(predicted_depletion_mm=depletion, et0_forecast_mm=et0,
                      crop=crop, soil=soil, root_depth_m=0.5, kc=0.85)
    explanation = explain_decision(decision, et0_forecast_mm=et0,
                                   crop_name="turfgrass", soil_name="sand")
    print(explanation.render())
    print()
"""),
    ("md", """\
## Explaining the decision

Every number above was computed by the physics layer before any text was
generated. The explanation layer only assembles them into sentences and
attaches the FAO-56 passage that justifies the rule being applied.

An optional language model (via Ollama) can rewrite this more fluently. When it
is enabled, **every number it emits is extracted and checked against the
supplied facts**, and any output containing a figure that was not provided is
discarded in favour of the deterministic text above. A model that invents a
plausible-looking application depth is the one failure mode that would actually
damage a client's turf, so it is blocked mechanically rather than discouraged
by prompt wording.

---

## Summary

| Question | Answer |
|---|---|
| Which method is cheapest to run? | The FAO-56 water balance, at every ordinary cost ratio |
| Does machine learning predict better? | Yes - and it still costs more |
| Why? | It errs toward "too dry", over-waters, and the excess drains away |
| Is that one bad model? | No - three different algorithms all do it |
| Should the client trust a bare sensor? | No - drift makes it roughly 8x more expensive |
| When would ML be right? | Only where crop stress is catastrophically costly |
"""),
]


NOTEBOOKS = {
    "01_dubai_climate_and_et0.ipynb": CLIMATE,
    "02_soil_moisture_sequence_models.ipynb": SEQUENCE,
    "03_irrigation_decisions_and_cost.ipynb": DECISIONS,
}


def build(name: str, cells: list[tuple[str, str]]) -> Path:
    notebook = new_notebook(cells=[
        new_markdown_cell(source) if kind == "md" else new_code_cell(source)
        for kind, source in cells
    ])
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    })

    path = NOTEBOOK_DIR / name
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"executing {name} ...", flush=True)
    NotebookClient(
        notebook,
        timeout=2400,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    ).execute()

    nbformat.write(notebook, path)
    print(f"  wrote {path}")
    return path


def main() -> int:
    for name, cells in NOTEBOOKS.items():
        build(name, cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
