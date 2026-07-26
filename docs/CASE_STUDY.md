# Case Study — Dubai Smart Irrigation Advisor

**Physics-grounded irrigation decisions, and an honest test of whether ML helps**

Hamza Aziz · AI Product Engineer
Python · scikit-learn · FAO-56 agronomy · decision theory

> **Provenance:** a self-directed build responding to a real published project
> brief, not a client engagement. The requirements are the client's; the
> architecture, the physics implementation and the findings are mine. Say it
> this way — the work stands on its own.

---

## The brief

A project posting sought AI/ML research support for a Dubai smart irrigation
prototype: combine soil-moisture sensors with historical, current and forecast
weather to predict irrigation requirements. Build XGBoost, Random Forest and
LSTM/GRU models. Implement evapotranspiration and soil-water balance. Add an
LLM/RAG layer to explain recommendations. Compare rule-based, ML and
LLM-assisted approaches.

Two things stood out on reading it.

**First, a hard dependency nobody had surfaced.** A separate IoT engineer was
building the ESP32 sensor rig. So there was no logged soil-moisture data —
and a sequence model trained on nothing is not a modelling problem, it is an
impossibility. Any honest response has to address that before it addresses
model selection.

**Second, the comparison was framed backwards.** "Compare rule-based, ML-based
and LLM-assisted approaches" presumes the interesting question is which model
predicts soil moisture best. It isn't. Irrigation is a decision under
asymmetric cost, and the best predictor is not necessarily the best decision
rule. I wanted to know whether that gap was real or theoretical.

It was real, and it is the finding this project exists to report.

---

## What I built

A physics-first irrigation advisor where every layer can be checked:

```
climate → physics (FAO-56 ET0 + water balance) → predictors → decision → explanation
```

### 1. Physics validated against the standard's own arithmetic

Reference evapotranspiration is the foundation. Get it wrong and nothing
downstream is worth computing. So it is verified against FAO-56's three
published worked examples — Bangkok 5.72, Brussels 3.88, Lyon 4.56 mm/day —
**including intermediate quantities**, because a wrong `Ra` cancelling against
a wrong `Rnl` would pass a final-answer test.

All three reproduce exactly.

Example 20 (temperature data only) is the one that matters operationally: a
field site rarely has radiation or humidity instrumentation, so the degraded
path has to be right too.

### 2. A bug the seasonal shape concealed

The first Dubai ET0 climatology looked correct — summer peak, winter trough,
smooth annual curve. The total was 2,446 mm against a published UAE range of
roughly 2,000–2,200.

Wind normals are reported at the standard meteorological height of **10 m**.
Penman-Monteith requires **2 m**. Applying FAO-56 Eq. 47 brought the total to
2,226 mm and the summer peak from 9.6 to 8.7 mm/day.

The seasonal *shape* was right the whole time, which is precisely why nothing
except a magnitude check against published climatology could have caught it.
The fix was not just the conversion — it was making the measurement height
part of the data (`WIND_10M_MS`, `WIND_MEASUREMENT_HEIGHT_M`) and converting
at point of use, with a regression test. A bare list of numbers with no stated
units is how this error gets reintroduced in six months.

**The generalisable lesson: validate magnitude against an external reference,
not just shape against intuition.** Plausible-looking output is the most
dangerous kind.

### 2b. The same lesson, learned again

Later, adding 30 years of NASA POWER observations gave a stronger external
reference than the published range: real Dubai ET0 averages **2,275 mm/year**,
against the corrected climatology's 2,224 mm. Two derivations sharing no inputs,
agreeing to 2.3%.

Worth noting the original "2,000–2,200" target was itself slightly wrong as
used — it is a *national* envelope covering cooler inland and mountain areas,
and coastal Dubai legitimately sits above it. The check still worked, because a
9% error is larger than that ambiguity. It would not have caught a 3% one.

Then the lesson repeated in a third form. The dashboard reported Dubai's annual
rainfall as **3,272 mm** — the 30-year total presented as an annual figure —
directly above a caption asserting that rainfall never offsets demand. The same
slip put monthly totals on an axis labelled mm/day, about 30× too tall. The true
figure is 109 mm.

Nothing in the physics was wrong, so no existing test failed. The fix was a test
anchoring the quantity to published climatology. **A number that no test
compares against the outside world will eventually drift, and units are where it
drifts first.**

### 3. Synthetic data as the correct answer, not a compromise

Since no sensor data existed, I built the simulator instead: validated ET0
drives a real soil water balance, which drives a simulated probe carrying the
failure modes that break field deployments — calibration offset, salinity
drift, noise, dropout, 12-bit quantisation.

Salinity drift is not incidental in the Gulf. Irrigation water is desalinated
or brackish, salts accumulate in the root zone, and capacitance probes read
that as moisture.

### 4. Decisions evaluated on cost, not error

| Predictor | Cost (AED) | Water (mm) | Stress days | Drainage (mm) | RMSE (mm) |
|---|---:|---:|---:|---:|---:|
| **Physics (FAO-56)** | **2,645** | 926 | 5 | 9 | 2.49 |
| Sensor only | 20,062 | 865 | 99 | 1 | 12.28 |
| Physics + fusion | 16,533 | 900 | 97 | 4 | 9.96 |
| Random forest | 3,055 | 1,150 | 0 | 209 | 1.89 |
| Gradient boosting | 3,064 | 1,154 | 0 | 212 | 1.93 |
| XGBoost | 3,036 | 1,143 | 0 | 202 | **1.88** |

Four results worth stating plainly.

**Trusting the probe costs 7.6× the physics baseline.** Drift makes it read
progressively wetter than reality, so the controller starves the crop — 99
stress days out of 120. Naive fusion inherits the bias and barely improves on
it. This is what a large amount of commercial "smart irrigation" actually does.

**The best RMSE does not win on cost.** XGBoost predicts depletion more
accurately than the physics baseline and still costs 391 AED more to run. It
errs toward "drier than reality" and irrigates accordingly: 217 mm of extra
water, 194 mm of it draining past the root zone.

**It is not one model getting unlucky.** Random forest cannot extrapolate
beyond its training range; the two boosters can. All three still land on the
same side of the truth, over-water by the same margin, and lose to the water
balance by the same amount — they finish within 1% of each other on cost.
Changing the estimator does not help, because the estimator was never the
problem: a squared-error objective produces a model tuned for a symmetric
world, and irrigation is not symmetric. That makes the finding a statement
about the loss function rather than about a library.

**The crossover is at roughly a 48:1 stress-to-water cost ratio.** Below that,
physics wins — with no training data, no sensor, no inference. Above it, ML's
systematic over-watering becomes cheap insurance and it wins. Both halves of
that sentence are needed for the recommendation to be honest.

---

## Engineering decisions worth defending

**Estimation and decision are separate layers.** A predictor answers "how
depleted is the root zone", the policy answers "what do I do about it". Fusing
them would make it impossible to attribute a cost difference to the model
rather than the trigger threshold.

**The simulator cannot leak the answer.** Predictors receive an `Observation`
carrying only field-visible information. There is a test asserting true
depletion is not reachable from it — a simulator that leaks proves nothing.

**Training and evaluation use different weather seeds**, and training data is
generated under the physics policy. A model that only ever sees states a
broken controller visits will not generalise to a working one.

**Cost parameters are exposed, not buried.** The 12:1 stress-to-water ratio is
a policy judgement, not a measurement. It is a constructor argument, and the
sensitivity sweep shows exactly how much the conclusion depends on it — which
is the difference between a recommendation and an assertion.

**The explanation layer never computes.** Same discipline as the reporting
work: numbers are calculated in tested Python, retrieval supplies the
agronomic justification with a citation, and the language layer only
assembles. Offline by default, so an outage degrades wording and not reasoning.

---

## What I deliberately did not build

**No live weather API.** `DailyWeather` is the interface a client implements.
The 30-year NASA POWER record is downloaded once by hand and committed, rather
than fetched at runtime: a pipeline that silently downloads cannot be
reproduced, and it fails in a way that looks like a modelling bug.

**No sequence model in the decision loop**, despite LSTM/GRU being named in
the brief — and this is the one worth explaining, because the answer changed.

The original position was that sequence models could not be built at all: with
no logged sensor history they would be fitted entirely to simulator dynamics,
learning my water balance rather than Dubai's soil, and reporting an impressive
validation score that meant nothing.

That objection was about the *target*, not the architecture. NASA POWER
publishes modelled root-zone soil wetness for the Dubai cell, produced by
MERRA-2's land surface from satellite and reanalysis inputs — real, external,
and completely independent of anything here. So the models were built and
trained against it, and they work: the LSTM beats persistence on RMSE and
recovers about 72% of soil moisture variance from weather alone.

They are still not wired into irrigation scheduling, for a reason that is
specific and checkable: **that grid cell is bare desert, not farmland.** Its
dynamics are rain-driven. Using a model of how untouched sand behaves to
schedule water on an irrigated plot would be exactly the category error this
project was built to expose — an impressive number applied to the wrong
question. The honest deliverable is the soil-atmosphere response, plus the
finding that the sensor is worth more than the model.

**No spatial model, no root growth, single soil column.** Named rather than
quietly omitted.

---

## Talking points

**Why start with physics rather than the models the brief asked for?**
Because FAO-56 is a validated, free, hardware-free baseline that has scheduled
irrigation since 1998. If a model cannot beat it, the model should not ship.
Starting anywhere else means never finding out.

**Your best model lost. Isn't that a negative result?**
It is the result. The client's actual question is "what should we deploy",
and the answer is a water balance plus a bias-corrected sensor — cheaper,
simpler, and better at the cost ratio that applies. I also showed exactly
where that flips, so the recommendation survives a change in their economics.

**How do you know the physics is right?**
It reproduces FAO-56's three published worked examples exactly, including
intermediates. And the Dubai climatology it generates lands inside the
published UAE annual range — which is how I found the 10 m wind bug.

**What is weakest here?**
The simulator is my model of Dubai soil, so every result is conditional on it
being approximately right. I have bounded that where I can — validated physics,
published climatology, realistic sensor pathologies — but no amount of internal
consistency substitutes for one season of real logged data. That is the first
thing I would ask the client for, and the first thing I would re-run against.
