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

**The generalizable lesson: validate magnitude against an external reference,
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
drift, noise, dropout, 12-bit quantization.

Salinity drift is not incidental in the Gulf. Irrigation water is desalinated
or brackish, salts accumulate in the root zone, and capacitance probes read
that as moisture.

### 4. Decisions evaluated on cost, not error

| Predictor            | Cost (AED) | Water (mm) | Stress days | Drainage (mm) | RMSE (mm) |
|----------------------|------------|------------|-------------|---------------|-----------|
| **Physics (FAO-56)** | **2,550**  | 918        | 5           | 0             | **0.00**  |
| Sensor only          | 17,023     | 872        | 94          | 1             | 10.83     |
| Physics + fusion     | 13,400     | 887        | 89          | 0             | 9.34      |
| Random forest        | 5,977      | 910        | 36          | 1             | 5.74      |
| Gradient boosting    | 2,677      | 934        | 9           | 14            | 0.91      |
| XGBoost              | 2,824      | 920        | 11          | 2             | 1.42      |

RMSE compares each morning's estimate with the true morning state. Four results worth stating plainly.

**The physics baseline is the simulator's own model, so it is exact here.** It runs the same water balance on the same weather that generates the truth, and its error is zero by construction. That is internal consistency, not field accuracy. The comparison measures how close everything else gets to a perfect process model, and what the gap costs.

**Trusting the probe costs 6.7× the physics baseline.** Drift makes it read progressively wetter than reality, so the controller starves the crop: 94 stress days out of 120. Fusion at a low gain helps but inherits the bias, because in here it can only pull an exact estimate toward a biased one. This is what a large amount of commercial "smart irrigation" actually does.

**The best learned model gets within 5%, and all three err the same way.** Gradient boosting tracks depletion to 0.91 mm and costs 127 AED more per season; XGBoost (1.42 mm) costs 274 AED more; random forest costs more than twice as much. All three estimate the root zone as wetter than it is, so all three let it dry past RAW on more days than the water balance. RMSE reports how far off a model is, not which way, and the asymmetric cost charges for the way.

**There is no crossover in the range tested.** Physics wins at every stress-to-water ratio from 1:1 to 48:1, and the learned models fall further behind as stress gets dearer, because their error points toward under-watering. The recommendation follows: with good weather data and a calibrated crop and soil model, the water balance needs no training data, no sensor and no inference. Where ML could earn its place is where that model is wrong, which this simulator cannot represent and a field trial should test first.

**A correction.** An earlier version of this section reported XGBoost as the most accurate model yet 391 AED dearer, over-watering by about 200 mm, and blamed the squared-error objective. Those figures came from three bugs: RMSE scored against the end-of-day state instead of the morning state, a physics predictor stepping a day ahead of the state it estimated, and a depletion fraction adjusted by ET0 instead of ETc. With them fixed and the loss function unchanged, the over-watering disappeared, so the earlier attribution to the loss function did not hold.

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
broken controller visits will not generalize to a working one.

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
