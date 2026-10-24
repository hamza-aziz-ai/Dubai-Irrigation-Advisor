# Dubai Smart Irrigation Advisor

Physics-grounded irrigation decisions for Dubai — FAO-56 evapotranspiration
validated against 30 years of NASA observations, a real soil water balance,
PyTorch sequence models trained on independent satellite-derived soil
moisture, and an honest test of whether any of the machine learning improves
on a 1998 equation.

```bash
pipenv install --dev
pipenv run pytest -q
pipenv run python scripts/demo.py               # narrated end-to-end
pipenv run streamlit run app/dashboard.py       # interactive dashboard
```

Everything runs offline against committed data. No API keys, no network, no
hardware.

**Notebooks** — [climate and ET0](notebooks/01_dubai_climate_and_et0.ipynb) ·
[sequence models](notebooks/02_soil_moisture_sequence_models.ipynb) ·
[decisions and cost](notebooks/03_irrigation_decisions_and_cost.ipynb)

---

## The finding

Six ways to decide when to irrigate, run over the same simulated 120-day Dubai summer, scored on operating cost rather than prediction error:

| Predictor                    | Cost (AED) | Water (mm) | Stress days | Drainage (mm) | RMSE (mm) | Bias (mm) |
|------------------------------|------------|------------|-------------|---------------|-----------|-----------|
| **Physics (FAO-56 balance)** | **2,550**  | 918        | 5           | 0             | **0.00**  | 0.00      |
| Sensor only                  | 17,023     | 872        | 94          | 1             | 10.83     | −10.18    |
| Physics + sensor fusion      | 13,400     | 887        | 89          | 0             | 9.34      | −8.91     |
| Random forest                | 5,977      | 910        | 36          | 1             | 5.74      | −5.31     |
| Gradient boosting            | 2,677      | 934        | 9           | 14            | 0.91      | −0.43     |
| XGBoost                      | 2,824      | 920        | 11          | 2             | 1.42      | −1.28     |

RMSE and bias compare each morning's estimate with the true morning state.

**Inside this simulator the physics baseline is a perfect model, and nothing beats it.** The water balance that estimates depletion is the same balance that generates the true state, fed the same weather, so its error is zero by construction. That is internal consistency, not field accuracy: in a real field it would drift with every loss it does not model. What the comparison measures is how close each other method gets to a perfect process model, and what the gap costs.

**The best learned model gets within 5%.** Gradient boosting tracks depletion to 0.91 mm and costs 127 AED more per season. XGBoost, at 1.42 mm, costs 274 AED more; random forest, which cannot extrapolate beyond its training range, costs more than twice as much.

**All three learned models err on the wet side.** Each estimates the root zone as wetter than it is, so each lets it dry past RAW on more days than the water balance does: 9, 11 and 36 stress days against 5. Under asymmetric cost that is the expensive direction. RMSE reports how far off a model is, not which way, and the field charges for the way.

**Trusting the probe is the costliest mistake.** The simulated sensor reads wet - a calibration offset plus salinity drift - so a controller that believes it starves the crop for most of the season. Fusing it into the water balance at a low gain helps, but here it can only pull an exact estimate toward a biased one.

### Cost sensitivity

| Stress penalty (AED/mm) | Ratio | Physics   | Sensor | Fusion | RF     | GBM   | XGB   | Winner  |
|-------------------------|-------|-----------|--------|--------|--------|-------|-------|---------|
| 2.5                     | 1×    | **2,447** | 3,524  | 3,277  | 2,713  | 2,494 | 2,475 | Physics |
| 5.0                     | 2×    | **2,457** | 4,751  | 4,197  | 3,010  | 2,510 | 2,507 | Physics |
| 15.0                    | 6×    | **2,494** | 9,660  | 7,878  | 4,197  | 2,577 | 2,634 | Physics |
| 30.0                    | 12×   | **2,550** | 17,023 | 13,400 | 5,977  | 2,677 | 2,824 | Physics |
| 60.0                    | 24×   | **2,662** | 31,750 | 24,444 | 9,537  | 2,878 | 3,205 | Physics |
| 120.0                   | 48×   | **2,887** | 61,203 | 46,530 | 16,658 | 3,279 | 3,965 | Physics |

The water balance wins at every ratio swept, and the learned models fall further behind as stress gets dearer, because their error points toward under-watering. At parity the boosters are within 2% of physics; at 48× the best of them is 14% behind.

That is the answer a client should get: with good weather data and a calibrated crop and soil model, a 1998 water balance needs no training data, no sensor and no inference, and the learned models only approach it. Where they could earn their place is where that model is wrong - an unmeasured loss, a mis-specified soil - which this simulator cannot represent, and which is the first thing a field trial should test.

### A correction

An earlier version of this table reported XGBoost as the most accurate model yet 391 AED dearer than physics, with the learned models over-watering by about 200 mm. Those figures came from three bugs, fixed together: RMSE was scored against the end-of-day state rather than the morning state each estimate describes; the physics predictor paired yesterday's water with today's ET, putting it a day ahead of the state it estimated; and the depletion fraction was adjusted by reference ET0 instead of crop ETc, as FAO-56 specifies. The over-watering disappeared with the loss function unchanged, so it was not the squared-error objective's doing, as this README used to argue. Regression tests now pin the timing contract.

---

## The physics is validated twice

**Against FAO's own worked examples**, including the intermediate quantities —
a wrong `Ra` cancelling against a wrong `Rnl` would pass a final-answer-only
test.

| FAO-56 example | Case                            | Computed | Published |
|----------------|---------------------------------|----------|-----------|
| Example 17     | Bangkok, monthly data           | 5.72     | 5.72      |
| Example 18     | Brussels, daily + RH, 10 m wind | 3.88     | 3.88      |
| Example 20     | Lyon, temperature only          | 4.56     | 4.56      |

Those examples are Bangkok, Brussels and Lyon. None is a desert, so passing
them says nothing about Dubai.

**So the second check is against reality.** Reference ET0 is derived twice from
unrelated sources — once from published monthly climate normals, once from 30
years of NASA satellite and reanalysis observations. Neither was tuned to the
other.

| Route                      | Annual ET0                        |
|----------------------------|-----------------------------------|
| NASA POWER, 1995–2024 mean | **2,275 mm** (2,071–2,392, sd 65) |
| Published climate normals  | **2,224 mm**                      |

They agree to 2.3%, and every monthly mean agrees within 0.36 mm/day. Two
derivations sharing no inputs do not land on the same seasonal curve by
accident.

### A bug the seasonal shape hid

The first Dubai ET0 climatology looked right — summer peak, winter trough,
smooth curve — and was wrong by about 9%.

Cause: wind normals are reported at the standard meteorological height of
**10 m**, and Penman-Monteith requires **2 m**. Applying FAO-56 Eq. 47 brought
the annual total from 2,446 mm to 2,226 mm and the summer peak from 9.6 to
8.7 mm/day.

The *shape* was correct throughout, which is why only a magnitude check against
independent data caught it. Measurement height is now carried explicitly in the
data (`WIND_10M_MS`, `WIND_MEASUREMENT_HEIGHT_M`) and converted at point of
use, with a regression test.

### A second units bug, caught the same way

The dashboard once reported Dubai's annual rainfall as 3,272 mm — the 30-year
total presented as an annual figure — directly above a caption stating that
rainfall never offsets demand. The same slip put monthly rainfall totals on a
chart axis labelled mm/day, roughly 30× too tall.

Neither error touched the physics, so no existing test noticed. The fix was a
test anchoring the quantity to published climatology, which is the only thing
that distinguishes 109 mm from 3,272 mm. Numbers that no test compares against
the outside world are numbers waiting to drift.

---

## Real data, and its limits

Thirty years of daily NASA POWER observations for Dubai — public domain, no
account, no key — downloaded once and committed, so every figure reproduces
offline. Full provenance in [docs/DATA.md](docs/DATA.md).

The record supplies every FAO-56 input **plus** root-zone soil wetness
(`GWETROOT`), which is what makes the sequence models honest: the target was
produced by NASA's land-surface model, not by this project's simulator, so a
network fitted to it cannot be fitting my own assumptions.

Stated plainly, because it matters: **that soil moisture series is bare desert,
not farmland.** The grid cell is ~55 × 65 km and overwhelmingly unirrigated
sand. It describes how real Dubai soil dries and rewets, which is genuinely
useful and genuinely independent — and it is not an irrigated root zone. The
sequence models learn the soil-atmosphere response; they are not drop-in
replacements for the depletion predictors above, and the module says so at
the top.

---

## Sequence models: LSTM and GRU in PyTorch

Two deliberately separated questions, because conflating them is how
soil-moisture models come to look far better than they are.

**Forecast** — predict tomorrow's wetness given the past *including* today's
wetness. Soil moisture is strongly autocorrelated, so the honest bar is
persistence: "assume tomorrow equals today".

**Estimate** — predict wetness from weather alone, no moisture readings at any
lag. Much harder, and much closer to the operational case this project cares
about: no probe, or a probe that has drifted.

| Task     | Method      | RMSE        | MAE         | R²        |
|----------|-------------|-------------|-------------|-----------|
| Forecast | Persistence | 0.00580     | **0.00142** | 0.977     |
| Forecast | **LSTM**    | **0.00467** | 0.00256     | 0.985     |
| Forecast | GRU         | 0.00493     | 0.00267     | 0.984     |
| Estimate | Climatology | 0.03498     | 0.02316     | 0.175     |
| Estimate | **LSTM**    | **0.02022** | **0.01432** | **0.724** |
| Estimate | GRU         | 0.02147     | 0.01525     | 0.689     |

These figures move in the fifth decimal between runs. cuDNN's recurrent
kernels are not deterministic by default, so they reproduce in magnitude but
not digit for digit — worth knowing before reading significance into a gap
that small. Nothing below turns on differences at that scale: the RMSE/MAE
inversion is a factor of 1.8, not a rounding artefact.

### The same lesson, arrived at independently

**The LSTM wins on RMSE. Persistence wins on MAE.** Same models, same data,
same test years, opposite conclusions.

Not a paradox. Persistence is *exactly* right on the many days when nothing
happens in a desert, so its average error is tiny — and badly wrong on the few
days it rains, which RMSE punishes by squaring. The network is slightly wrong
every day and handles the jumps.

Persistence is better on a normal day; the network is better on the days that
matter. Which is "accurate" depends entirely on what each kind of error costs —
the same argument the cost table makes, reached from real data by a different
route.

Two other results worth stating: weather alone explains about 72% of soil
moisture variance with no probe at all, and the forecast error is roughly 4×
smaller than the estimate error — **a working sensor is worth more than any
amount of modelling.**

Leakage is guarded structurally, not by inspection: chronological splits
(train ≤2016, validate ≤2020, test 2021–2024), the scaler fitted on training
years only, and the forecast window ending the day *before* its target. Each
has a test, and one of them uses synthetic data carrying an artificial regime
shift to prove the check can actually fail.

---

## The LLM layer, and how the grounding is enforced

The language model never computes a number. Every LLM project claims something
like that; almost none of them check it. A prompt saying "do not calculate" is
a request, and the failure mode is not refusal — it is a fluent sentence
containing `14.2 mm` where the truth was 13.8.

So the rule is enforced *after* generation. Every number the model emits is
extracted and matched against the facts it was given. Anything unmatched means
the model computed, inferred or invented, and the output is discarded in favour
of the deterministic explainer.

Two details that matter more than they look:

- **Normalization runs before verification.** Models emit `ET₀` and `mm day⁻¹`
  with subscript and superscript digits, which a number regex cannot see. A
  model could state a fabricated `²².⁷` and walk straight past the check.
  Folding to ASCII first closes that hole — there is an adversarial test for
  exactly this.
- **Numbers quoted from cited passages are permitted**, because they are
  grounded in a source rather than in a computation.

The LLM is not load-bearing. Ollama absent, model unpulled, request failed,
output empty, or grounding check failed — all five degrade to the templated
explanation, and the returned `engine` field names what actually produced the
text and why. The headline carrying the actionable number is templated on
every path.

Enabled with `IRRIGATION_EXPLAIN_ENGINE=ollama` or the dashboard checkbox; off
by default so a fresh clone is deterministic. Note that `gpt-oss:120b-cloud`
runs on Ollama's servers, so it needs network — point
`IRRIGATION_OLLAMA_MODEL` at a local tag to keep everything on-machine.

---

## Dashboard

```bash
pipenv run streamlit run app/dashboard.py
```

Four tabs: today's decision with a grounded explanation, the six-method cost
comparison with live cost-asymmetry sliders, the 30-year climate record, and an
explicit statement of what the data cannot tell you.

Accessibility is treated as a requirement rather than a polish step. Colour is
never the only carrier of meaning — the Okabe-Ito palette stays distinguishable
under the common forms of colour vision deficiency, and every series also has a
marker and dash pattern so charts survive greyscale printing and bad
projectors. Every chart has a caption stating its *conclusion*, so a screen
reader user reaches the finding rather than a description of the axes.

---

## Architecture

```
data/           NASA POWER cache      → the only networked code, run by hand
   ↓
climate/        weather in            → DailyWeather (the API contract)
   ↓
physics/        FAO-56 ET0            → validated against FAO + 30y NASA data
   ↓            soil water balance    → depletion, Ks, drainage
   ↓
models/         predictors            → estimate depletion from field-visible data
   ↓            sequence              → LSTM/GRU on real soil moisture
   ↓
decision/       policy                → depletion + cost asymmetry → irrigate / hold
   ↓
explain/        grounded advisor      → cites computed numbers; LLM verified or discarded
   ↓
viz/ app/       charts and dashboard  → accessible by construction
```

**Layer rules:**

| Layer      | May do                          | May not do            |
|------------|---------------------------------|-----------------------|
| `physics`  | Deterministic calculation       | I/O, state, any model |
| `data`     | Read committed files            | Fetch at runtime      |
| `models`   | Estimate depletion              | See true soil state   |
| `decision` | Apply cost asymmetry            | Estimate anything     |
| `explain`  | Assemble prose from given facts | Compute a number      |

A predictor only ever receives an `Observation` — noisy probe reading, weather,
its own history. A test asserts the true state is not reachable from it,
because a simulator that leaks the answer proves nothing.

---

## What is deliberately not here

- **No live weather API.** `DailyWeather` is the interface a client would
  implement. The NASA record is committed rather than fetched, because a
  pipeline that silently downloads at runtime cannot be reproduced and fails in
  a way that looks like a modelling bug.
- **No in-situ soil moisture.** A separate hardware workstream builds the ESP32
  probe. The simulated sensor carries the failure modes that break real
  deployments — calibration offset, salinity drift, dropout, 12-bit
  quantization — but it is a model of a probe, not a probe.
- **The sequence models are not wired into the decision loop.** They are
  trained on desert soil moisture, and using them to schedule irrigation on a
  watered plot would be exactly the overclaim this project exists to expose.
- **The RAG corpus is four passages, not a corpus.** The contract — decision
  facts in, cited passages out, model never computes, output verified — is the
  part that transfers.
- **Costs are a policy choice, not a measurement.** The 12:1 stress-to-water
  ratio is a parameter precisely so an agronomist can argue with it, and the
  sensitivity table shows how much the conclusion depends on it.
- **Single soil column, single crop.** No spatial variability, no canopy model,
  no root growth over the season.

---

## Sources

- Allen, Pereira, Raes, Smith (1998). *Crop evapotranspiration — Guidelines for
  computing crop water requirements.* FAO Irrigation and Drainage Paper 56.
  <https://www.fao.org/4/x0490e/x0490e00.htm> — equations, worked examples,
  crop coefficients (Table 12), depletion fractions (Table 22), soil water
  properties (Table 19).
- NASA Langley Research Center POWER Project, funded through the NASA Earth
  Science Directorate Applied Science Program.
  <https://power.larc.nasa.gov/> — daily climate and soil moisture, 1995–2024.
- Okabe, M. & Ito, K. (2008). *Colour Universal Design.*
  <https://jfly.uni-koeln.de/color/> — the chart palette.
- Constants and their sources: [docs/AGRONOMY.md](docs/AGRONOMY.md).
  Data provenance and limits: [docs/DATA.md](docs/DATA.md).

---

Portfolio project built against a real published brief. Not a client
deployment.
