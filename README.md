# SKYLANCE-X

**ATC Emergency Suggestion Engine** — a decision-support system for real-time air traffic control interventions. Given a live sector state (aircraft positions, fuel levels, runway availability), it simulates 10 minutes into the future, detects safety breaches, and recommends a single best control action: assign a runway, hold, or vector.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io) + [Plotly](https://plotly.com/python/) |
| Cascade simulator | Pure Python (no external deps; dead-reckoning, ICAO 5 nm / 1000 ft separation) |
| Physics validation | [BlueSky ATC Simulator](https://github.com/TUDelft-CNS-ATM/bluesky) (TU Delft) |
| Baseline comparison | scikit-learn RandomForest + greedy rule-based policy |
| Safety certification | Split conformal prediction (Papadopoulos et al. 2002; distribution-free guarantee) |
| Schema | Python `dataclasses` with full JSON serialisation |

---

## Project Structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard — radar plot, fuel chart, breach timeline, sidebar controls. All mutable state in `st.session_state`. |
| `state_schema.py` | `AircraftState` and `SectorState` dataclasses; `make_mock_sector()` factory. Single source of truth for data structures. |
| `cascade_engine.py` | Pure-Python forward simulator. Dead-reckons aircraft over a 10-min horizon at 60 s checkpoints. Checks ICAO separation and fuel reserves. No BlueSky dependency. |
| `recommender.py` | Enumerates up to 20 candidate actions (runway assignments → swaps → holds → vectors), scores each via `CascadeEngine`, picks highest safety + fairness + efficiency reward. |
| `conformal.py` | `ConformalCertifier` — split conformal prediction with finite-sample coverage guarantee P(violated ∧ certified safe) ≤ α + 1/(n_cal + 1). Default α = 0.05. |
| `explainer.py` | Minimal-perturbation counterfactual explainer. Finds the cheapest single-feature change that flips the recommendation and renders it as a plain-English sentence. |
| `bluesky_adapter.py` | Wraps BlueSky behind the SKYLANCE-X interface. Snapshot/restore for non-destructive lookahead. Shadows fuel and burn rate because BlueSky's OpenAP model does not decrement mass. |
| `baseline.py` | `GreedyBaseline` (pure rules) and `RFBaseline` (RandomForest trained on greedy labels). Used for side-by-side comparison in the dashboard. |
| `bluesky/` | Full TU Delft BlueSky simulator (bundled). |

### Data Flow

```
app.py (UI)
  ├─ state_schema.py      ← SectorState, AircraftState
  ├─ cascade_engine.py    ← evaluate actions → CascadeScore
  ├─ recommender.py       ← enumerate + rank → Recommendation
  ├─ conformal.py         ← certify → Certificate
  ├─ explainer.py         ← minimal flip → Counterfactual
  ├─ baseline.py          ← greedy / RF comparison
  └─ bluesky_adapter.py   ← physics-accurate validation (BlueSky)
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** BlueSky initialises module-level singletons on first import (~30 s one-time setup). The dashboard shows a spinner during this step.

---

## Quickstart

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Using the dashboard

| Control | What it does |
|---|---|
| **Seed / Aircraft slider** → Load | Generate a new random sector (reproducible by seed) |
| **Advance 60 s** | Step the live BlueSky simulation forward by 60 seconds |
| **Runway checkboxes** | Toggle runway availability in real time |
| **Declare emergency** | Inject a MAYDAY / PAN-PAN / FUEL / MED emergency onto any aircraft |
| **SKYLANCE-X / Baseline toggle** | Switch between the conformal recommender and the RF/greedy baseline |
| **Explain decision** | Run the counterfactual explainer (~30–60 s) |

### Radar legend

- **Red dot** — active emergency (MAYDAY / PAN-PAN / FUEL / MED)
- **Amber dot** — fuel critical (< 30 min above regulatory reserve)
- **Green dot** — normal (colour intensity scales with fuel margin)
- **Blue square** — airport (EGLL London, EHAM Amsterdam, EDDF Frankfurt)
- **Dashed line** — 5-min projected path
- **Amber line** — recommended reroute (AssignRunway action)

---

## Sector model

Mock sectors are drawn from Amsterdam FIR airspace (51–53.5 °N, 3–7.5 °E). Each sector has 4–12 aircraft; by construction, two aircraft have active emergencies and one is fuel-critical. Runway availability is randomised with a 15 % closure probability, with at least one runway always open.

---

## Safety certificate

The conformal certifier is calibrated on 40 geometric scenarios during initialisation. For a new recommendation it computes:

```
nc = separation_breach_count + urgency_penalty × (1 − time_to_first_breach / horizon)
```

and issues a **Certified safe** verdict when `nc ≤ q̂` (95 % confidence). The p-value and threshold are displayed in the ribbon below the recommendation card.
