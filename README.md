# readmyrep.ai

Machine learning on raw wristband motion data to figure out **what you're lifting and how many reps you did** — no manual logging.

A wrist-worn accelerometer + gyroscope streams motion data while someone trains. This project turns that raw signal into two answers:

1. **Which exercise is this?** — bench press, squat, overhead press, deadlift, barbell row, or rest.
2. **How many reps was that?** — counted directly from the motion signal.

## Status

🚧 In progress — built step by step, one stage per commit.

- [ ] Raw sensor data
- [ ] Dataset construction
- [ ] Visualization
- [ ] Outlier detection & removal
- [ ] Feature engineering
- [ ] Predictive modeling
- [ ] Repetition counting
- [ ] Production packaging (API, tests, CI)

## Project structure

```
├── data
│   ├── external       <- Data from third party sources
│   ├── raw            <- Original MetaMotion sensor exports (immutable)
│   ├── interim        <- Intermediate, transformed data
│   └── processed      <- Final datasets used for modeling
├── docs               <- Documentation
├── models             <- Trained and serialized models
├── notebooks          <- Jupyter notebooks for exploration
├── notes              <- Learning notes taken while building this
├── references         <- Data dictionaries, papers, manuals
├── reports/figures    <- Generated graphics and figures
└── src
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    ├── features       <- Scripts to turn raw data into features
    │   └── build_features.py
    ├── models         <- Training and inference
    │   ├── train_model.py
    │   └── predict_model.py
    └── visualization  <- Plotting
        ├── plot_settings.py
        └── visualize.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Data

Wristband accelerometer (12.500 Hz) and gyroscope (25.000 Hz) recordings collected during barbell training sessions, labeled by exercise and weight category (heavy / medium).
