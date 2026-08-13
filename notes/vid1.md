# Video 1 — Introduction, Goal and The Quantified Self

> **One-line pitch:** Classify which barbell exercise someone is doing and count their reps, using only wrist-worn accelerometer and gyroscope data.

---

## 1. The Quantified Self

**Definition (worth memorising):**

> The quantified self is any individual engaged in the self-tracking of any kind of biological, physical, behavioural or environmental information.

Two properties make it "quantified self" rather than just data collection:

1. The tracking is **driven by a goal** the individual has.
2. There is a **desire to act** on the information collected.

**Interview framing:** this project sits in the *human activity recognition* (HAR) field — a well-established area of ML on time-series sensor data.

---

## 2. The Problem

### What exists today

Smartwatches auto-detect **cardio** well. Start running with an Apple Watch on and after 2-3 minutes you get: *"It looks like you're running — want to track this workout?"*

### What doesn't exist

That same auto-detection for **strength training**. Today you must open an app and manually log: exercise, weight, sets, reps.

### The goal

Invert the interaction. Instead of *telling* the app what you're about to do, the watch **observes** the workout and hands you the summary afterwards.

```
Current:   open app -> log "3x8 bench @ 80kg" -> lift
This:      lift -> watch figures out what you did
```

**Why this framing matters in an interview:** it shows you understand the *product* problem, not just the model. The ML exists to remove friction from logging.

---

## 3. The Two ML Tasks

| Task | Type | Output |
|---|---|---|
| Which exercise? | **Multi-class classification** | bench / squat / deadlift / OHP / row / rest |
| How many reps? | **Custom algorithm** (not ML) | integer count |

The five exercises are the **"big five" compound barbell lifts**:
bench press, deadlift, overhead press, barbell row, squat.

> **Note:** rep counting is *not* a model — it is a signal-processing algorithm built by hand in video 7. Good to flag in an interview: not every problem needs ML.

---

## 4. Why This Is Hard — Sensor Noise

**My own example:**

> Mid-squat, I pause for a second and move my hand to adjust the bar. The sensor records that movement, but it isn't part of the actual exercise.

This is the core difficulty. The sensor captures **everything**, including:

- Re-gripping or adjusting the bar
- Walking to the rack
- Resting between sets
- Scratching your nose

**This is exactly why `rest` is one of the classes** — the model must learn to recognise "not exercising" rather than forcing every window into a lift.

It is also why the pipeline needs outlier detection (video 4) and low-pass filtering (video 5): to separate *movement that is the exercise* from *movement that is noise*.

---

## 5. The Sensors

| Sensor | Measures | Unit |
|---|---|---|
| **Accelerometer** | linear acceleration | g (g-force) |
| **Gyroscope** | rotation / angular velocity | degrees/second |

Both come from a **MetaMotion** wrist sensor. Together they capture translation *and* orientation — you need both, since a squat and a deadlift may accelerate similarly but rotate the wrist differently.

---

## 6. Series Roadmap

| # | Video | Key techniques |
|---|---|---|
| 1 | Intro, goal, quantified self, dataset | — |
| 2 | Converting raw data, reading CSVs, cleaning | pandas, resampling |
| 3 | Visualising data | time-series plotting |
| 4 | Outlier detection | IQR, **Chauvenet's criterion**, **Local Outlier Factor** |
| 5 | Feature engineering | low-pass filter, **PCA**, frequency/Fourier features, clustering |
| 6 | Predictive modelling | Naive Bayes, **SVM**, **Random Forest**, **Neural Network** |
| 7 | Counting repetitions | custom peak-detection algorithm |

---

## Interview Questions

<details>
<summary><b>Q: What problem does this project solve?</b></summary>

Strength training has no automatic tracking, unlike cardio. Users must manually log every set. This project classifies the exercise and counts reps from wrist motion data, so the workout logs itself.
</details>

<details>
<summary><b>Q: Why is this a classification problem and not regression?</b></summary>

The target is a discrete, unordered set of exercise categories (bench, squat, and so on). There is no meaningful ordering or distance between "squat" and "row", so regression would be meaningless. Rep counting *is* numeric, but it is solved algorithmically via peak detection, not regression.
</details>

<details>
<summary><b>Q: Why include a "rest" class?</b></summary>

Because the sensor streams continuously. Without a rest class the model is forced to label every window as some exercise, including periods when the person is sitting between sets. Rest makes the model usable on a continuous real-world stream rather than pre-segmented clips.
</details>

<details>
<summary><b>Q: Why do you need both an accelerometer and a gyroscope?</b></summary>

They capture complementary physics. The accelerometer measures linear acceleration; the gyroscope measures rotation. Two exercises can have similar acceleration profiles but different wrist rotation — using both makes them separable.
</details>

<details>
<summary><b>Q: What makes sensor data hard to work with here?</b></summary>

It is noisy and unsegmented. The sensor records non-exercise movement (adjusting grip, walking, resting) mixed into the same stream as the lift. There are no natural boundaries marking where a set begins or ends, and the two sensors sample at different rates.
</details>

<details>
<summary><b>Q: What field does this belong to?</b></summary>

Human Activity Recognition (HAR) — supervised classification on multivariate time-series sensor data. The reference text is *Machine Learning for the Quantified Self* by Hoogendoorn and Funk.
</details>

---

## Key Terms

| Term | Meaning |
|---|---|
| **Quantified self** | Goal-driven self-tracking with intent to act on the data |
| **HAR** | Human Activity Recognition |
| **Accelerometer** | Measures linear acceleration (g) |
| **Gyroscope** | Measures angular velocity (deg/s) |
| **Compound lift** | Multi-joint barbell movement (the big five) |
