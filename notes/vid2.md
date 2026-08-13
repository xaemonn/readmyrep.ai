# Video 2 — Converting Raw Data Into a Single Dataset

> **Goal:** Read all the separate raw CSV files, extract their labels, and merge them into a single clean DataFrame.

---

## 1. The Dataset

### How it was collected

Five participants (`n=5`) with **different years of experience** in fitness.

The procedure per set:

1. Participant gets ready to do, say, squats
2. They are wearing the watch
3. We press record on the watch
4. Participant performs the set
5. We stop recording and export the data

Each participant did **one heavy set and one medium (lighter) set** per exercise.

> **Why vary experience level and weight?** It forces the model to generalise. A beginner's squat and an experienced lifter's squat look different in the signal, and a heavy set moves slower than a light one. Training across both makes the model robust rather than overfitted to one person's technique.

### What is in the data

| Property | Value |
|---|---|
| Participants | 5 (A, B, C, D, E) |
| Labels | bench, dead, ohp, row, squat, **rest** |
| Categories | heavy, medium, sitting, standing |
| Total CSV files | 187 |
| Accelerometer files | 94 |
| Gyroscope files | 93 |
| Accelerometer rows (merged) | 23,578 |
| Gyroscope rows (merged) | 47,218 |

**Note on the mismatch:** 94 accelerometer files but only 93 gyroscope files — one recording has no gyroscope partner. Expect a size difference when merging; it is the data, not a bug.

### The REST class

Data recorded while participants were **resting between sets** — either sitting or standing. This is why `sitting` and `standing` appear as categories alongside `heavy` and `medium`.

---

## 2. The Two Sensors

| Sensor | Measures | Unit | Sample rate |
|---|---|---|---|
| **Accelerometer** | how speed is changing (linear acceleration) | g (g-force) | 12.500 Hz |
| **Gyroscope** | rotation | degrees/second | 25.000 Hz |

The two sensors write **separate files** for the same recording, and they alternate in `data/raw/MetaMotion`.

**The gyroscope samples at twice the rate**, which is why it has roughly twice the rows (47,218 vs 23,578). Reconciling these two rates is what resampling solves later in this video.

---

## 3. Why Data Processing Is Needed

Two problems with the raw export:

1. **The labels live in the filename, not in the data.** Open any CSV and there is no column saying "this is a bench press".
2. **Each recording is split across two files** — one accelerometer, one gyroscope.

### The filename is data

```
A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv
|   |     |     |                                                  |            |
|   |     |     └─ RPE (rate of perceived exertion)                 |            └─ sample rate
|   |     └─ category (heavy / medium)                              └─ sensor type
|   └─ label (the exercise)
└─ participant
```

Parsing these strings into `participant`, `label` and `category` columns is the core work of this video.

---

## 4. Learning Setup

This is **supervised learning** — we have labelled data (from the filenames).

We build a **multi-class classification** model.

The data is a mix of **structured and unstructured** sources: structured metadata from filenames, unstructured raw sensor time series.

---

## 5. Epoch and Unix Time

**Unix time** is a date and time representation widely used in computing. It measures time by the number of **seconds** (or ms / ns) elapsed since:

```
00:00:00 UTC on 1 January 1970    <- "the Unix epoch"
```

### Why it is good

It is **the same on every device**. It standardises time, with no timezone or locale ambiguity.

### How a device actually works

It calculates everything internally in Unix time, then converts to a readable format based on the device's current settings.

### The three time columns in each CSV

| Column | What it is |
|---|---|
| `epoch (ms)` | Unix milliseconds — **use this one** |
| `time (01:00)` | Same instant, human-readable. `01:00` is the UTC offset |
| `elapsed (s)` | Seconds since this recording started |

**We index on `epoch (ms)`** because it is unambiguous and identical across both sensors, which is what makes merging accelerometer and gyroscope data reliable. The other two get deleted.

> The 1-hour difference in `time (01:00)` is the offset between UTC and CET winter time.

---

## 6. Key pandas Operations

```python
# Parse the filename into features
participant = f.split("-")[0].replace(data_path, "")
label       = f.split("-")[1]
category    = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

# Convert Unix ms into a real datetime index
df.index = pd.to_datetime(df["epoch (ms)"], unit="ms")

# Drop the redundant time columns
del df["epoch (ms)"], df["time (01:00)"], df["elapsed (s)"]
```

**Why set a datetime index?** `df.info()` shows pandas does not know these are datetime variables by default — they are just numbers or strings. Without conversion, time-based built-in functions do not work. Once converted you get `.dt.month`, resampling, time slicing, and so on.

### Gotcha: `rstrip` is not a suffix strip

`.rstrip("_MetaWear_2019")` does **not** remove that string. `rstrip` removes any trailing character that appears in the *set* `{_, M, e, t, a, W, r, 2, 0, 1, 9}`. It works on this dataset by luck — the categories end in `y`, `m`, `g`. A category ending in `e` or `t` would be silently chewed.

### Gotcha: Windows path separators

`glob()` joins paths with the OS separator — backslash on Windows, forward slash on macOS. So `data_path` must include the trailing separator:

```python
data_path = "../../data/raw/MetaMotion\\"   # Windows
```

Without it, `participant` comes out as `\A` instead of `A`, silently poisoning every downstream groupby.

---

## 7. Resampling (Frequency Conversion)

The two sensors run at different rates:

```
Accelerometer:  12.500 Hz
Gyroscope:      25.000 Hz
```

To merge them into one DataFrame, both must share a common time grid. Resampling converts them to a single fixed interval, aggregating multiple readings into one row per interval.

---

## Interview Questions

<details>
<summary><b>Q: Where did the labels come from?</b></summary>

They were encoded in the filenames by the researchers during collection, in the form `participant-label-category`. The first processing step parses those strings into columns. There was no manual annotation of the signal itself.
</details>

<details>
<summary><b>Q: Why index on epoch time instead of the human-readable timestamp?</b></summary>

Epoch time is an unambiguous integer count of milliseconds since 1 Jan 1970 UTC, identical on every device and free of timezone or locale issues. That makes it a reliable join key for aligning the accelerometer and gyroscope recordings, which were written as separate files.
</details>

<details>
<summary><b>Q: The two sensors sample at different frequencies. How do you handle that?</b></summary>

Resampling to a common time grid. The accelerometer runs at 12.5 Hz and the gyroscope at 25 Hz, so the gyroscope has roughly twice the rows. Both are resampled onto a shared fixed interval and aggregated, after which they can be merged row-for-row.
</details>

<details>
<summary><b>Q: Why did you convert the index to datetime rather than leaving it as an integer?</b></summary>

pandas treats raw epoch integers as ordinary numbers, so no time-aware functionality works. Converting with `pd.to_datetime(..., unit="ms")` unlocks resampling, time-based slicing and datetime accessors, all of which the rest of the pipeline depends on.
</details>

<details>
<summary><b>Q: Why collect data from participants with different experience levels and different weights?</b></summary>

To make the model generalise. Technique varies with experience, and bar speed varies with load, so both change the signal shape. Training across that variation prevents the model overfitting to one person's form or one tempo.
</details>

<details>
<summary><b>Q: What kind of learning problem is this, and what type of data?</b></summary>

Supervised multi-class classification on multivariate time-series data. It combines structured metadata (parsed from filenames) with unstructured raw sensor streams.
</details>

<details>
<summary><b>Q: You have 94 accelerometer files but 93 gyroscope files. What does that mean for the merge?</b></summary>

One accelerometer recording has no gyroscope counterpart. On an inner join that recording is dropped; on an outer join it produces NaNs in the gyroscope columns. Either is defensible, but it must be a deliberate choice rather than an unnoticed row-count mismatch.
</details>

---

## Key Terms

| Term | Meaning |
|---|---|
| **Epoch / Unix time** | Milliseconds since 00:00:00 UTC, 1 Jan 1970 |
| **Resampling** | Converting a time series to a different fixed frequency |
| **Hz** | Samples per second |
| **RPE** | Rate of Perceived Exertion — how hard the set felt |
| **Supervised learning** | Training on labelled examples |
| **Multi-class classification** | Predicting one of 3+ mutually exclusive classes |
