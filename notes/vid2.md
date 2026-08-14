# Video 2 — Converting Raw Data Into a Single Dataset

> **Goal:** Read all the separate raw CSV files, extract their labels, merge accelerometer and gyroscope into one DataFrame, resample to a common frequency, and export.

**Pipeline in one line:**

```
187 raw CSVs -> parse filenames -> datetime index -> merge acc+gyr -> resample to 200ms -> 01_data_processed.pkl
```

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
| --- | --- |
| Participants | 5 (A, B, C, D, E) |
| Labels | bench, dead, ohp, row, squat, **rest** |
| Categories | heavy, medium, sitting, standing |
| Total CSV files | 187 |
| Accelerometer files | 94 |
| Gyroscope files | 93 |
| Accelerometer rows (merged) | 23,578 |
| Gyroscope rows (merged) | 47,218 |
| **Final resampled rows** | **9,009** |

**Note on the mismatch:** 94 accelerometer files but only 93 gyroscope files — one recording has no gyroscope partner. Expect a size difference when merging; it is the data, not a bug.

### The REST class

Data recorded while participants were **resting between sets** — either sitting or standing. This is why `sitting` and `standing` appear as categories alongside `heavy` and `medium`.

---

## 2. The Two Sensors

| Sensor | Measures | Unit | Sample rate |
| --- | --- | --- | --- |
| **Accelerometer** | how speed is changing (linear acceleration) | g (g-force) | 12.500 Hz |
| **Gyroscope** | rotation | degrees/second | 25.000 Hz |

The two sensors write **separate files** for the same recording, and they alternate in `data/raw/MetaMotion`.

**The gyroscope samples at twice the rate**, which is why it has roughly twice the rows (47,218 vs 23,578).

> **Why not just sample everything at a high frequency?** Measuring at a higher frequency means the device requires **more battery power**. Sample rate is a hardware trade-off, not a free choice — a real constraint on wearables.

---

## 3. Why Data Processing Is Needed

Two problems with the raw export:

1. **The labels live in the filename, not in the data.** Open any CSV and there is no column saying "this is a bench press".
2. **Each recording is split across two files** — one accelerometer, one gyroscope.

### The filename is data

```
A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv
|   |     |     |                                                  |            |
|   |     |     +- RPE (rate of perceived exertion)                 |            +- sample rate
|   |     +- category (heavy / medium)                              +- sensor type
|   +- label (the exercise)
+- participant
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
| --- | --- |
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

A DataFrame whose index is a pandas datetime object is a **time-series DataFrame** — that is what unlocks `.resample()`.

### Gotcha: `rstrip` is not a suffix strip

`.rstrip("_MetaWear_2019")` does **not** remove that string. `rstrip` removes any trailing character that appears in the *set* `{_, M, e, t, a, W, r, 2, 0, 1, 9}`. It works on this dataset by luck — the categories end in `y`, `m`, `g`. A category ending in `e` or `t` would be silently chewed.

### Gotcha: Windows path separators

`glob()` joins paths with the OS separator — backslash on Windows, forward slash on macOS. So `data_path` must include the trailing separator:

```python
data_path = "../../data/raw/MetaMotion\\"   # Windows
```

Without it, `participant` comes out as `\A` instead of `A`, silently poisoning every downstream groupby.

---

## 7. Merging the Two Sensors

```python
data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
```

- `axis=0` concatenates **row-wise** (stacking)
- `axis=1` concatenates **column-wise** (side by side)

`acc_df.iloc[:, :3]` takes only the accelerometer's x/y/z. The metadata columns (participant, label, category, set) are dropped because `gyr_df` already carries them — keeping both would duplicate every one.

### The result is mostly NaN

After merging, **most rows have NaN values** — only around 1,000 rows have data for both sensors simultaneously.

**Why:** the two sensors sample at different rates and their timestamps rarely land on the exact same millisecond. Aligning on a datetime index means a row exists wherever *either* sensor recorded, so each row typically has one sensor's values and NaNs for the other.

This is the problem resampling solves.

### Gotcha: renaming columns is positional

```python
data_merged.columns = ["acc_x", "acc_y", "acc_z",
                       "gyr_x", "gyr_y", "gyr_z",
                       "participant", "label", "category", "set"]
```

Assigning `.columns` renames **by position**. pandas does no name matching and issues no warning. Get the order wrong and your participant column is labelled "label" — it runs clean and corrupts everything downstream.

Always verify after renaming:

```python
data_merged["label"].unique()        # expect bench/squat/... not A/B/C
data_merged["participant"].unique()  # expect A-E
```

---

## 8. Resampling (Frequency Conversion)

```
Accelerometer:  12.500 Hz   ->  one reading every 80ms
Gyroscope:      25.000 Hz   ->  one reading every 40ms
```

Resampling puts both sensors on a **common time grid** so they can occupy the same row. We chose **200ms** — a compromise that keeps enough temporal detail to distinguish exercises without blowing up the row count.

### Different aggregation per column type

```python
sampling = {
    "acc_x": "mean", "acc_y": "mean", "acc_z": "mean",
    "gyr_x": "mean", "gyr_y": "mean", "gyr_z": "mean",
    "label": "last", "category": "last",
    "participant": "last", "set": "last",
}
```

- **Numeric sensor columns -> `mean`**: averaging several readings smooths sensor noise.
- **Categorical columns -> `last`**: taking the mean of "bench" is meaningless. Any value in the window works since they are constant within a set; `last` is simply the convention.

### Gotcha: do not resample the whole DataFrame at once

The dataset spans **about a week**. Resampling it in one call would generate a row for **every single 200ms interval across the entire week**, including all the hours where nobody was training — an enormous DataFrame of empty rows.

**Solution: split by day first, resample each day, then recombine.**

```python
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

data_resampled["set"] = data_resampled["set"].astype("int")
```

`pd.Grouper(freq="D")` splits the frame into one group per day. `.dropna()` removes the empty intervals *within* each day (between sets). The result is **9,009 rows**.

`set` is cast back to `int` because resampling promotes it to float.

---

## 9. Exporting: Why Pickle

```python
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
```

Pickle over CSV because pickle files are:

- **Smaller** in size
- **Faster** to load
- **Type-preserving** — no conversions to worry about when exporting and reading back

That last point matters most here: a CSV round-trip loses the datetime index and the dtypes, so you would have to re-parse everything each time. Pickle restores the DataFrame exactly as it was.

> **Caveat worth knowing:** pickle is Python-specific and not safe to load from untrusted sources, since unpickling can execute arbitrary code. Fine for your own intermediate files, not for data exchange. (Parquet is the usual production answer — fast, typed, and cross-language.)

The file goes in `data/interim/` — processed but not yet the final modelling dataset.

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

Resampling to a common time grid. The accelerometer runs at 12.5 Hz and the gyroscope at 25 Hz, so their timestamps rarely coincide — merging on the datetime index alone leaves most rows half-empty. Resampling both to a fixed 200ms interval, averaging the numeric columns within each window, puts them on the same rows.

</details>

<details>
<summary><b>Q: Why 200ms specifically?</b></summary>

It is a trade-off. Too fine and you keep near-empty rows and a huge dataset; too coarse and you blur the movement pattern that distinguishes one exercise from another. 200ms aggregates roughly 2-3 accelerometer and 5 gyroscope readings per row, smoothing noise while preserving the shape of a repetition.

</details>

<details>
<summary><b>Q: Why did you resample day by day instead of all at once?</b></summary>

The data spans about a week, but training only happened in short bursts. Resampling the whole frame would emit a row for every 200ms interval across the entire week, including all the idle hours — an enormous mostly-empty DataFrame. Grouping by day with `pd.Grouper(freq="D")`, resampling each day, dropping NaNs and concatenating gives the same result without the memory blow-up.

</details>

<details>
<summary><b>Q: Why use mean for the sensor columns but last for the labels?</b></summary>

They are different data types. Averaging several accelerometer readings within a window smooths sensor noise, which is desirable. Averaging a categorical label is undefined — there is no mean of "bench" and "squat". Since the label is constant across a set, taking any single value works, and `last` is the convention.

</details>

<details>
<summary><b>Q: Why export to pickle rather than CSV?</b></summary>

Pickle is smaller, faster to load, and preserves dtypes and the datetime index exactly. A CSV round-trip would flatten the index back to strings and require re-parsing every time. The trade-off is that pickle is Python-only and unsafe to load from untrusted sources, so it suits intermediate artefacts rather than data exchange — Parquet would be the production choice.

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

<details>
<summary><b>Q: Why does sample rate matter on a real device?</b></summary>

Battery. A higher sampling frequency means more measurements, more processing and more power draw. On a wearable that has to last a day, sample rate is a hardware constraint traded against signal fidelity — which is why the accelerometer runs at 12.5 Hz rather than matching the gyroscope's 25 Hz.

</details>

---

## Key Terms

| Term | Meaning |
| --- | --- |
| **Epoch / Unix time** | Milliseconds since 00:00:00 UTC, 1 Jan 1970 |
| **Time-series DataFrame** | DataFrame whose index is a pandas datetime object |
| **Resampling** | Converting a time series to a different fixed frequency |
| **`pd.Grouper(freq="D")`** | Splits a time-indexed frame into one group per day |
| **Hz** | Samples per second |
| **RPE** | Rate of Perceived Exertion — how hard the set felt |
| **Supervised learning** | Training on labelled examples |
| **Multi-class classification** | Predicting one of 3+ mutually exclusive classes |
| **Pickle** | Python binary serialisation — small, fast, type-preserving |
