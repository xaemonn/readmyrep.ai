# Video 4 — Outlier Detection

> **Goal:** Check whether there are outliers (extreme values) in our data that we want to remove, using various methods.

**Input:** `data/interim/01_data_processed.pkl` (9,009 rows)

**Three methods covered:**

| Method | Type | Basis |
| --- | --- | --- |
| **IQR** | Distribution-based | Quartiles — no distribution assumption |
| **Chauvenet's criterion** | Distribution-based | Assumes a normal distribution |
| **Local Outlier Factor** | Distance-based | Local density of neighbours |

---

## 1. Why Outlier Detection Matters Here

An outlier is a value far outside the normal range. Two possibilities, and telling them apart is the whole job:

1. **Sensor error or glitch** -> remove it, it is noise
2. **Genuine extreme movement** -> keep it, it may be the signal

Getting this wrong in either direction is costly. Delete real data and you strip out the very peaks that distinguish a heavy deadlift. Keep sensor glitches and they distort every downstream feature.

> **Key caution from the video:** plot outliers **in time**, because some flagged values may not actually be outliers. A boxplot collapses the time dimension entirely — a value can look extreme in a histogram while being a perfectly normal part of a repetition.

---

## 2. Setup Gotcha

Plots were not showing in the interactive window. Fix:

```python
%matplotlib inline
```

This is a magic command — interactive window or notebook only, not a plain `.py` run. It must run before matplotlib draws anything, since the backend locks in on first use.

---

## 3. First Look: Boxplots

```python
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

outlier_columns = list(df.columns[:6])   # acc_x/y/z, gyr_x/y/z

df[["acc_y", "label"]].boxplot(by="label", figsize=(20,10))
df[outlier_columns[:3] + ["label"]].boxplot(by="label", figsize=(20,10), layout=(1,3))
df[outlier_columns[3:] + ["label"]].boxplot(by="label", figsize=(20,10), layout=(1,3))
```

`df.columns[:6]` grabs the six sensor columns and skips the metadata (participant, label, category, set) — you never want to run outlier detection on categorical columns.

`by="label"` splits into one box per exercise, which is what reveals that outliers are not spread evenly across classes.

**Reading a boxplot:** the box spans Q1 to Q3, the line inside is the median, whiskers extend 1.5x IQR, and anything beyond the whiskers is plotted as an individual outlier point.

---

## 4. The Helper: `plot_binary_outliers`

Taken from the ML4QS book's open-source repository ([mhoogen/ML4QS](https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py)).

It plots the real values against sample number, with **normal points as `+` and outliers as red `r+`**. This is the "plot outliers in time" view — it shows *where* in the recording a flagged point sits, which a boxplot cannot.

```python
dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
dataset[outlier_col] = dataset[outlier_col].astype("bool")
```

The `astype("bool")` matters because the column is used as a boolean mask (`~dataset[outlier_col]`), and masking with floats or objects fails.

---

## 5. Interquartile Range (IQR)

**Distribution-based method.** Makes no assumption about the shape of the distribution — it only uses quartiles.

```python
Q1 = dataset[col].quantile(0.25)
Q3 = dataset[col].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (dataset[col] > upper_bound)
```

- **Q1** = 25th percentile, **Q3** = 75th percentile
- **IQR** = the range covering the middle 50% of the data
- Anything more than **1.5 x IQR** beyond either quartile is flagged

**Why 1.5?** It is the convention Tukey chose when inventing the boxplot. For normally distributed data it corresponds to roughly 2.7 standard deviations, flagging about 0.7% of points. It is a tuning knob, not a law — raise it to flag fewer points, lower it to flag more.

**Why IQR is robust:** quartiles are barely affected by extreme values, unlike the mean and standard deviation, which get dragged toward the very outliers you are trying to find.

### Results across the whole dataset

| Column | Outliers | Share |
| --- | --- | --- |
| acc_x | 631 | 7.0% |
| acc_y | 0 | 0.0% |
| acc_z | 2 | 0.0% |
| gyr_x | 642 | 7.1% |
| gyr_y | 1,011 | 11.2% |
| gyr_z | 1,485 | 16.5% |

**Gyroscope columns produce far more outliers than accelerometer columns.** Rotation varies much more freely than linear acceleration during a lift.

Flagging 16.5% of a column as outliers is a red flag in itself — a genuine outlier rate should be small. That is a sign the method is being applied wrongly, not that the data is broken.

---

## 6. The Key Finding: Outliers Cluster in REST

**My observation:**

> When the majority of data is in blue dots and a few sets within the data look red, statistically that red data is underrepresented. Mean and standard deviation are determined by the majority of the data.

> In IQR we looked at all sets in one plot. Red outliers were in REST sets — during rest, participants had no limitations on what they do.

**Quantified:**

```
rest is  12.3% of all rows
but      83.7% of all acc_x outliers
```

That is a 7x over-representation. During rest, participants could do anything — scratch, walk, fidget, put the bar down — so the motion is unconstrained and looks nothing like the repetitive pattern of a lift.

### Why this breaks the method

IQR computed over the **entire dataset** builds one set of bounds from a pool that mixes six different activities. The quartiles are dominated by the majority classes (the actual exercises), so any activity with a genuinely different distribution gets flagged wholesale.

**These are not errors. They are a different distribution being judged by the wrong yardstick.**

### The fix

> So let's split the data (by exercise) into groups and then apply this method to improve the results.

Group by `label` first, then run outlier detection **within each group**. Each exercise is then judged against its own distribution, and rest stops being flagged for the crime of being rest.

This is the central lesson of the video: **outlier detection must be applied within homogeneous groups**, not across a mixed population.

---

## 7. Still To Do

- **Chauvenet's criterion** — check for normal distribution first, then apply
- **Local Outlier Factor** — distance-based, uses local density instead of global bounds
- Compare all three grouped by label
- Choose one method, replace outliers with NaN, export

---

## Interview Questions

<details>
<summary><b>Q: How did you handle outliers, and why that approach?</b></summary>

I compared three methods: IQR and Chauvenet's criterion, both distribution-based, and Local Outlier Factor, which is distance-based. The important finding was not which method won but that applying any of them across the whole dataset was wrong — the data contains six different activities with different distributions, so detection had to be run per exercise group.

</details>

<details>
<summary><b>Q: What is the IQR method and why is it robust?</b></summary>

It flags values more than 1.5 times the interquartile range beyond the first or third quartile. It is robust because quartiles are barely influenced by extreme values, unlike the mean and standard deviation, which are pulled toward the outliers you are trying to detect. It also makes no assumption about the distribution's shape.

</details>

<details>
<summary><b>Q: You found 83.7% of outliers came from the rest class, which is only 12.3% of the data. What did you conclude?</b></summary>

That the method was misapplied rather than the data being faulty. Rest periods are unconstrained movement, so their distribution differs fundamentally from the repetitive patterns of a lift. Computing global bounds over a mixed population means the majority classes set the thresholds and any minority distribution gets flagged wholesale. The fix was to group by exercise and detect outliers within each group.

</details>

<details>
<summary><b>Q: Why is 1.5 the multiplier?</b></summary>

Convention, from Tukey's original boxplot. For normally distributed data it corresponds to about 2.7 standard deviations and flags roughly 0.7% of points. It is a tunable parameter, not a rule — larger values are more permissive, smaller ones more aggressive.

</details>

<details>
<summary><b>Q: How do you decide whether an outlier is an error or real data?</b></summary>

By plotting it in time rather than relying on a boxplot. A boxplot collapses the time axis, so a value can look extreme in aggregate while being a normal part of a repetition. Plotting flagged points against sample number shows whether they are isolated spikes, which suggests sensor error, or sustained stretches, which suggests genuine movement the method has misjudged.

</details>

<details>
<summary><b>Q: Why did the gyroscope columns produce so many more outliers?</b></summary>

Rotation varies more freely than linear acceleration during a lift. The gyroscope flagged up to 16.5% of values on the raw global method versus near zero for two of the accelerometer axes. A rate that high is itself evidence the method is wrong for the data rather than evidence of bad sensors.

</details>

<details>
<summary><b>Q: Why exclude the metadata columns from outlier detection?</b></summary>

They are categorical — participant, label, category, set. Quartiles and distances are undefined for them. `df.columns[:6]` selects only the six numeric sensor axes.

</details>

---

## Key Terms

| Term | Meaning |
| --- | --- |
| **Outlier** | Value far outside the normal range — either sensor error or genuine extreme |
| **IQR** | Interquartile range, Q3 - Q1, covering the middle 50% |
| **Q1 / Q3** | 25th and 75th percentiles |
| **Tukey's fence** | The 1.5 x IQR boundary |
| **Distribution-based** | Flags points by where they sit in the value distribution |
| **Distance-based** | Flags points by how isolated they are from neighbours |
| **Robust statistic** | One that is not distorted by extreme values (median, quartiles) |
| **Chauvenet's criterion** | Normality-assuming test for rejecting a data point |
| **LOF** | Local Outlier Factor — density-based detection |
