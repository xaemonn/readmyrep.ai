# Video 3 — Visualising the Data

> **Goal:** Create data visualisations to better understand the accelerometer and gyroscope data for different exercises.

**Input:** `data/interim/01_data_processed.pkl` (9,009 rows, 10 columns)

---

## 1. Why Visualise Before Modelling

This is not decoration — it is the step that tells you whether the problem is solvable at all.

**What you are checking for:**

1. **Is the data sane?** Wrong units, flat lines, or nonsense ranges show up instantly in a plot and are invisible in `.describe()`.
2. **Are the classes separable?** If bench and squat produce visibly different signal shapes, a classifier can learn them. If every exercise looks identical, no model will save you.
3. **What features should you build?** Seeing that each rep is a clean oscillation is what motivates the frequency-domain features in video 5 and the peak-counting algorithm in video 7.

**Key finding:** all these exercise signals are **very unique** — visibly different shapes per exercise. That is the green light for the whole project.

> **Interview framing:** "I plotted per-exercise signals before touching a model, to confirm the classes were visually separable and to decide what features to engineer." That answer shows method, not just tool use.

---

## 2. Series vs DataFrame

**My question:** what is the difference in `all_axis_df["acc_y"]`?

| Expression | Returns | Dimensions |
| --- | --- | --- |
| `df["acc_y"]` | **Series** | 1D — a single labelled column |
| `df[["acc_y"]]` | **DataFrame** | 2D — a table that happens to have one column |
| `df[["acc_x", "acc_y", "acc_z"]]` | **DataFrame** | 2D — three columns |

Single brackets pull **one column out** as a Series. Double brackets pass a **list of column names**, so you get a table back.

### Why it matters for plotting

```python
plt.plot(set_df["acc_y"])                       # Series -> one line
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot() # DataFrame -> three lines, auto-legend
```

A DataFrame's `.plot()` draws **one line per column** and builds the legend from the column names automatically. A Series plots a single unnamed line. That is exactly why the multi-axis plot uses double brackets.

**Rule of thumb:** a DataFrame is a dict of Series sharing one index. Selecting one key gives you the Series; selecting a list of keys gives you a smaller DataFrame.

---

## 3. Plotting a Single Set

```python
set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"])
plt.plot(set_df["acc_y"].reset_index(drop=True))
```

### Why `reset_index(drop=True)`

The index is a **datetime**. Plotting against it puts real timestamps on the x-axis, which:

- Makes the axis labels unreadable
- Makes two sets recorded on different days **impossible to overlay** — they sit far apart on the time axis

`reset_index(drop=True)` replaces the datetime index with `0, 1, 2, 3...` — a **sample number**. Every set then starts at x=0 and can be compared directly on the same axes.

`drop=True` discards the old index rather than adding it as a column.

**This is why the x-axis label is "samples" rather than "time".**

---

## 4. Plot Settings

```python
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)   # widen the time-series graphs
mpl.rcParams["figure.dpi"] = 100           # nice resolution on export
```

**Why 20x5 (wide and short):** time-series data is long in one dimension. A square plot squashes hundreds of samples into a narrow space and hides the repetition pattern. A wide aspect ratio lets you actually see individual reps.

**Why set DPI:** controls resolution when figures are exported to `reports/figures/`. 100 is a readable default.

`rcParams` are matplotlib's global defaults — set them once and every subsequent plot inherits them.

---

## 5. Filtering with `.query()`

```python
category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()
```

`.query()` takes a string expression and is equivalent to boolean masking:

```python
df.query("label == 'squat'")      # same as
df[df["label"] == "squat"]
```

It reads more cleanly when chaining several conditions.

### f-strings inside query

```python
label = "squat"
participant = "A"
df.query(f"label == '{label}'").query(f"participant == '{participant}'")
```

The f-string **enables variables to be written inside the query string**. Note the nested quotes: double quotes for the f-string, single quotes around the value inside it.

---

## 6. Comparing Groups on One Plot

```python
category_df.groupby(["category"])["acc_y"].plot()
```

`groupby(...).plot()` draws **one line per group** on the same axes. Comparing medium vs heavy sets for the same exercise and participant isolates the effect of load: heavier sets move slower, so the oscillations stretch out.

Same pattern for comparing participants:

```python
participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()
participant_df.groupby(["participant"])["acc_y"].plot()
```

`sort_values("participant")` keeps the legend in alphabetical order.

**What to look for:** signals that share a shape across participants but differ in amplitude or tempo. Shared shape means the model can generalise across people; wildly different shapes would warn of a personalisation problem.

---

## 7. Looping Over All Combinations

```python
labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()
```

### Why the `if len(...) > 0` guard

**Some participants did not perform some exercises.** Without the guard you get empty plots and matplotlib warnings.

Actual coverage (rows per combination, 0 = missing):

| label | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| bench | 279 | 157 | 232 | 359 | 638 |
| dead | 534 | **0** | 463 | **0** | 534 |
| ohp | 726 | 561 | 170 | **0** | 219 |
| rest | 740 | **0** | **0** | **0** | 370 |
| row | 85 | **0** | 340 | 309 | 683 |
| squat | 624 | 125 | 276 | 384 | 201 |

**7 of the 30 combinations are empty.** This is worth remembering — it is also a mild class-imbalance warning for video 6.

`.title()` on the string capitalises each word for the plot title.

---

## 8. Plotting Both Sensors

The same loop is repeated with `[["gyr_x", "gyr_y", "gyr_z"]]` instead of the accelerometer columns. Next step in the video is combining both sensors into a **single figure with two stacked subplots**, so accelerometer and gyroscope for the same set can be read together.

---

## Interview Questions

<details>
<summary><b>Q: Why visualise the data before building a model?</b></summary>

Three reasons. To sanity-check the data — wrong units or flat signals are obvious in a plot and invisible in summary statistics. To confirm the classes are separable, since if the exercises look identical no classifier will help. And to decide what features to engineer: seeing that each repetition is a clean oscillation is what motivates frequency-domain features and the peak-detection rep counter.

</details>

<details>
<summary><b>Q: What is the difference between a pandas Series and a DataFrame?</b></summary>

A Series is one-dimensional — a single labelled column. A DataFrame is two-dimensional, effectively a dict of Series sharing one index. `df["col"]` returns a Series; `df[["col"]]` passes a list and returns a one-column DataFrame. It matters for plotting: a DataFrame's `.plot()` draws one line per column with an automatic legend, while a Series plots a single line.

</details>

<details>
<summary><b>Q: Why reset the index before plotting?</b></summary>

The index is a datetime. Plotting against it spreads sets recorded on different days across the time axis, so they cannot be overlaid or compared. Resetting to an integer sample number puts every set on a common x-axis starting at zero.

</details>

<details>
<summary><b>Q: What did the visualisations tell you?</b></summary>

That the exercises produce visibly distinct signal shapes, which confirmed the classification problem was tractable before any modelling. They also revealed that heavier sets have slower, stretched oscillations, and that some participant/exercise combinations have no data at all — 7 of 30 are empty.

</details>

<details>
<summary><b>Q: Why is the figure size set to 20x5?</b></summary>

Time-series data is long in one dimension. A square figure compresses hundreds of samples horizontally and hides the repetition pattern; a wide, short aspect ratio makes individual reps visible.

</details>

<details>
<summary><b>Q: You found missing participant/exercise combinations. Why does that matter?</b></summary>

It signals class imbalance and uneven coverage. Some exercises are represented by fewer participants, so the model has less variety to generalise from for those classes. It also matters for the train/test split — splitting naively could leave a class with no representation in one of the folds.

</details>

---

## Key Terms

| Term | Meaning |
| --- | --- |
| **Series** | 1D labelled array — one column |
| **DataFrame** | 2D table — a dict of Series sharing an index |
| **`reset_index(drop=True)`** | Replace the index with 0,1,2... and discard the old one |
| **`.query()`** | Filter rows using a string expression |
| **`rcParams`** | matplotlib global default settings |
| **DPI** | Dots per inch — export resolution |
| **`groupby(...).plot()`** | One line per group on shared axes |
