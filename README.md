# Kharchify AI/ML Internship Task

Two deliverables in one small, reproducible repository.

| Part | What it does | File |
| --- | --- | --- |
| A | Dish category classifier: TF-IDF (word + character n-grams) into Logistic Regression | `part_a.py`, `kharchify_task.ipynb` |
| B | Plain English sales questions answered with pandas, with a safe fallback | `part_b.py` |

## Data

* `data/dishes.csv` - 130 dish names, 6 categories (Starter, Main Course, Dessert, Bread, Rice, Beverage)
* `data/sales.csv` - 937 order lines, 20 Oct to 17 Nov 2026, columns `date, item, quantity, price`

## Setup

```bash
git clone <this repo>
cd kharchify-task
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Model training and evaluation (writes `part_a_results.json` and `category_model.joblib`):

```bash
python part_a.py
```

Sales questions:

```bash
python part_b.py
```

Or open the notebook, which runs both parts with outputs saved:

```bash
jupyter notebook kharchify_task.ipynb
```

## Results

* Test accuracy **0.727** on a 25% stratified hold-out (33 dishes), 5-fold CV **0.723 +/- 0.071**
* Bread and Main Course are reliable; Beverage and Dessert are the weak pair, and they fail into each other
* See `WRITEUP.md` for the approach, the limits, and what I would do next

## Design choices worth knowing

1. **No LLM touches the arithmetic in Part B.** A deterministic parser resolves the question and pandas computes every figure, so each answer can be traced back to rows.
2. **Fuzzy dish matching.** "panner butter masla" still resolves, because real users type like that.
3. **A polite, honest fallback.** A question the parser cannot place gets a sentence naming what it can answer, never a guess.

## Repository layout

```
.
├── data/
│   ├── dishes.csv
│   └── sales.csv
├── part_a.py               # classifier, evaluation, unseen dishes
├── part_b.py               # question parser and pandas answering
├── kharchify_task.ipynb    # notebook, outputs included
├── part_a_results.json     # metrics from the last run
├── part_b_examples.txt     # sample question/answer pairs
├── WRITEUP.md              # one page: approach, results, limits
├── requirements.txt
└── README.md
```
