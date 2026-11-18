# Kharchify task: approach, results and limits

## Part A, dish category classifier

**Approach.** I clean both files defensively first: names trimmed and whitespace collapsed, categories title-cased, duplicate dish rows dropped, non-numeric quantities and prices coerced and removed. The supplied data turned out tidy, but menu exports rarely are, so the cleaning is part of the deliverable rather than a formality.

The classifier is TF-IDF into Logistic Regression, with two feature blocks: word n-grams of one and two tokens, and character n-grams of three to five characters. Word n-grams catch "chicken biryani". Character n-grams are the reason typos and Indian-English spellings ("panner", "masla") still land in the right place. Logistic Regression was chosen over anything heavier because with 130 examples the job is to generalise from shared vocabulary, not to fit a complex boundary, and because its errors can be read directly from the weights. A 75/25 stratified split keeps every category represented; the seed is fixed at 42.

**Results.** Test accuracy is 0.727 on 33 held-out dishes, and 5-fold cross-validation agrees at 0.723 +/- 0.071, so the hold-out is not a lucky draw. Bread scores 1.00 and Main Course 0.88. Beverage sits at 0.36 and Dessert at 0.40, and the confusion matrix shows why: they swap with each other. "Cold Coffee" and "Chocolate Mousse" share almost no vocabulary with the labels a human would use, and a bag of n-grams has no idea that a cold drink is liquid or that a mousse is sweet.

**Where the five unseen dishes fail.** Rose Lassi (0.83) and Tandoori Broccoli (0.65) are confident and correct. Mango Kulfi Falooda is correct at 0.55 but only just. Paneer Kathi Roll comes back Starter at 0.54 with Main Course at 0.26, and Veg Hakka Noodles is Starter at 0.45: both are mains, and both are wrong for the same reason. The model knows words, and "paneer" appears mostly in starters while no noodle dish exists in training as a main. That is a label-coverage problem, not a model-capacity one.

**What I would improve.** Add a small handful of examples for the missing patterns (rolls, noodles, wraps) rather than changing the algorithm. Route any prediction below 0.6 to a human, since that band is where the errors concentrate. Longer term, derive features from the sales data too, price and time of day distinguish a beverage from a dessert far better than a dish name does.

## Part B, plain English sales questions

**Approach.** A deterministic parser does the understanding and pandas does the arithmetic. It resolves three things: the intent (top sellers by quantity or revenue, quantity of one dish, revenue of one dish, total revenue, slowest sellers), the time window (today, yesterday, this week, last week, this month, last month, or an explicit date), and the dish name, tried as exact, then substring, then a fuzzy match at a 0.72 cutoff. Every answer states the window and the number, and says when a fuzzy read was applied.

There is no language model anywhere in the calculation path, on purpose. A number that a language model produced has no audit trail; a number pandas produced can be traced to rows in a CSV, which is the only kind of answer a restaurant owner should act on.

**Behaviour on inputs it cannot place.** An unrecognised question returns a polite sentence naming what it can answer, rather than a guess. A dish that did not sell that day says so plainly. A date range outside the data says which dates exist.

**Limits.** Single-dish questions only; "how many naan and roti together" needs a joining rule. Week means Monday to Sunday, not the last seven days, and that is worth confirming with whoever uses it. Comparisons ("twice as much as last week") and forecasts are out of scope by design.
