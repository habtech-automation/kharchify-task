"""
Part B: Plain-English sales questions over sales.csv.

Design note: no LLM does any arithmetic. A small rule-based parser works out
*what* is being asked (intent, time window, dish, top-N), and pandas does every
calculation. When the question cannot be understood or the dish cannot be
matched, the assistant says so politely instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from difflib import get_close_matches

import pandas as pd


# ----------------------------------------------------------------------------
# 1. Loading and cleaning
# ----------------------------------------------------------------------------
def load_sales(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["item"] = df["item"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for col in ("quantity", "price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "item", "quantity", "price"])
    df = df[(df["quantity"] > 0) & (df["price"] >= 0)]
    df = df.drop_duplicates()
    df["revenue"] = df["quantity"] * df["price"]
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Time windows
# ----------------------------------------------------------------------------
@dataclass
class Window:
    start: date
    end: date
    label: str


def resolve_window(text: str, today: date) -> Window | None:
    """Map words in the question to a date range. None means 'no time given'."""
    t = text.lower()

    if re.search(r"\btoday\b", t):
        return Window(today, today, "today")
    if re.search(r"\byesterday\b", t):
        d = today - timedelta(days=1)
        return Window(d, d, "yesterday")

    m = re.search(r"(\d{1,2})\s*(?:days?|day)\b", t)
    if re.search(r"last\s+(\d+)\s+days", t) or re.search(r"past\s+(\d+)\s+days", t):
        n = int(m.group(1)) if m else 7
        return Window(today - timedelta(days=n - 1), today, f"the last {n} days")

    if re.search(r"\bthis week\b", t) or re.search(r"\bcurrent week\b", t):
        start = today - timedelta(days=today.weekday())
        return Window(start, today, "this week")
    if re.search(r"\blast week\b", t):
        start = today - timedelta(days=today.weekday() + 7)
        return Window(start, start + timedelta(days=6), "last week")

    if re.search(r"\bthis month\b", t) or re.search(r"\bcurrent month\b", t):
        return Window(today.replace(day=1), today, "this month")
    if re.search(r"\blast month\b", t):
        first_this = today.replace(day=1)
        last_month_end = first_this - timedelta(days=1)
        return Window(last_month_end.replace(day=1), last_month_end, "last month")

    m = re.search(r"(\d{4}-\d{2}-\d{2})", t)
    if m:
        d = date.fromisoformat(m.group(1))
        return Window(d, d, d.isoformat())

    return None


# ----------------------------------------------------------------------------
# 3. Dish matching (exact first, then tolerant of typos)
# ----------------------------------------------------------------------------
def match_dish(text: str, items: list[str]) -> tuple[str | None, bool]:
    """Return (dish, was_fuzzy). Exact or substring match wins; otherwise
    a close spelling is accepted, which is what a human typist needs."""
    t = text.lower().strip()

    for item in items:                      # exact
        if t == item.lower():
            return item, False

    hits = [i for i in items if i.lower() in t]   # longest substring wins
    if hits:
        return max(hits, key=len), False

    close = get_close_matches(t, [i.lower() for i in items], n=1, cutoff=0.72)
    if close:
        return next(i for i in items if i.lower() == close[0]), True

    # last resort: fuzzy match any 2-3 word phrase inside the question
    words = t.split()
    for size in (3, 2):
        for i in range(len(words) - size + 1):
            phrase = " ".join(words[i:i + size])
            close = get_close_matches(phrase, [x.lower() for x in items], n=1, cutoff=0.8)
            if close:
                return next(x for x in items if x.lower() == close[0]), True

    return None, False


# ----------------------------------------------------------------------------
# 4. The question answerer
# ----------------------------------------------------------------------------
class SalesAssistant:
    def __init__(self, df: pd.DataFrame, today: date | None = None):
        self.df = df
        self.items = sorted(df["item"].unique())
        self.today = today or date.today()

    # -- helpers ------------------------------------------------------------
    def _scope(self, window: Window | None) -> pd.DataFrame:
        if window is None:
            return self.df
        return self.df[(self.df["date"] >= window.start) & (self.df["date"] <= window.end)]

    @staticmethod
    def _when(window: Window | None) -> str:
        return f" {window.label}" if window else " across all recorded sales"

    @staticmethod
    def _top_n(text: str, default: int = 3) -> int:
        m = re.search(r"\btop\s+(\d{1,2})\b", text.lower())
        if m:
            return max(1, min(20, int(m.group(1))))
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        m = re.search(r"\btop\s+([a-z]+)\b", text.lower())
        if m and m.group(1) in words:
            return words[m.group(1)]
        return default

    # -- main entry point ---------------------------------------------------
    def ask(self, question: str) -> str:
        if not question or not question.strip():
            return "Ask me about sales, for example: top 3 dishes yesterday."

        q = question.lower().strip()
        window = resolve_window(q, self.today)
        dish, fuzzy = match_dish(q, self.items)
        scope = self._scope(window)

        if scope.empty:
            return (f"I have no sales records for{self._when(window)}. "
                    f"My data covers {self.df['date'].min()} to {self.df['date'].max()}.")

        note = f" (I read that as {dish})" if fuzzy else ""

        # --- quantity of one dish -----------------------------------------
        if dish and re.search(r"\bhow many\b|\bquantity\b|\bsold\b|\bunits\b", q):
            rows = scope[scope["item"] == dish]
            if rows.empty:
                return f"{dish} did not sell{self._when(window)}."
            qty = int(rows["quantity"].sum())
            return f"You sold {qty} x {dish}{self._when(window)}.{note}"

        # --- revenue of one dish ------------------------------------------
        if dish and re.search(r"revenue|earn|income|sales value|turnover|how much", q):
            rows = scope[scope["item"] == dish]
            if rows.empty:
                return f"{dish} did not sell{self._when(window)}."
            value = rows["revenue"].sum()
            return f"{dish} brought in INR {value:,.0f}{self._when(window)}.{note}"

        # --- best / top sellers -------------------------------------------
        if re.search(r"\btop\b|\bbest\b|\bmost popular\b|\bhighest\b|\bbestsell", q):
            n = self._top_n(q)
            by_qty = re.search(r"revenue|value|earn|income|turnover|money", q) is None
            metric = "quantity" if by_qty else "revenue"
            ranked = (scope.groupby("item")[metric].sum()
                      .sort_values(ascending=False).head(n))
            if ranked.empty:
                return "I have no sales in that period."
            unit = "sold" if by_qty else "in revenue"
            lines = [f"{i}. {name}: {int(val)} {unit}" if by_qty
                     else f"{i}. {name}: INR {val:,.0f}"
                     for i, (name, val) in enumerate(ranked.items(), 1)]
            return f"Top {len(lines)} by {'quantity' if by_qty else 'revenue'}{self._when(window)}:\n" + "\n".join(lines)

        # --- worst sellers --------------------------------------------------
        if re.search(r"\bworst\b|\bleast\b|\bslowest\b|\bnot selling\b", q):
            n = self._top_n(q)
            ranked = scope.groupby("item")["quantity"].sum().sort_values().head(n)
            lines = [f"{i}. {name}: {int(val)} sold"
                     for i, (name, val) in enumerate(ranked.items(), 1)]
            return f"Slowest {len(lines)} by quantity{self._when(window)}:\n" + "\n".join(lines)

        # --- total revenue / total orders ----------------------------------
        if re.search(r"\btotal\b.*(revenue|sales|earn|turnover)|how much.*(revenue|money|earn)", q):
            return (f"Total revenue{self._when(window)} was INR "
                    f"{scope['revenue'].sum():,.0f} from {int(scope['quantity'].sum())} items sold.")
        if re.search(r"\btotal\b.*(orders|items|quantity|sold)", q):
            return f"{int(scope['quantity'].sum())} items were sold{self._when(window)}."

        # --- a dish was named but the verb was unclear ---------------------
        if dish:
            rows = scope[scope["item"] == dish]
            if rows.empty:
                return f"{dish} did not sell{self._when(window)}.{note}"
            return (f"{dish}{self._when(window)}: {int(rows['quantity'].sum())} sold, "
                    f"INR {rows['revenue'].sum():,.0f} in revenue.{note}")

        # --- polite fallback -----------------------------------------------
        return ("I could not turn that into a sales question. I can answer things like "
                "\"top 3 dishes yesterday\", \"how many butter naan this week\", "
                "\"revenue from chicken biryani this month\", or \"total sales last week\".")


# ----------------------------------------------------------------------------
# 5. Example run
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_sales("data/sales.csv")
    bot = SalesAssistant(df)
    examples = [
        "What were my top 3 dishes yesterday?",
        "How many Paneer Butter Masala did we sell today?",
        "Top 5 items by revenue this month",
        "How many panner butter masla did we sell today?",
        "Total revenue last week",
        "Which items are not selling?",
        "What is the weather in Mumbai?",
    ]
    for q in examples:
        print(f"Q: {q}\nA: {bot.ask(q)}\n")
