"""Leakage-safe feature engineering for the re-ranker.

Two safety rules are enforced by construction:

* Item aggregates (smoothed CTR / booking rate) are computed on the training
  window only and joined onto every row, with the prior mean as the cold-start
  fallback — the evaluation window never contributes to its own features.
* Per-user history uses a cumulative window over strictly earlier timestamps
  (``rangeBetween(unboundedPreceding, -1)`` over ``ts``), so a row can never
  see itself, the future, or any other row of its own results page. The
  same-page exclusion matters at serving time: the ranker scores a whole page
  before any click on that page can be observed.

``true_relevance``, ``grade``, ``quality`` and ``position`` are deliberately
not features: the first three are evaluation-only ground truth, and position
is chosen by the ranker at serving time.
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

FEATURE_COLS = [
    "item_ctr",
    "item_book_rate",
    "user_prior_ctr",
    "price_rank",
    "price_ratio_search",
]

CTR_PRIOR_CLICKS = 2.0
CTR_PRIOR_IMPS = 40.0
BOOK_PRIOR_BOOKS = 1.0
BOOK_PRIOR_IMPS = 80.0
USER_PRIOR_CLICKS = 1.0
USER_PRIOR_IMPS = 20.0


def item_statistics(train_impressions: DataFrame) -> DataFrame:
    """Beta-smoothed item CTR and booking rate from the training window."""
    return train_impressions.groupBy("item_id").agg(
        (
            (F.sum(F.col("clicked").cast("double")) + F.lit(CTR_PRIOR_CLICKS))
            / (F.count(F.lit(1)) + F.lit(CTR_PRIOR_IMPS))
        ).alias("item_ctr"),
        (
            (F.sum(F.col("booked").cast("double")) + F.lit(BOOK_PRIOR_BOOKS))
            / (F.count(F.lit(1)) + F.lit(BOOK_PRIOR_IMPS))
        ).alias("item_book_rate"),
    )


def build_features(impressions: DataFrame, item_stats: DataFrame) -> DataFrame:
    """Attach model features to every impression row."""
    ctr_prior = CTR_PRIOR_CLICKS / CTR_PRIOR_IMPS
    book_prior = BOOK_PRIOR_BOOKS / BOOK_PRIOR_IMPS

    out = (
        impressions.join(item_stats, "item_id", "left")
        .withColumn("item_ctr", F.coalesce(F.col("item_ctr"), F.lit(ctr_prior)))
        .withColumn("item_book_rate", F.coalesce(F.col("item_book_rate"), F.lit(book_prior)))
    )

    # Range frame over strictly earlier timestamps: all impressions of a search
    # share the search's ts, so the frame can only contain whole earlier
    # searches — never rows from this impression's own results page.
    w_user = (
        Window.partitionBy("user_id")
        .orderBy("ts")
        .rangeBetween(Window.unboundedPreceding, -1)
    )
    out = out.withColumn(
        "user_prior_ctr",
        (
            F.coalesce(F.sum(F.col("clicked").cast("double")).over(w_user), F.lit(0.0))
            + F.lit(USER_PRIOR_CLICKS)
        )
        / (F.coalesce(F.count(F.lit(1)).over(w_user), F.lit(0)) + F.lit(USER_PRIOR_IMPS)),
    )

    w_search = Window.partitionBy("search_id").orderBy("price")
    out = out.withColumn("price_rank", F.percent_rank().over(w_search))

    w_search_all = Window.partitionBy("search_id")
    out = out.withColumn(
        "price_ratio_search", F.col("price") / F.avg("price").over(w_search_all)
    )

    return out
