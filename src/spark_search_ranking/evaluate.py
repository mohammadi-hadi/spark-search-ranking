"""Ranking metrics computed against the generator's ground-truth grades.

Because the log is synthetic, every impression carries a relevance grade
derived from the true (hidden) relevance: 2 for the search's top-3 items,
1 for the top-10, 0 otherwise. Evaluating orderings against these grades
avoids the trap of evaluating on logged clicks, which are themselves
position-biased.

All metrics are implemented with window functions and aggregate to a single
number per ordering, so they run unchanged at any scale.
"""

import math

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def ndcg_at_k(df: DataFrame, score_col: str, k: int = 10) -> float:
    """Mean NDCG@k over searches for the ordering induced by ``score_col``."""
    w_score = Window.partitionBy("search_id").orderBy(F.col(score_col).desc(), "item_id")
    w_ideal = Window.partitionBy("search_id").orderBy(F.col("grade").desc(), "item_id")

    gains = df.select(
        "search_id",
        "grade",
        F.row_number().over(w_score).alias("pos"),
        F.row_number().over(w_ideal).alias("ideal_pos"),
    )

    def discounted(pos_col: str) -> F.Column:
        return F.when(
            F.col(pos_col) <= k,
            (F.pow(F.lit(2.0), F.col("grade")) - F.lit(1.0))
            / (F.log(F.lit(2.0), F.col(pos_col) + F.lit(1.0))),
        ).otherwise(F.lit(0.0))

    per_search = gains.groupBy("search_id").agg(
        F.sum(discounted("pos")).alias("dcg"),
        F.sum(discounted("ideal_pos")).alias("idcg"),
    )
    result = (
        per_search.filter(F.col("idcg") > 0)
        .agg(F.avg(F.col("dcg") / F.col("idcg")))
        .first()[0]
    )
    return float(result) if result is not None else math.nan


def mrr_top_grade(df: DataFrame, score_col: str) -> float:
    """Mean reciprocal rank of the first grade-2 item under ``score_col``."""
    w_score = Window.partitionBy("search_id").orderBy(F.col(score_col).desc(), "item_id")
    ranked = df.withColumn("pos", F.row_number().over(w_score))
    first_hit = (
        ranked.filter(F.col("grade") == 2)
        .groupBy("search_id")
        .agg(F.min("pos").alias("first_pos"))
    )
    result = first_hit.agg(F.avg(F.lit(1.0) / F.col("first_pos"))).first()[0]
    return float(result) if result is not None else math.nan
