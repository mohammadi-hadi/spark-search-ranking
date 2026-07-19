"""Position-bias (examination propensity) estimation and IPS weighting.

Propensities are estimated from the randomized "explore" slice of the log
only. On uniformly shuffled results, relevance is independent of position, so
the per-position click-through rate is proportional to the examination
probability, and normalizing by position 1 yields the propensity curve
directly (Craswell et al.'s examination hypothesis; Joachims et al.'s
counterfactual LTR setup).

Estimating propensities from exploit traffic instead would confound
examination with the production ranker's placement of relevant items on top —
the classic mistake this module exists to avoid.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def estimate_position_bias(impressions: DataFrame) -> DataFrame:
    """Return (position, propensity) estimated from explore traffic.

    propensity(p) = CTR_explore(p) / CTR_explore(1), clipped into (0, 1].
    """
    explore = impressions.filter(F.col("is_explore"))
    if explore.limit(1).count() == 0:
        raise ValueError("no explore traffic in the log; cannot estimate propensities")

    by_pos = explore.groupBy("position").agg(
        F.avg(F.col("clicked").cast("double")).alias("ctr"),
        F.count(F.lit(1)).alias("impressions"),
    )
    ctr_1 = by_pos.filter(F.col("position") == 1).select("ctr").first()[0]
    if not ctr_1 or ctr_1 <= 0.0:
        raise ValueError("CTR at position 1 is zero in explore traffic; log too small")

    return (
        by_pos.withColumn("propensity", F.least(F.col("ctr") / F.lit(ctr_1), F.lit(1.0)))
        .select("position", "propensity", "impressions")
        .orderBy("position")
    )


def add_ips_weights(
    impressions: DataFrame, propensities: DataFrame, clip: float = 0.05
) -> DataFrame:
    """Attach inverse-propensity weights for click-model training.

    Clicked rows get weight 1 / max(propensity, clip); unclicked rows keep
    weight 1. Weighting only the positives corrects the examination bias of
    observed clicks under the examination hypothesis; unclicked rows are left
    untouched because "not examined" and "examined but rejected" cannot be
    distinguished in the log. Clipping bounds the variance contributed by
    low-propensity (deep) positions at the cost of a small residual bias.
    """
    props = propensities.select("position", "propensity")
    return (
        impressions.join(props, "position", "left")
        .withColumn(
            "ips_weight",
            F.when(
                F.col("clicked"),
                F.lit(1.0) / F.greatest(F.coalesce(F.col("propensity"), F.lit(1.0)), F.lit(clip)),
            ).otherwise(F.lit(1.0)),
        )
        .drop("propensity")
    )
