from pyspark.sql import functions as F

from spark_search_ranking.datagen import generate_log
from spark_search_ranking.features import (
    FEATURE_COLS,
    USER_PRIOR_CLICKS,
    USER_PRIOR_IMPS,
    build_features,
    item_statistics,
)


def _featured(spark):
    log = generate_log(
        spark, n_searches=1200, n_users=200, n_items=600, n_cities=8, k=10, seed=5
    )
    train = log.filter(F.col("day") < 24)
    return log, build_features(log, item_statistics(train)).cache()


def test_features_present_and_finite(spark):
    _, feats = _featured(spark)
    for col in FEATURE_COLS:
        assert feats.filter(F.col(col).isNull() | F.isnan(F.col(col))).count() == 0, col


def test_user_history_has_no_leakage(spark):
    _, feats = _featured(spark)
    # A user's very first impression has no history: the feature must equal the prior.
    w_first = (
        feats.groupBy("user_id")
        .agg(F.min(F.struct("ts", "search_id", "position", "user_prior_ctr")).alias("s"))
        .select(F.col("s.user_prior_ctr").alias("first_ctr"))
    )
    prior = USER_PRIOR_CLICKS / USER_PRIOR_IMPS
    off = w_first.filter(F.abs(F.col("first_ctr") - F.lit(prior)) > 1e-9)
    assert off.count() == 0


def test_user_history_excludes_current_search(spark):
    _, feats = _featured(spark)
    # All impressions of a search share its timestamp, so the strictly-earlier
    # window must give every row of a search the same history value: clicks on
    # a results page must never feed features of later positions on that page.
    varying = (
        feats.groupBy("search_id")
        .agg(F.countDistinct("user_prior_ctr").alias("n"))
        .filter(F.col("n") > 1)
    )
    assert varying.count() == 0


def test_cold_items_get_prior(spark):
    log, _ = _featured(spark)
    train = log.filter(F.col("day") < 24)
    stats = item_statistics(train)
    test_rows = build_features(log.filter(F.col("day") >= 24), stats)
    cold = test_rows.join(stats.select("item_id"), "item_id", "left_anti")
    if cold.count() > 0:
        distinct_ctr = cold.select("item_ctr").distinct().collect()
        assert len(distinct_ctr) == 1


def test_price_rank_bounds(spark):
    _, feats = _featured(spark)
    bad = feats.filter((F.col("price_rank") < 0) | (F.col("price_rank") > 1))
    assert bad.count() == 0
