import pytest
from pyspark.sql import functions as F

from spark_search_ranking.bias import add_ips_weights, estimate_position_bias
from spark_search_ranking.datagen import generate_log

ETA = 0.7


@pytest.fixture(scope="module")
def explore_log(spark):
    # Fully randomized traffic isolates the examination effect.
    return generate_log(
        spark,
        n_searches=5000,
        n_users=500,
        n_items=1000,
        n_cities=10,
        k=10,
        eta=ETA,
        explore_frac=1.0,
        seed=11,
    ).cache()


def test_propensities_recover_generator_curve(explore_log):
    props = {
        r["position"]: r["propensity"]
        for r in estimate_position_bias(explore_log).collect()
    }
    assert props[1] == pytest.approx(1.0)
    errors = [abs(props[p] - p ** (-ETA)) for p in range(1, 11)]
    assert max(errors) < 0.12
    assert props[1] > props[4] > props[10]


def test_no_explore_traffic_raises(spark, explore_log):
    exploit_only = explore_log.withColumn("is_explore", F.lit(False))
    with pytest.raises(ValueError):
        estimate_position_bias(exploit_only)


def test_ips_weights(explore_log):
    props = estimate_position_bias(explore_log)
    weighted = add_ips_weights(explore_log, props, clip=0.05)

    unclicked = weighted.filter(~F.col("clicked"))
    assert unclicked.filter(F.col("ips_weight") != 1.0).count() == 0

    clicked = weighted.filter(F.col("clicked"))
    bounds = clicked.agg(
        F.min("ips_weight").alias("lo"), F.max("ips_weight").alias("hi")
    ).first()
    assert bounds["lo"] >= 1.0 - 1e-9
    assert bounds["hi"] <= 1.0 / 0.05 + 1e-9

    # Deeper positions must receive at least the weight of position 1.
    by_pos = {
        r["position"]: r["w"]
        for r in clicked.groupBy("position").agg(F.avg("ips_weight").alias("w")).collect()
    }
    assert by_pos[10] > by_pos[1]
