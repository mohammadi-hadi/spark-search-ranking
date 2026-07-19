import pytest

from spark_search_ranking.evaluate import mrr_top_grade, ndcg_at_k


def _df(spark, rows):
    return spark.createDataFrame(rows, "search_id long, item_id long, grade int, s double")


def test_perfect_ordering_gives_ndcg_one(spark):
    df = _df(spark, [(1, 1, 2, 0.9), (1, 2, 1, 0.5), (1, 3, 0, 0.1)])
    assert ndcg_at_k(df, "s", k=10) == pytest.approx(1.0)


def test_reversed_ordering_matches_hand_computation(spark):
    df = _df(spark, [(1, 1, 2, 0.1), (1, 2, 1, 0.5), (1, 3, 0, 0.9)])
    # DCG  = 0/log2(2) + 1/log2(3) + 3/log2(4) = 2.1309298
    # IDCG = 3/log2(2) + 1/log2(3) + 0         = 3.6309298
    assert ndcg_at_k(df, "s", k=10) == pytest.approx(0.5868851, abs=1e-4)


def test_ndcg_cutoff_ignores_items_below_k(spark):
    df = _df(spark, [(1, 1, 0, 0.9), (1, 2, 0, 0.8), (1, 3, 2, 0.1)])
    assert ndcg_at_k(df, "s", k=2) == pytest.approx(0.0)


def test_ndcg_averages_over_searches(spark):
    df = _df(
        spark,
        [(1, 1, 2, 0.9), (1, 2, 0, 0.1), (2, 1, 2, 0.1), (2, 2, 0, 0.9)],
    )
    # Search 1 is perfect (1.0); search 2 puts the grade-2 item second:
    # DCG = 3/log2(3), IDCG = 3/log2(2) -> 0.6309298
    assert ndcg_at_k(df, "s", k=10) == pytest.approx((1.0 + 0.6309298) / 2, abs=1e-4)


def test_mrr(spark):
    df = _df(
        spark,
        [(1, 1, 2, 0.9), (1, 2, 0, 0.1), (2, 1, 0, 0.9), (2, 2, 2, 0.5), (2, 3, 1, 0.1)],
    )
    assert mrr_top_grade(df, "s") == pytest.approx((1.0 + 0.5) / 2)
