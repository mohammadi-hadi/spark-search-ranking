"""End-to-end smoke and quality test at reduced scale."""

import argparse

from spark_search_ranking.pipeline import run

ARGS = argparse.Namespace(
    searches=6000,
    users=600,
    items=1200,
    cities=12,
    days=30,
    k=10,
    eta=0.65,
    explore_frac=0.15,
    clip=0.05,
    max_iter=8,
    max_depth=4,
    seed=7,
)


def test_pipeline_end_to_end(spark):
    metrics = run(spark, ARGS)

    assert set(metrics) == {
        "production ranker (logged)",
        "item-CTR popularity",
        "GBT, unweighted clicks",
        "GBT, IPS-weighted clicks",
    }
    for name, m in metrics.items():
        assert 0.0 < m["ndcg@10"] <= 1.0, name
        assert 0.0 < m["mrr"] <= 1.0, name

    # The learned rankers must beat the noisy production ordering on ground truth.
    ips = metrics["GBT, IPS-weighted clicks"]["ndcg@10"]
    prod = metrics["production ranker (logged)"]["ndcg@10"]
    assert ips > prod
