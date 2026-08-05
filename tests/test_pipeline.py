"""End-to-end smoke and quality test at reduced scale."""

import pytest

from spark_search_ranking.pipeline import PipelineConfig, run

CFG = PipelineConfig(
    searches=6000,
    users=600,
    items=1200,
    cities=12,
    k=10,
    explore_frac=0.15,
    max_iter=8,
    max_depth=4,
)


def test_pipeline_end_to_end(spark):
    result = run(spark, CFG)

    metrics = result["metrics"]
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

    assert result["config"]["searches"] == CFG.searches
    assert result["counts"]["train"] > 0 and result["counts"]["eval"] > 0
    props = {p["position"]: p["propensity"] for p in result["propensities"]}
    assert props[1] == pytest.approx(1.0)


def test_config_defaults_are_complete():
    cfg = PipelineConfig()
    assert cfg.searches > 0 and cfg.days > 1 and 0.0 < cfg.explore_frac < 1.0
