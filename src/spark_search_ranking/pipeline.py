"""End-to-end pipeline: generate -> estimate bias -> weight -> train -> evaluate.

Run from the command line:

    spark-search-ranking --searches 20000

or programmatically:

    from spark_search_ranking import PipelineConfig, run
    result = run(spark, PipelineConfig(searches=20_000))

The last 20% of days form the evaluation window. Four orderings are compared
on it: the logged production ranker, a popularity baseline (smoothed item
CTR), an unweighted GBT click model, and the IPS-weighted GBT click model.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from .bias import add_ips_weights, estimate_position_bias
from .datagen import generate_log
from .evaluate import mrr_top_grade, ndcg_at_k
from .features import build_features, item_statistics
from .rank import score, train_reranker


@dataclass(frozen=True)
class PipelineConfig:
    """Scale, bias, and training knobs; defaults double as the CLI defaults."""

    searches: int = 20_000
    users: int = 2_000
    items: int = 5_000
    cities: int = 50
    days: int = 30
    k: int = 20
    eta: float = 0.65
    explore_frac: float = 0.1
    clip: float = 0.05
    max_iter: int = 30
    max_depth: int = 5
    seed: int = 7


def run(
    spark: SparkSession, config: PipelineConfig | None = None, *, verbose: bool = False
) -> dict:
    """Run the full pipeline and return config, split sizes, propensities, and metrics."""
    cfg = config if config is not None else PipelineConfig()
    log = generate_log(
        spark,
        n_searches=cfg.searches,
        n_users=cfg.users,
        n_items=cfg.items,
        n_cities=cfg.cities,
        n_days=cfg.days,
        k=cfg.k,
        eta=cfg.eta,
        explore_frac=cfg.explore_frac,
        seed=cfg.seed,
    ).cache()

    train_days = int(cfg.days * 0.8)
    train_raw = log.filter(F.col("day") < train_days)
    test_raw = log.filter(F.col("day") >= train_days)

    propensities = estimate_position_bias(train_raw)
    weighted_log = add_ips_weights(log, propensities, clip=cfg.clip)

    stats = item_statistics(train_raw)
    featured = build_features(weighted_log, stats).cache()
    train = featured.filter(F.col("day") < train_days)
    test = featured.filter(F.col("day") >= train_days)

    model_ips = train_reranker(
        train, weighted=True, max_iter=cfg.max_iter, max_depth=cfg.max_depth, seed=cfg.seed
    )
    model_plain = train_reranker(
        train, weighted=False, max_iter=cfg.max_iter, max_depth=cfg.max_depth, seed=cfg.seed
    )

    test_scored = score(model_ips, test, "score_ips")
    test_scored = score(model_plain, test_scored, "score_plain")

    orderings = {
        "production ranker (logged)": "prod_score",
        "item-CTR popularity": "item_ctr",
        "GBT, unweighted clicks": "score_plain",
        "GBT, IPS-weighted clicks": "score_ips",
    }
    metrics = {}
    for name, col in orderings.items():
        metrics[name] = {
            "ndcg@10": ndcg_at_k(test_scored, col, k=10),
            "mrr": mrr_top_grade(test_scored, col),
        }

    result = {
        "config": asdict(cfg),
        "counts": {"train": train_raw.count(), "eval": test_raw.count()},
        "propensities": [
            {
                "position": row["position"],
                "propensity": float(row["propensity"]),
                "impressions": int(row["impressions"]),
            }
            for row in propensities.collect()
        ],
        "metrics": metrics,
    }
    if verbose:
        _print_report(result)
    return result


def _print_report(result: dict) -> None:
    cfg, counts = result["config"], result["counts"]
    print(
        f"\nimpressions: train={counts['train']:,}  eval={counts['eval']:,}"
        f"  (k={cfg['k']}, eta={cfg['eta']})"
    )
    print("\nestimated propensities (explore slice):")
    for row in result["propensities"][:10]:
        print(f"  position {row['position']:>2}: {row['propensity']:.3f}")
    print(f"\n{'ordering':<28} {'NDCG@10':>8} {'MRR':>8}")
    for name, m in result["metrics"].items():
        print(f"{name:<28} {m['ndcg@10']:>8.4f} {m['mrr']:>8.4f}")


def main() -> None:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    # One flag per config field keeps the CLI and PipelineConfig from drifting.
    for field in fields(PipelineConfig):
        parser.add_argument(
            f"--{field.name.replace('_', '-')}",
            type=field.type,
            default=getattr(defaults, field.name),
        )
    parser.add_argument("--output", type=Path, default=None, help="write full result JSON here")
    args = parser.parse_args()
    cfg = PipelineConfig(**{f.name: getattr(args, f.name) for f in fields(PipelineConfig)})

    spark = (
        SparkSession.builder.appName("spark-search-ranking")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    started = time.time()
    result = run(spark, cfg, verbose=True)
    print(f"\ndone in {time.time() - started:.1f}s")
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.output}")
    spark.stop()


if __name__ == "__main__":
    main()
