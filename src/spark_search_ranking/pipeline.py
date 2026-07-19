"""End-to-end pipeline: generate -> estimate bias -> weight -> train -> evaluate.

Run locally:

    python -m spark_search_ranking.pipeline --searches 20000

The last 20% of days form the evaluation window. Four orderings are compared
on it: the logged production ranker, a popularity baseline (smoothed item
CTR), an unweighted GBT click model, and the IPS-weighted GBT click model.
"""

import argparse
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from .bias import add_ips_weights, estimate_position_bias
from .datagen import generate_log
from .evaluate import mrr_top_grade, ndcg_at_k
from .features import build_features, item_statistics
from .rank import score, train_reranker


def run(spark: SparkSession, args: argparse.Namespace) -> dict:
    log = generate_log(
        spark,
        n_searches=args.searches,
        n_users=args.users,
        n_items=args.items,
        n_cities=args.cities,
        n_days=args.days,
        k=args.k,
        eta=args.eta,
        explore_frac=args.explore_frac,
        seed=args.seed,
    ).cache()

    train_days = int(args.days * 0.8)
    train_raw = log.filter(F.col("day") < train_days)
    test_raw = log.filter(F.col("day") >= train_days)

    propensities = estimate_position_bias(train_raw)
    weighted_log = add_ips_weights(log, propensities, clip=args.clip)

    stats = item_statistics(train_raw)
    featured = build_features(weighted_log, stats).cache()
    train = featured.filter(F.col("day") < train_days)
    test = featured.filter(F.col("day") >= train_days)

    model_ips = train_reranker(
        train, weighted=True, max_iter=args.max_iter, max_depth=args.max_depth, seed=args.seed
    )
    model_plain = train_reranker(
        train, weighted=False, max_iter=args.max_iter, max_depth=args.max_depth, seed=args.seed
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

    n_train = train_raw.count()
    n_test = test_raw.count()
    print(f"\nimpressions: train={n_train:,}  eval={n_test:,}  (k={args.k}, eta={args.eta})")
    print("\nestimated propensities (explore slice):")
    for row in propensities.collect()[:10]:
        print(f"  position {row['position']:>2}: {row['propensity']:.3f}")
    print(f"\n{'ordering':<28} {'NDCG@10':>8} {'MRR':>8}")
    for name, m in metrics.items():
        print(f"{name:<28} {m['ndcg@10']:>8.4f} {m['mrr']:>8.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--searches", type=int, default=20_000)
    parser.add_argument("--users", type=int, default=2_000)
    parser.add_argument("--items", type=int, default=5_000)
    parser.add_argument("--cities", type=int, default=50)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.65)
    parser.add_argument("--explore-frac", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("spark-search-ranking")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    started = time.time()
    run(spark, args)
    print(f"\ndone in {time.time() - started:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
