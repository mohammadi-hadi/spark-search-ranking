"""Synthetic marketplace search-log generator with known ground truth.

Simulates a two-sided marketplace search scenario: users issue searches in a
city, an incumbent "production ranker" orders the city's items, and users
click/book following a position-biased examination model:

    P(examine | position) = position ** (-eta)
    P(click   | examine)  = sigmoid(1.5 * relevance - 1.2)
    P(book    | click)    = sigmoid(relevance - 2.2)

where relevance = item quality + a per-(user, item) affinity term. A
configurable fraction of searches ("explore" traffic) is ranked uniformly at
random, which is what makes unbiased propensity estimation possible downstream.

Everything is generated with Spark column expressions (no UDFs) and
hash/seed-based pseudo-randomness. Because Spark's ``rand``/``randn`` draws
depend on the partition layout, the generators pin their ``spark.range``
partition counts (``num_partitions``): with the same Spark version and
``spark.sql.shuffle.partitions``, a given seed yields the identical log on
any machine, regardless of core count. The true relevance and the per-search
relevance grade are kept in the output for evaluation only — they must never
be used as model features.
"""

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

SECONDS_PER_DAY = 86_400


def _sigmoid(x):
    return F.lit(1.0) / (F.lit(1.0) + F.exp(-x))


def generate_catalog(
    spark: SparkSession, n_items: int, n_cities: int, seed: int = 7, num_partitions: int = 8
) -> DataFrame:
    """Item catalog: city assignment, latent quality, and a quality-correlated price."""
    return (
        spark.range(n_items, numPartitions=num_partitions)
        .withColumnRenamed("id", "item_id")
        .withColumn("city_id", F.pmod(F.hash("item_id", F.lit(seed)), F.lit(n_cities)))
        .withColumn("quality", F.randn(seed + 1))
        .withColumn(
            "price",
            F.round(
                F.lit(100.0)
                * F.exp(F.lit(0.15) * F.col("quality") + F.lit(0.25) * F.randn(seed + 2)),
                2,
            ),
        )
    )


def generate_searches(
    spark: SparkSession,
    n_searches: int,
    n_users: int,
    n_cities: int,
    n_days: int = 30,
    explore_frac: float = 0.1,
    seed: int = 7,
    num_partitions: int = 8,
) -> DataFrame:
    """Search events: user, city, timestamp, and an explore-traffic flag."""
    return (
        spark.range(n_searches, numPartitions=num_partitions)
        .withColumnRenamed("id", "search_id")
        .withColumn("user_id", F.pmod(F.hash("search_id", F.lit(seed + 3)), F.lit(n_users)))
        .withColumn("city_id", F.pmod(F.hash("search_id", F.lit(seed + 4)), F.lit(n_cities)))
        .withColumn("day", F.pmod(F.hash("search_id", F.lit(seed + 5)), F.lit(n_days)))
        .withColumn(
            "ts",
            (
                F.col("day") * SECONDS_PER_DAY + F.floor(F.rand(seed + 6) * SECONDS_PER_DAY)
            ).cast("long"),
        )
        .withColumn("is_explore", F.rand(seed + 7) < F.lit(explore_frac))
    )


def generate_impressions(
    searches: DataFrame,
    catalog: DataFrame,
    k: int = 20,
    eta: float = 0.65,
    rank_noise: float = 1.2,
    affinity_scale: float = 0.8,
    seed: int = 7,
) -> DataFrame:
    """Top-k impressions per search with position-biased clicks and bookings.

    The production ranker orders by ``quality + rank_noise * noise`` (a noisy
    proxy of true quality); explore searches are ordered uniformly at random.
    The candidate join (searches x items-in-city) is the scale hotspot — the
    catalog side is broadcast.
    """
    cand = searches.join(F.broadcast(catalog), "city_id")

    affinity = (
        (F.pmod(F.hash("user_id", "item_id", F.lit(seed + 8)), F.lit(10_000)) / F.lit(10_000.0))
        - F.lit(0.5)
    ) * F.lit(2.0 * affinity_scale)

    cand = (
        cand.withColumn("true_relevance", F.col("quality") + affinity)
        .withColumn("prod_score", F.col("quality") + F.lit(rank_noise) * F.randn(seed + 9))
        .withColumn(
            "sort_key",
            F.when(F.col("is_explore"), F.rand(seed + 10)).otherwise(-F.col("prod_score")),
        )
    )

    # Grades rank the full candidate set, so a shown item's grade reflects how
    # good it truly is among everything the search could have returned.
    w_rel = Window.partitionBy("search_id").orderBy(F.col("true_relevance").desc(), "item_id")
    cand = cand.withColumn("rel_rank", F.row_number().over(w_rel)).withColumn(
        "grade",
        F.when(F.col("rel_rank") <= 3, F.lit(2))
        .when(F.col("rel_rank") <= 10, F.lit(1))
        .otherwise(F.lit(0)),
    )

    w_pos = Window.partitionBy("search_id").orderBy("sort_key", "item_id")
    impressions = (
        cand.withColumn("position", F.row_number().over(w_pos))
        .filter(F.col("position") <= k)
    )

    p_exam = F.pow(F.col("position").cast("double"), F.lit(-eta))
    examined = F.rand(seed + 11) < p_exam
    p_click = _sigmoid(F.lit(1.5) * F.col("true_relevance") - F.lit(1.2))
    clicked = examined & (F.rand(seed + 12) < p_click)
    p_book = _sigmoid(F.col("true_relevance") - F.lit(2.2))

    impressions = (
        impressions.withColumn("clicked", clicked)
        .withColumn("booked", F.col("clicked") & (F.rand(seed + 13) < p_book))
    )

    return impressions.select(
        "search_id",
        "user_id",
        "city_id",
        "day",
        "ts",
        "is_explore",
        "item_id",
        "position",
        "price",
        "prod_score",
        "clicked",
        "booked",
        "true_relevance",
        "grade",
    )


def generate_log(
    spark: SparkSession,
    n_searches: int = 20_000,
    n_users: int = 2_000,
    n_items: int = 5_000,
    n_cities: int = 50,
    n_days: int = 30,
    k: int = 20,
    eta: float = 0.65,
    explore_frac: float = 0.1,
    rank_noise: float = 1.2,
    affinity_scale: float = 0.8,
    seed: int = 7,
    num_partitions: int = 8,
) -> DataFrame:
    """End-to-end convenience wrapper: catalog + searches -> impression log."""
    catalog = generate_catalog(spark, n_items, n_cities, seed, num_partitions)
    searches = generate_searches(
        spark, n_searches, n_users, n_cities, n_days, explore_frac, seed, num_partitions
    )
    return generate_impressions(searches, catalog, k, eta, rank_noise, affinity_scale, seed)
