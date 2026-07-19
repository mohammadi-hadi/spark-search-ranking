from pyspark.sql import functions as F

from spark_search_ranking.datagen import generate_log

ARGS = dict(n_searches=1500, n_users=300, n_items=800, n_cities=10, k=10, seed=7)


def test_log_shape_and_positions(spark):
    log = generate_log(spark, **ARGS).cache()
    per_search = log.groupBy("search_id").agg(
        F.count(F.lit(1)).alias("n"),
        F.min("position").alias("min_pos"),
        F.max("position").alias("max_pos"),
        F.countDistinct("item_id").alias("n_items"),
    )
    bad = per_search.filter(
        (F.col("n") > 10)
        | (F.col("min_pos") != 1)
        | (F.col("max_pos") != F.col("n"))
        | (F.col("n_items") != F.col("n"))
    )
    assert bad.count() == 0


def test_determinism(spark):
    a = generate_log(spark, **ARGS)
    b = generate_log(spark, **ARGS)
    stats = [
        (
            df.count(),
            df.filter("clicked").count(),
            df.filter("booked").count(),
        )
        for df in (a, b)
    ]
    assert stats[0] == stats[1]
    assert stats[0][0] > 0 and stats[0][1] > 0


def test_ctr_decays_with_position(spark):
    log = generate_log(spark, **ARGS)
    ctr = {
        r["position"]: r["ctr"]
        for r in log.groupBy("position")
        .agg(F.avg(F.col("clicked").cast("double")).alias("ctr"))
        .collect()
    }
    assert ctr[1] > ctr[5] > ctr[10]


def test_grades_follow_true_relevance(spark):
    log = generate_log(spark, **ARGS)
    by_grade = {
        r["grade"]: r["rel"]
        for r in log.groupBy("grade")
        .agg(F.avg("true_relevance").alias("rel"))
        .collect()
    }
    assert by_grade[2] > by_grade[1] > by_grade[0]
