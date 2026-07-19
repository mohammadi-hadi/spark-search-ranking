"""Gradient-boosted-tree re-ranker over the engineered features.

The model is a pointwise click model: GBTClassifier on the ``clicked`` label,
optionally trained with the inverse-propensity weights from
:mod:`spark_search_ranking.bias` (Spark's ``weightCol``). Items are re-ranked
by the predicted click probability. Comparing the weighted and unweighted
variants against the logged production order isolates exactly what the
counterfactual correction buys.
"""

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .features import FEATURE_COLS


def train_reranker(
    train_df: DataFrame,
    weighted: bool = True,
    max_iter: int = 30,
    max_depth: int = 5,
    seed: int = 7,
) -> PipelineModel:
    """Fit the (optionally IPS-weighted) GBT click model."""
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS, outputCol="features", handleInvalid="error"
    )
    gbt = GBTClassifier(
        labelCol="label",
        featuresCol="features",
        maxIter=max_iter,
        maxDepth=max_depth,
        subsamplingRate=0.8,
        seed=seed,
    )
    if weighted:
        gbt.setWeightCol("ips_weight")

    train_df = train_df.withColumn("label", F.col("clicked").cast("double"))
    return Pipeline(stages=[assembler, gbt]).fit(train_df)


def score(model: PipelineModel, df: DataFrame, score_col: str) -> DataFrame:
    """Attach the model's click probability as ``score_col``."""
    scored = model.transform(df).withColumn(
        score_col, vector_to_array(F.col("probability")).getItem(1)
    )
    return scored.drop("features", "rawPrediction", "probability", "prediction", "label")
