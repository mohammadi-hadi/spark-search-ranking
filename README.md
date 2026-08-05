# spark-search-ranking

[![CI](https://github.com/mohammadi-hadi/spark-search-ranking/actions/workflows/ci.yml/badge.svg)](https://github.com/mohammadi-hadi/spark-search-ranking/actions/workflows/ci.yml)

Counterfactual learning-to-rank for marketplace search logs, implemented end to end in PySpark: position-bias estimation from randomized traffic, inverse-propensity-weighted training, and NDCG evaluation against known ground truth.

Clicks in search logs are a biased signal — users click what they *see*, and what they see is what the incumbent ranker put on top. Training the next ranker naively on those clicks bakes the old ranker's mistakes in. This project is a compact, fully tested reference implementation of the standard correction: estimate examination propensities from a randomized "explore" slice, reweight clicks by inverse propensity, and re-rank.

Because the log is synthetic with a known generative process, the pipeline's correctness is *provable* in tests: the propensity estimator must recover the generator's position-bias curve, and the re-ranker must beat the logged production ordering on true relevance.

```mermaid
flowchart LR
    A["synthetic search log<br/>90% exploit / 10% explore"] --> B["propensity estimation<br/>explore slice only"]
    A --> C["leakage-safe features<br/>cumulative windows"]
    B --> D["IPS weights"]
    C --> E["GBT re-ranker<br/>weightCol"]
    D --> E
    E --> F["NDCG@10 / MRR<br/>vs. ground-truth grades"]
```

## What it demonstrates

- **Position-bias correction**: propensities estimated from randomized traffic only (estimating them from exploit traffic confounds examination with relevance — the classic mistake), IPS weighting with variance clipping.
- **Leakage-safe feature engineering**: item statistics from the training window only with cold-start priors; per-user history via cumulative windows over strictly earlier events (`rowsBetween(unboundedPreceding, -1)`).
- **Spark at the core**: no UDFs anywhere — generation, features, and metrics are all column expressions and window functions, so everything scales with the cluster. Generation pins its partition layout, so a given seed yields the identical log on any machine (same Spark version and shuffle settings).
- **Testable ML**: a deterministic generator with known ground truth turns "does the correction work?" into unit tests, including a hand-computed NDCG case and a no-leakage assertion.

## Installation

```bash
pip install "spark-search-ranking @ git+https://github.com/mohammadi-hadi/spark-search-ranking.git"
```

Pin a release by appending `@v0.2.0` to the URL; wheels and sdists are also attached to [GitHub Releases](https://github.com/mohammadi-hadi/spark-search-ranking/releases). Requires Python ≥ 3.10, PySpark 3.5–4.2 (both tested in CI), and a JVM — Java 17 works across the whole supported range.

## Quickstart

Installing gives you the `spark_search_ranking` library and a `spark-search-ranking` console script:

```bash
spark-search-ranking --searches 20000             # or: python -m spark_search_ranking.pipeline
```

Scale knobs: `--searches`, `--items`, `--users`, `--k` (results per search), `--eta` (position-bias strength), `--explore-frac`; `--output metrics.json` writes the full result (config, propensities, metrics) as JSON. The candidate join (searches × items-in-city) is the scale hotspot; the catalog side is broadcast. On a cluster, submit the same module with `spark-submit`.

## Use as a library

Every stage is a plain function over DataFrames, so each runs standalone on your own log:

```python
import spark_search_ranking as ssr

log = spark.read.parquet("impressions/")     # or ssr.generate_log(spark) to try it out

props = ssr.estimate_position_bias(log.filter("day < 24"))   # explore slice only
weighted = ssr.add_ips_weights(log, props, clip=0.05)
feats = ssr.build_features(weighted, ssr.item_statistics(weighted.filter("day < 24")))
model = ssr.train_reranker(feats.filter("day < 24"), weighted=True)
scored = ssr.score(model, feats.filter("day >= 24"), "score")
print(ssr.ndcg_at_k(scored, "score", k=10))  # needs a ground-truth `grade` column
```

Expected columns per stage (one row = one impression):

| Stage | Needs |
|---|---|
| `estimate_position_bias` | `is_explore`, `position`, `clicked` |
| `add_ips_weights` | `position`, `clicked` |
| `item_statistics` | `item_id`, `clicked`, `booked` |
| `build_features` | `search_id`, `user_id`, `item_id`, `ts`, `price`, `clicked` |
| `train_reranker` / `score` | the `FEATURE_COLS` produced by `build_features` (plus `clicked` and `ips_weight` to train) |
| `ndcg_at_k` / `mrr_top_grade` | `search_id`, `item_id`, `grade`, and the score column |

The whole comparison pipeline is also callable: `ssr.run(spark, ssr.PipelineConfig(searches=50_000))` returns config, split sizes, estimated propensities, and metrics as one dict.

## Example results

60,000 searches, 8,000 items, k = 20, η = 0.65, seed 7 — 1.2M impressions (960,940 train / 239,060 eval), 3.6 minutes on a 4-core CI runner (the `workflow_dispatch` demo job; reproduce with the command it runs):

| Ordering on held-out days | NDCG@10 | MRR |
|---|---|---|
| Production ranker (logged) | 0.5284 | 0.4861 |
| Item-CTR popularity | 0.8769 | 0.8448 |
| GBT, unweighted clicks | 0.8686 | 0.8291 |
| GBT, IPS-weighted clicks | 0.8712 | 0.8336 |

The estimated propensities track the generator's curve closely (position 2: 0.602 vs. true 2^-0.65 = 0.637; position 10: 0.209 vs. 0.224). Two honest observations worth making: the IPS-weighted model consistently edges out the unweighted one, and smoothed item popularity is a very strong baseline here — with a homogeneous population (small per-user affinity), item CTR is nearly an oracle for quality. Increase `affinity_scale` and `eta` to grow the personalization and correction headroom; that sensitivity is itself the point of having a generator with knobs.

## Design notes

- **Why an explore slice?** Without intervention, propensities are not identified from logs alone: high positions get more clicks both because they are examined more *and* because the ranker put better items there. A small randomized slice (industry practice) separates the two.
- **Why weight only positives?** Under the examination hypothesis, a click requires examination, so clicked rows are reweighted by 1/propensity. Unclicked rows are ambiguous (not examined vs. examined and rejected) and keep weight 1 — the standard practical treatment.
- **Why is position not a feature?** The ranker chooses positions at serving time; training on logged position would leak the incumbent ranker's decisions into the model.
- **Evaluation against ground truth, not clicks.** Held-out clicks are themselves position-biased; evaluating on them rewards mimicking the old ranker. The generator's true relevance (graded per search) gives an unbiased target — the luxury a synthetic environment buys.
- **Limitations.** The click model is examination-based (cascade-style browsing, no trust bias); the propensity estimator assumes position-independent relevance in explore traffic (true here by construction); IPS clipping trades a small bias for bounded variance.

## Development

```bash
git clone https://github.com/mohammadi-hadi/spark-search-ranking.git
cd spark-search-ranking
pip install -e ".[dev]"
pytest                                            # full suite, a few minutes on a laptop
```

## Releasing

Bump `__version__` in `src/spark_search_ranking/__init__.py`, commit, then tag and push:

```bash
git tag v0.2.0 && git push origin v0.2.0
```

The [release workflow](.github/workflows/release.yml) builds the sdist and wheel, verifies the tag matches the package version, and attaches both to a GitHub Release. To additionally publish to PyPI, configure [trusted publishing](https://docs.pypi.org/trusted-publishers/) for this repository (environment `pypi`) and set the repository variable `PYPI_PUBLISH` to `true`.

## References

- Joachims, Swaminathan & Schnabel — *Unbiased Learning-to-Rank with Biased Feedback*, WSDM 2017.
- Craswell, Zoeter, Taylor & Ramsey — *An Experimental Comparison of Click Position-Bias Models*, WSDM 2008.
- Wang, Bendersky, Metzler & Najork — *Learning to Rank with Selection Bias in Personal Search*, SIGIR 2016.
- Bernardi, Mavridis & Estevez — *150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com*, KDD 2019.

## License

MIT — see [LICENSE](LICENSE).
