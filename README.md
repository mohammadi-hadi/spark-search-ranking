# spark-search-ranking

[![CI](https://github.com/mohammadi-hadi/spark-search-ranking/actions/workflows/ci.yml/badge.svg)](https://github.com/mohammadi-hadi/spark-search-ranking/actions/workflows/ci.yml)

Counterfactual learning-to-rank for marketplace search logs, implemented end to end in PySpark: position-bias estimation from randomized traffic, inverse-propensity-weighted training, and NDCG evaluation against known ground truth.

Clicks in search logs are a biased signal — users click what they *see*, and what they see is what the incumbent ranker put on top. Training the next ranker naively on those clicks bakes the old ranker's mistakes in. This project is a compact, fully tested reference implementation of the standard correction: estimate examination propensities from a randomized "explore" slice, reweight clicks by inverse propensity, and re-rank.

Because the log is synthetic with a known generative process, the pipeline's correctness is *provable* in tests: the propensity estimator must recover the generator's position-bias curve, and the re-ranker must beat the logged production ordering on true relevance.

```mermaid
flowchart LR
    A[synthetic search log<br/>90% exploit / 10% explore] --> B[propensity estimation<br/>explore slice only]
    A --> C[leakage-safe features<br/>cumulative windows]
    B --> D[IPS weights]
    C --> E[GBT re-ranker<br/>weightCol]
    D --> E
    E --> F[NDCG@10 / MRR<br/>vs. ground-truth grades]
```

## What it demonstrates

- **Position-bias correction**: propensities estimated from randomized traffic only (estimating them from exploit traffic confounds examination with relevance — the classic mistake), IPS weighting with variance clipping.
- **Leakage-safe feature engineering**: item statistics from the training window only with cold-start priors; per-user history via cumulative windows over strictly earlier events (`rowsBetween(unboundedPreceding, -1)`).
- **Spark at the core**: no UDFs anywhere — generation, features, and metrics are all column expressions and window functions, so everything scales with the cluster.
- **Testable ML**: a deterministic generator with known ground truth turns "does the correction work?" into unit tests, including a hand-computed NDCG case and a no-leakage assertion.

## Quickstart

```bash
pip install -e ".[dev]"
pytest                                            # full suite, a few minutes on a laptop
python -m spark_search_ranking.pipeline --searches 20000
```

Scale knobs: `--searches`, `--items`, `--users`, `--k` (results per search), `--eta` (position-bias strength), `--explore-frac`. The candidate join (searches × items-in-city) is the scale hotspot; the catalog side is broadcast. On a cluster, submit the same module with `spark-submit`.

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

## References

- Joachims, Swaminathan & Schnabel — *Unbiased Learning-to-Rank with Biased Feedback*, WSDM 2017.
- Craswell, Zoeter, Taylor & Ramsey — *An Experimental Comparison of Click Position-Bias Models*, WSDM 2008.
- Wang, Bendersky, Metzler & Najork — *Learning to Rank with Selection Bias in Personal Search*, SIGIR 2016.
- Bernardi, Mavridis & Estevez — *150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com*, KDD 2019.

## License

MIT — see [LICENSE](LICENSE).
