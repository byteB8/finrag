# Results

FinanceBench open-source sample: 150 questions over 368 filings. Brackets are 95% bootstrap intervals over questions.

## Data

- 368 PDFs, 54,120 pages, 111,561 chunks of at most 512 tokens (64 overlap), never crossing a page.
- Labelled evidence found on its labelled page: 184 of 189 items; on a neighbouring page instead: 0; document missing: 0.

## Retrieval

A hit means a retrieved chunk lies on a labelled evidence page (page) or in the labelled filing (doc).

| configuration | page hit@1 | page hit@5 | page hit@10 | doc hit@5 | MRR |
|---|---|---|---|---|---|
| bm25 | 4.7 [1.3, 8.0] | 10.7 [6.0, 15.3] | 13.3 [8.0, 18.7] | 27.3 [20.7, 34.7] | 0.077 |
| dense | 26.7 [19.3, 34.0] | 52.7 [44.7, 60.7] | 62.7 [54.7, 70.0] | 86.7 [81.3, 92.0] | 0.375 |
| hybrid | 15.3 [10.0, 21.3] | 32.7 [25.3, 40.0] | 46.0 [38.0, 54.0] | 70.7 [63.3, 78.0] | 0.239 |
| hybrid+company | 16.7 [10.7, 22.7] | 32.0 [24.7, 39.3] | 44.0 [36.0, 52.0] | 77.3 [70.7, 84.0] | 0.243 |
| hybrid+company+rerank | 26.7 [20.0, 34.0] | 58.7 [50.7, 66.7] | 72.0 [64.7, 79.3] | 90.0 [85.3, 94.7] | 0.413 |
| hybrid+company+year+rerank | 26.7 [20.0, 34.0] | 64.0 [56.0, 71.3] | 78.0 [71.3, 84.7] | 92.7 [88.0, 96.7] | 0.427 |
| scoped-company: hybrid+rerank | 27.3 [20.0, 34.7] | 60.0 [52.0, 68.0] | 74.7 [67.3, 81.3] | 90.0 [85.3, 94.7] | 0.423 |
| scoped-filing: hybrid+rerank | 39.3 [31.3, 47.3] | 76.0 [68.7, 82.7] | 90.0 [84.7, 94.7] | 100.0 [100.0, 100.0] | 0.554 |

Company detection from the question text: exactly right 131, includes the right company 131, none found 19, wrong 0 (of 150).

## Answers

Verdicts from an LLM judge (Qwen3-14B-AWQ) against the analysts' reference answers. The judge files most answers that say the information is missing under *incorrect*, so the *says info missing* column counts them from the answer text. *Numbers not in cited text* counts answers with any number (in the answer or its reasoning) that appears as written in no cited excerpt; computed values (ratios, sums, unit conversions) count too, so it is a coarse upper bound on ungrounded numbers, not a hallucination rate.

| run | correct | incorrect | refusal | says info missing | valid citations | with numbers not in cited text |
|---|---|---|---|---|---|---|
| labelled evidence pages (oracle), prompt v1 | 64.7 [57.3, 72.0] | 29.3 [22.0, 36.7] | 6.0 [2.7, 10.0] | 30 | 143/150 | 83 |
| labelled evidence pages (oracle), prompt v2 | 67.3 [60.0, 74.7] | 28.7 [21.3, 36.0] | 4.0 [1.3, 7.3] | 18 | 147/150 | 87 |
| retrieved chunks, prompt v1 | 57.3 [49.3, 65.3] | 36.0 [28.7, 44.0] | 6.7 [3.3, 10.7] | 36 | 144/150 | 98 |
| retrieved chunks, prompt v1, before the decoding-loop fixes | 56.7 [48.7, 64.7] | 36.7 [29.3, 44.7] | 6.7 [2.7, 10.7] | 45 | 141/150 | 101 |
| retrieved chunks, prompt v2 | 54.7 [46.7, 62.7] | 42.0 [34.0, 50.0] | 3.3 [0.7, 6.7] | 22 | 150/150 | 105 |

## Prompt A/B (v2 - v1, same questions)

Prompt v2 was written after reading v1's errors on these same questions (see the notebook), so it is tuned on the test set.

| context | v1 correct % | v2 correct % | difference [95% CI] | p |
|---|---|---|---|---|
| labelled evidence pages (oracle) | 64.7 | 67.3 | +2.7 [-3.3, +8.7] | 0.46 |
| retrieved chunks | 57.3 | 54.7 | -2.7 [-9.3, +4.0] | 0.48 |

## Judge check

On 30 randomly sampled answers (retrieved chunks, prompt v1, before the decoding-loop fixes), graded blind by hand, the judge agreed on correct vs not correct for 27 (Cohen's kappa 0.8); on all three verdicts for 25 (kappa 0.686; the hand grades marked refusals as incorrect). Share graded correct on that sample: by hand 53.3%, by the judge 50.0%.

## FinanceBench paper baselines

Human-graded runs released with the benchmark (same 150 questions).

| model | setting | correct | incorrect | refusal |
|---|---|---|---|---|
| gpt-4-1106-preview | labelled evidence pages given | 85.3 | 14.7 | 0.0 |
| gpt-4-1106-preview | one index over all filings | 19.3 | 13.3 | 67.3 |
| gpt-4-1106-preview | the question's filing only | 50.0 | 11.3 | 38.7 |
| gpt-4 | labelled evidence pages given | 84.0 | 10.0 | 6.0 |
| gpt-4 | one index over all filings | 16.7 | 8.7 | 74.7 |
| gpt-4 | the question's filing only | 42.0 | 10.7 | 47.3 |
| llama2 | one index over all filings | 19.3 | 69.3 | 11.3 |
| llama2 | the question's filing only | 41.3 | 54.0 | 4.7 |

