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
| dense+company+rerank | 27.3 [20.0, 34.7] | 62.7 [54.7, 70.7] | 75.3 [68.0, 82.0] | 92.0 [87.3, 96.0] | 0.427 |
| dense+company+year+rerank | 26.7 [19.3, 34.0] | 66.0 [58.0, 73.3] | 80.7 [74.0, 86.7] | 93.3 [89.3, 96.7] | 0.438 |
| scoped-company: hybrid+rerank | 27.3 [20.0, 34.7] | 60.0 [52.0, 68.0] | 74.7 [67.3, 81.3] | 90.0 [85.3, 94.7] | 0.423 |
| scoped-filing: hybrid+rerank | 39.3 [31.3, 47.3] | 76.0 [68.7, 82.7] | 90.0 [84.7, 94.7] | 100.0 [100.0, 100.0] | 0.554 |

Company detection from the question text: exactly right 131, includes the right company 131, none found 19, wrong 0 (of 150).

## Answers

Verdicts from an LLM judge (Qwen3-14B-AWQ) against the analysts' reference answers. The judge files most answers that say the information is missing under *incorrect*, so the *says info missing* column counts them from the answer text. *Numbers not in cited text* counts answers with any number (in the answer or its reasoning) that appears as written in no cited excerpt; computed values (ratios, sums, unit conversions) count too, so it is a coarse upper bound on ungrounded numbers, not a hallucination rate.

| run | correct | incorrect | refusal | says info missing | valid citations | with numbers not in cited text |
|---|---|---|---|---|---|---|
| labelled evidence pages (oracle), prompt v1, hybrid + company + rerank | 64.0 [56.0, 71.3] | 28.0 [20.7, 35.3] | 8.0 [4.0, 12.7] | 28 | 142/150 | 89 |
| labelled evidence pages (oracle), prompt v2, hybrid + company + rerank | 66.0 [58.0, 73.3] | 31.3 [24.0, 38.7] | 2.7 [0.7, 5.3] | 20 | 148/150 | 90 |
| retrieved chunks, prompt v1, hybrid + company + rerank | 58.0 [50.0, 66.0] | 37.3 [30.0, 45.3] | 4.7 [1.3, 8.0] | 34 | 144/150 | 101 |
| retrieved chunks, prompt v1, dense + company + year + rerank | 60.7 [52.7, 68.7] | 33.3 [26.0, 40.7] | 6.0 [2.7, 10.0] | 33 | 145/150 | 103 |
| retrieved chunks, prompt v1, hybrid + company + year + rerank | 59.3 [51.3, 66.7] | 36.0 [28.7, 44.0] | 4.7 [1.3, 8.7] | 31 | 143/150 | 100 |
| retrieved chunks, prompt v1, before the decoding-loop fixes | 56.7 [48.7, 64.7] | 36.7 [29.3, 44.7] | 6.7 [2.7, 10.7] | 45 | 141/150 | 101 |
| retrieved chunks, prompt v2, hybrid + company + rerank | 59.3 [51.3, 67.3] | 37.3 [30.0, 45.3] | 3.3 [0.7, 6.7] | 25 | 148/150 | 104 |

## Latency (HTTP API)

`serve.sh up` + `bench.py`: 150 questions, one at a time, then 8 concurrent clients. Generation on one RTX A4000; embedding and reranking share the other.

| stage | p50 ms | p95 ms |
|---|---|---|
| embed | 34.0 | 39.3 |
| search | 26.6 | 34.9 |
| rerank | 701.8 | 784.4 |
| generate | 4580.1 | 6871.9 |
| total | 5334.4 | 7658.5 |

Concurrent: 0.7 questions/s; wall time p50 10296.9 ms, p95 17900.6 ms.

## Abstention

Retrieved context. *Declined* = the answer's `answerable` flag is false (rule *flag*) or its text says the information is missing (rule *text*). Unanswerable questions: 50 FinanceBench questions with every year moved 10 years later or the company replaced by one outside the corpus. Selective accuracy = correct among answered questions.

| run | rule | coverage (answerable) | correct overall | selective accuracy | declined on unanswerable |
|---|---|---|---|---|---|
| prompt v1, hybrid + company + rerank | flag | 77.3 | 58.0 | 69.8 | 50/50 (100%) |
| prompt v1, hybrid + company + rerank | text | 77.3 | 58.0 | 70.7 | 49/50 (98%) |
| prompt v1, dense + company + year + rerank | flag | 78.0 | 60.7 | 70.1 | 48/50 (96%) |
| prompt v1, dense + company + year + rerank | text | 78.0 | 60.7 | 70.1 | 46/50 (92%) |
| prompt v1, hybrid + company + year + rerank | flag | 79.3 | 59.3 | 66.4 | 50/50 (100%) |
| prompt v1, hybrid + company + year + rerank | text | 79.3 | 59.3 | 66.4 | 49/50 (98%) |
| prompt v2, hybrid + company + rerank | flag | 68.0 | 59.3 | 69.6 | 48/50 (96%) |
| prompt v2, hybrid + company + rerank | text | 83.3 | 59.3 | 66.4 | 41/50 (82%) |

## Prompt A/B (v2 - v1, same questions)

Prompt v2 was written after reading v1's errors on these same questions (see the notebook), so it is tuned on the test set.

| context | v1 correct % | v2 correct % | difference [95% CI] | p |
|---|---|---|---|---|
| labelled evidence pages (oracle) | 64.0 | 66.0 | +2.0 [-5.3, +9.3] | 0.68 |
| retrieved chunks | 58.0 | 59.3 | +1.3 [-4.7, +7.3] | 0.74 |

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

