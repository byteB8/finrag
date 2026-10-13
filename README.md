# finrag

Cited-answer question answering over company filings, measured on
[FinanceBench](https://github.com/patronus-ai/financebench): 150 analyst-written questions over 368 SEC
filings, each with a reference answer and the page that holds the evidence.

The pipeline:

1. **Parse and chunk** every PDF page (PyMuPDF), into chunks of at most 512 tokens that never cross a page,
   each with a header naming the company, filing and page. A check confirms that the labelled evidence
   occurs on its labelled page.
2. **Retrieve** with BM25 and dense embeddings (Qwen3-Embedding-0.6B), fused by reciprocal rank, optionally
   restricted to the company named in the question (ticker / alias dictionary), and reranked by a
   cross-encoder (Qwen3-Reranker-0.6B).
3. **Answer** with a local model (Qwen3.5-4B through vLLM) in a fixed JSON format: reasoning, answer, cited
   excerpt numbers, and an `answerable` flag. Each answer is checked for valid citations and for numbers that
   appear in no cited excerpt.
4. **Grade** each answer against the reference with an LLM judge (Qwen3-14B-AWQ): correct / incorrect /
   refusal, as in the FinanceBench paper, whose human-graded runs are the baselines.

Results, with 95% bootstrap intervals over questions: [`results/RESULTS.md`](results/RESULTS.md).

## Results at a glance

- **Answers, searching all 368 filings:** 57.3% correct [49.3, 65.3] with a local 4B model. On the same
  150 questions, the FinanceBench paper's GPT-4-Turbo with one vector store over all filings answered
  19.3% correctly (it refused 67%). The gap comes from retrieval: given the labelled evidence pages
  directly, the 4B model reaches 64.7%, against GPT-4-Turbo's 85.3%.
- **Retrieval:** the right evidence page is among the top 5 chunks for 64.0% of questions [56.0, 71.3] (top 10:
  78.0%) when the search is limited to the company and year detected in the question and reranked. Dense
  retrieval alone gets 52.7%. Naive fusion with BM25 *drops* it to 32.7%, because BM25 is weak on this corpus
  (10.7%). The reranker is the largest single gain (32.0% -> 58.7%). The answer runs used the configuration
  without the year filter (top 10: 72.0%), fixed before these retrieval results were seen.
- **Judge check:** on 30 answers graded blind by hand, the LLM judge agreed on correct vs not correct for 27
  (Cohen's kappa 0.80); it was slightly stricter (50.0% vs 53.3% correct).
- **Prompt A/B:** a second prompt that discourages unnecessary refusals made no significant difference
  (+2.7 points with the evidence pages, -2.7 with retrieval; p ~ 0.5). When retrieval misses the evidence,
  fewer refusals mean more wrong answers.
- **Engineering findings:** constrained JSON decoding looped on whitespace and on the citations array until
  bounded. Disallowing whitespace then flipped the `answerable` flag on 87 of 150 answers (prompt v2), so
  refusals are measured from the answer text and the judge, not from the flag.

LLM-judge numbers are not directly comparable with the paper's human grades; see the judge check.

## Running it

On a Linux machine with an NVIDIA GPU (16 GB is enough):

```bash
./setup.sh                 # uv, Python 3.11, vLLM, model weights, FinanceBench
./batch.sh                 # prep -> index -> search -> gen -> grade -> report
python test.py             # unit tests (no GPU needed)
```

Single stages: `python run.py {prep,index,search,gen,grade,report}`; settings are in `configs/mvp.toml`.

FinanceBench is fetched, not redistributed; cite Islam et al., *FinanceBench* (arXiv:2311.11944).
