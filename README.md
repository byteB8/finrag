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

## Running it

On a Linux machine with an NVIDIA GPU (16 GB is enough):

```bash
./setup.sh                 # uv, Python 3.11, vLLM, model weights, FinanceBench
./batch.sh                 # prep -> index -> search -> gen -> grade -> report
python test.py             # unit tests (no GPU needed)
```

Single stages: `python run.py {prep,index,search,gen,grade,report}`; settings are in `configs/mvp.toml`.

FinanceBench is fetched, not redistributed; cite Islam et al., *FinanceBench* (arXiv:2311.11944).
