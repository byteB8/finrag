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

- **Answers, searching all 368 filings:** 60.7% correct [52.7, 68.7] with a local 4B model (dense retrieval,
  company and year filters, reranking). On the same 150 questions, the FinanceBench paper's GPT-4-Turbo with
  one vector store over all filings scored 19.3% (it refused 67%). Given the labelled evidence pages directly,
  the 4B model reaches 64.0% (GPT-4-Turbo: 85.3%), so the gain comes from retrieval, not the generator.
- **Retrieval:** the right evidence page is in the top 5 chunks for 66.0% of questions [58.0, 73.3] (top 10:
  80.7%). Reranking is the largest single gain (32.0% -> 58.7%). Naive fusion with BM25, which reaches 10.7%
  on its own here, *lowers* dense retrieval from 52.7% to 32.7%.
- **Abstention:** on 50 unanswerable questions (years moved past the corpus, or companies outside it) the
  system declines 48-50, while answering 77-78% of the real questions at about 70% accuracy. The failures
  that get through: substituting the latest fiscal period, and attributing another company's figure.
- **Judge check:** on 30 answers graded blind by hand, the LLM judge agreed on correct vs not correct for 27
  (Cohen's kappa 0.80); it was slightly stricter (50.0% vs 53.3% correct).
- **Prompt A/B:** a prompt that discourages refusals made no significant difference (+1.3 / +2.0 points, p >= 0.68).
- **Service:** 5.3 s median per question one at a time (generation 4.6 s, reranking 0.7 s), 0.70 questions/s with
  8 concurrent clients, on two RTX A4000s.
- **Engineering findings:** constrained JSON decoding looped (whitespace, the citations array) until bounded;
  vLLM 0.31 ignores the per-request whitespace switch; and compact JSON flipped the model's `answerable` flag
  on 87 of 150 answers until llguidance with natural `", "` / `": "` separators restored it.

LLM-judge numbers are not directly comparable with the paper's human grades; see the judge check. Prompt v2
and the year-filtered configurations were chosen after seeing results on the same 150 questions.

## Service

`serve.sh up` starts three vLLM model servers (generation on one GPU; embedding and reranking on the other,
OpenAI-compatible, localhost only) and a FastAPI app that does retrieval in memory and calls them:

- `POST /ask` `{"question": "...", "company": null}` returns the answer, the cited filings and pages with excerpts,
  an `abstained` flag, citation and number checks, and per-stage timings.
- `POST /ask/stream`: the same, as Server-Sent Events (retrieval, then answer tokens, then the result).
- `GET /health`.

`bench.py` measures latency (sequential and concurrent); see the report's latency section.

## Running it

On a Linux machine with an NVIDIA GPU (16 GB is enough):

```bash
./setup.sh                 # uv, Python 3.11, vLLM, model weights, FinanceBench
./batch.sh                 # prep -> index -> search -> gen -> grade -> abstention -> report
./serve.sh up              # model servers + HTTP API on 127.0.0.1:8010 (/docs for the OpenAPI page)
python bench.py            # API latency, sequential and concurrent
python test.py             # unit tests (no GPU needed)
```

Single stages: `python run.py {prep,index,search,gen,grade,report}`; settings are in `configs/mvp.toml`.

FinanceBench is fetched, not redistributed; cite Islam et al., *FinanceBench* (arXiv:2311.11944).
