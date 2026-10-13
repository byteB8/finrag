"""Retrieval: BM25, dense embeddings, reciprocal-rank fusion, a cross-encoder reranker, metadata filters."""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- pure helpers (unit-tested without a GPU)

def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal-rank fusion: score(d) = sum over rankings of 1 / (k + rank of d)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))


def top_k(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[int]:
    """Indices of the k highest scores, best first, among rows where `mask` is true."""
    s = np.asarray(scores, dtype=np.float64)
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-s, k - 1)[:k]
    return idx[np.argsort(-s[idx], kind="stable")].tolist()


# ---------------------------------------------------------------- BM25

class BM25:
    def __init__(self, retriever, stemmer):
        self.r, self.stemmer = retriever, stemmer

    @staticmethod
    def _stemmer():
        import Stemmer
        return Stemmer.Stemmer("english")

    @classmethod
    def build(cls, texts: list[str]) -> "BM25":
        import bm25s
        stemmer = cls._stemmer()
        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)
        r = bm25s.BM25()
        r.index(tokens, show_progress=False)
        return cls(r, stemmer)

    def save(self, path: Path) -> None:
        self.r.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "BM25":
        import bm25s
        return cls(bm25s.BM25.load(str(path)), cls._stemmer())

    def scores(self, query: str) -> np.ndarray:
        import bm25s
        q = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, return_ids=False, show_progress=False)
        return np.asarray(self.r.get_scores(q[0]), dtype=np.float32)


# ---------------------------------------------------------------- embedding and reranking on vLLM (GPU)
# Both models run through vLLM's pooling runner, which packs sequences without padding; a plain
# transformers forward pass over the same 111k chunks ran at about a quarter of the GPU's throughput.

EOT = "<|endoftext|>"   # Qwen3-Embedding pools the hidden state of this final token


class Embedder:
    """L2-normalised Qwen3-Embedding vectors (float16). Queries carry an instruction; documents do not.
    One engine per process: a second vLLM engine in the same process can hang it at exit."""

    def __init__(self, name: str, util: float, max_tokens: int = 2048):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tok = AutoTokenizer.from_pretrained(name)
        self.eot = self.tok.convert_tokens_to_ids(EOT)
        self.max_tokens = max_tokens
        self.llm = LLM(model=name, runner="pooling", dtype="float16", gpu_memory_utilization=util,
                       max_model_len=max_tokens, seed=0)

    def __call__(self, texts: list[str], instruction: str | None = None, slice_size: int = 8192,
                 log=None) -> np.ndarray:
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery:{t}" for t in texts]
        parts = []
        for i in range(0, len(texts), slice_size):
            ids = self.tok(texts[i:i + slice_size], add_special_tokens=False)["input_ids"]
            prompts = [{"prompt_token_ids": x[: self.max_tokens - 1] + [self.eot]} for x in ids]
            out = self.llm.embed(prompts, use_tqdm=False)
            v = np.asarray([r.outputs.embedding for r in out], dtype=np.float32)
            parts.append((v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float16))
            if log:
                log(f"embedded {min(i + slice_size, len(texts))}/{len(texts)}")
        return np.concatenate(parts)


RERANK_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
                 "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def rerank_scores(name: str, pairs: list[tuple[str, str]], instruction: str, util: float,
                  max_tokens: int = 2048) -> list[float]:
    """P("yes") from Qwen3-Reranker for each (query, document) pair, loaded as a two-token classifier
    (the "no"/"yes" rows of its LM head), as in the model card's vLLM recipe."""
    from vllm import LLM

    llm = LLM(model=name, runner="pooling", dtype="float16", gpu_memory_utilization=util,
              max_model_len=max_tokens, seed=0,
              hf_overrides={"architectures": ["Qwen3ForSequenceClassification"],
                            "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": True})
    queries = [f"{RERANK_PREFIX}<Instruct>: {instruction}\n<Query>: {q}\n" for q, _ in pairs]
    docs = [f"<Document>: {d}{RERANK_SUFFIX}" for _, d in pairs]
    return [r.outputs.score for r in llm.score(queries, docs, use_tqdm=False)]
