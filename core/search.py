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


# ---------------------------------------------------------------- dense encoder and reranker (GPU)

class Encoder:
    """Qwen3-Embedding: last-token pooling (left padding), L2-normalised; queries carry an instruction."""

    def __init__(self, name: str, max_len: int = 1024, batch: int = 32):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.model = AutoModel.from_pretrained(name, dtype=torch.float16).cuda().eval()
        self.max_len, self.batch = max_len, batch
        ids = self.tok("a")["input_ids"]
        eos = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.suffix = "" if ids[-1] == eos else "<|endoftext|>"   # the pooled token must be the end-of-text token

    def encode(self, texts: list[str], instruction: str | None = None, log_every: int = 0) -> np.ndarray:
        torch = self.torch
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery:{t}" for t in texts]
        texts = [t + self.suffix for t in texts]
        order = np.argsort([-len(t) for t in texts], kind="stable")   # longest first: a too-big batch fails early
        out = None
        for b, i in enumerate(range(0, len(texts), self.batch)):
            idx = order[i:i + self.batch]
            enc = self.tok([texts[j] for j in idx], padding=True, truncation=True, max_length=self.max_len,
                           return_tensors="pt").to("cuda")
            with torch.inference_mode():
                h = self.model(**enc).last_hidden_state[:, -1]
            emb = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy().astype(np.float16)
            if out is None:
                out = np.zeros((len(texts), emb.shape[1]), dtype=np.float16)
            out[idx] = emb
            if log_every and b % log_every == 0:
                print(f"  encoded {min(i + self.batch, len(texts))}/{len(texts)}", flush=True)
        return out


class Reranker:
    """Qwen3-Reranker: P("yes") for an instruction / query / document prompt, read from the last position."""

    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
              "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, name: str, instruction: str, max_len: int = 2048, batch: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).cuda().eval()
        self.instruction, self.max_len, self.batch = instruction, max_len, batch
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suf = self.tok.encode(self.SUFFIX, add_special_tokens=False)

    def score(self, query: str, docs: list[str]) -> list[float]:
        torch = self.torch
        pairs = [f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {d}" for d in docs]
        out: list[float] = []
        for i in range(0, len(pairs), self.batch):
            enc = self.tok(pairs[i:i + self.batch], padding=False, truncation="longest_first",
                           return_attention_mask=False, max_length=self.max_len - len(self.pre) - len(self.suf))
            ids = [self.pre + x + self.suf for x in enc["input_ids"]]
            batch = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :]
            pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            out.extend(torch.log_softmax(pair, dim=1)[:, 1].exp().tolist())
        return out
