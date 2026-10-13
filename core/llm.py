"""vLLM offline inference shared by the answer and grading stages: JSON-constrained chat, greedy decoding."""
from __future__ import annotations

import json


# Constrained JSON: free whitespace let greedy decoding loop (newlines/tabs until the token limit), and
# compact JSON ("," / ":", xgrammar's only no-whitespace layout) shifted the model's choices: it flipped a
# boolean field on 87 of 150 answers. llguidance takes separators from the schema's "x-guidance" key, so
# the output is single-line JSON with the usual ", " / ": " spacing (on 30 test questions: flag
# disagreements 17 -> 7, no cut-offs; bounded free whitespace ({0,12}) still cut off 7 of 30). vLLM 0.31 reads disable_any_whitespace only from the engine
# config (the per-request flag is ignored), and the engine flag needs an explicit backend.
NATURAL_JSON = {"whitespace_flexible": False, "item_separator": ", ", "key_separator": ": "}


def make_llm(model: str, max_model_len: int, util: float, thinking: bool = False, **kw):
    """`thinking`: the qwen3 reasoning parser makes the JSON constraint start only after `</think>`."""
    from vllm import LLM
    so = {"backend": "guidance", "disable_any_whitespace": True}
    if thinking:
        so["reasoning_parser"] = "qwen3"
    return LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=util, seed=0,
               enable_prefix_caching=True, structured_outputs_config=so, **kw)


def with_separators(schema: dict) -> dict:
    return {**schema, "x-guidance": NATURAL_JSON}


def json_params(schema: dict, max_tokens: int, temperature: float = 0.0, **sampling):
    """Sampling parameters whose output must match `schema` (the API was renamed across vLLM versions).
    `sampling`: extra SamplingParams fields (top_p, top_k, seed) for the thinking runs."""
    from vllm import SamplingParams
    schema = with_separators(schema)
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              structured_outputs=StructuredOutputsParams(json=schema), **sampling)
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              guided_decoding=GuidedDecodingParams(json=schema), **sampling)


def split_reasoning(text: str) -> tuple[str, str]:
    """(reasoning, rest) for output that may carry a thinking section; the chat template can open <think> in the
    prompt, so the reasoning is everything before the last </think>."""
    if "</think>" not in text:
        return "", text
    head, _, tail = text.rpartition("</think>")
    return head.replace("<think>", "").strip(), tail.strip()


def chat(llm, conversations: list[list[dict]], params, thinking: bool = False) -> list[dict]:
    outs = llm.chat(conversations, params, chat_template_kwargs={"enable_thinking": thinking}, use_tqdm=False)
    return [{"text": o.outputs[0].text, "finish": o.outputs[0].finish_reason,
             "prompt_tokens": len(o.prompt_token_ids), "output_tokens": len(o.outputs[0].token_ids)}
            for o in outs]


def parse_json(text: str) -> dict | None:
    """The constrained output is JSON unless generation was cut off; None then."""
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None
