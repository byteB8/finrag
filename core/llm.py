"""vLLM offline inference shared by the answer and grading stages: JSON-constrained chat, greedy decoding."""
from __future__ import annotations

import json


def make_llm(model: str, max_model_len: int, util: float, **kw):
    from vllm import LLM
    return LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=util, seed=0,
               enable_prefix_caching=True, **kw)


def json_params(schema: dict, max_tokens: int, temperature: float = 0.0):
    """Sampling parameters whose output must match `schema` (the API was renamed across vLLM versions)."""
    from vllm import SamplingParams
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              structured_outputs=StructuredOutputsParams(json=schema))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              guided_decoding=GuidedDecodingParams(json=schema))


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
