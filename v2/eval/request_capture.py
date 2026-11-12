"""Exact Per-Call LLM Request Capture for Rollouts (Train/Inference Context Parity by Construction).

swegemma `SessionTrace` files omit ADK `EventsCompactionConfig` summaries, so any LLM call made after a compaction
(token_threshold=14336 on Kaggle) cannot be re-rendered from the trace. This module wraps
`google.adk.models.lite_llm.LiteLlm.generate_content_async` and appends, for every call, the exact LiteLLM
`messages` + `tools` ADK sent and the assistant message it got back to a per-rollout JSONL file:

  {"messages": [...], "tools": [...], "response_message": {...}, "finish_reason": "STOP", "is_agent_call": true,
   "prompt_token_ids": [...], "completion_token_ids": [...], "completion_logprobs": [...]}

Token-in/token-out: while a log is active, `LiteLLMClient.acompletion` also asks vLLM for `return_token_ids` and
per-token `logprobs`, so each record carries the exact served prompt ids, the sampled ids and their sampling
logprobs (the old-policy logprobs for RL). Sampling itself is unchanged. `collect_and_prepare_rft` prefers these
request logs over trace reconstruction when they exist.
"""

from __future__ import annotations

import contextvars
import functools
import json
import os
from pathlib import Path
from typing import Any

CURRENT_REQUEST_LOG: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "swegemma_request_log", default=None
)
# Per-call mutable holder the acompletion wrapper fills with token ids / logprobs (None = capture off)
_TOKEN_HOLDER: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "swegemma_token_holder", default=None
)
# Async RL with several rollout replicas: an async callable returning the server base URL (no "/v1") for the next call
# of the current rollout (v2.eval.async_protocol.Router.base_for bound to its run_key); None = the configured api_base
CURRENT_ROUTE: contextvars.ContextVar[Any] = contextvars.ContextVar("swegemma_route", default=None)


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    val = getattr(obj, key, None)
    if val is None and hasattr(obj, "model_dump"):
        val = obj.model_dump().get(key)
    return val


def _choice_tokens(choice: Any) -> tuple[list[int], list[float]]:
    psf = _get(choice, "provider_specific_fields") or {}
    ids = list(_get(choice, "token_ids") or psf.get("token_ids") or [])
    lp_obj = _get(choice, "logprobs")
    content = (_get(lp_obj, "content") if lp_obj is not None else None) or []
    lps = [float(_get(c, "logprob")) for c in content]
    return ids, lps


def read_policy_pointer() -> str:
    """Current policy adapter name from the file named by V2_POLICY_POINTER ("" when unset or unreadable).

    Read once per model call, so each call is sampled by exactly one version (versions switch at call boundaries)."""
    pointer = os.environ.get("V2_POLICY_POINTER")
    if not pointer:
        return ""
    try:
        return Path(pointer).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def route_to_policy(model: Any, version: str) -> str:
    """LiteLLM model id `<provider>/<served name>` -> `<provider>/<version>` (the hot-loaded adapter name).

    The served name may itself contain "/" (a base-model alias served under its checkpoint path, e.g.
    `openai//home/.../gemma-4-e4b-it-qat-w4a16-ct`), so only the provider prefix is kept."""
    provider, sep, _ = str(model).partition("/")
    return f"{provider}/{version}" if sep else version


def _install_token_capture(lite_llm: Any) -> None:
    client_cls = lite_llm.LiteLLMClient
    orig = client_cls.acompletion
    if getattr(orig, "_v2_token_capture", False):
        return

    @functools.wraps(orig)
    async def _acompletion(self: Any, model: Any, messages: Any, tools: Any, **kwargs: Any) -> Any:
        holder = _TOKEN_HOLDER.get()
        if holder is None:
            return await orig(self, model, messages, tools, **kwargs)
        # Async RL: route every call to the newest policy adapter named in the pointer file (the trainer hot-loads
        # each version into vLLM under a fresh name), and record which version served the call.
        version = read_policy_pointer()
        if version:
            model = route_to_policy(model, version)
            holder["policy_version"] = version
        route = CURRENT_ROUTE.get()
        if route is not None:  # replica routing: same engine closure on every server, so only throughput changes
            base = str(await route()).rstrip("/")
            kwargs["api_base"] = base + "/v1"
            holder["server"] = base
        kwargs["logprobs"] = True
        kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), "return_token_ids": True}
        resp = await orig(self, model, messages, tools, **kwargs)
        if not kwargs.get("stream"):
            holder["prompt_token_ids"] = list(_get(resp, "prompt_token_ids") or [])
            choices = _get(resp, "choices") or []
            if choices:
                holder["completion_token_ids"], holder["completion_logprobs"] = _choice_tokens(choices[0])
            return resp

        async def _stream() -> Any:
            ids: list[int] = []
            lps: list[float] = []
            async for chunk in resp:
                if "prompt_token_ids" not in holder and _get(chunk, "prompt_token_ids"):
                    holder["prompt_token_ids"] = list(_get(chunk, "prompt_token_ids"))
                choices = _get(chunk, "choices") or []
                if choices:
                    c_ids, c_lps = _choice_tokens(choices[0])
                    ids.extend(c_ids)
                    lps.extend(c_lps)
                yield chunk
            holder["completion_token_ids"], holder["completion_logprobs"] = ids, lps

        return _stream()

    _acompletion._v2_token_capture = True  # type: ignore[attr-defined]
    client_cls.acompletion = _acompletion


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return _jsonable(obj.model_dump(exclude_none=True))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def install_request_capture() -> bool:
    """Idempotently wrap `LiteLlm.generate_content_async`; returns True if newly installed."""
    from google.adk.models import lite_llm

    _install_token_capture(lite_llm)
    cls = lite_llm.LiteLlm
    orig = cls.generate_content_async
    if getattr(orig, "_v2_request_capture", False):
        return False

    @functools.wraps(orig)
    async def _captured(self: Any, llm_request: Any, stream: bool = False) -> Any:
        log_path = CURRENT_REQUEST_LOG.get()
        if log_path is None:
            async for resp in orig(self, llm_request, stream=stream):
                yield resp
            return

        effective_model = getattr(llm_request, "model", None) or self.model
        messages: Any = None
        tools: Any = None
        try:
            inputs = await lite_llm._get_completion_inputs(llm_request, effective_model)
            messages, tools = inputs[0], inputs[1]
        except Exception as exc:  # capture must never break a rollout
            messages = {"capture_error": f"{type(exc).__name__}: {exc}"}

        final_resp: Any = None
        holder: dict[str, Any] = {}
        token = _TOKEN_HOLDER.set(holder)
        try:
            async for resp in orig(self, llm_request, stream=stream):
                if not getattr(resp, "partial", False):
                    final_resp = resp
                yield resp
        finally:
            try:
                _TOKEN_HOLDER.reset(token)
            except ValueError:  # generator closed from another context; the holder is per-call anyway
                pass

        response_message: Any = None
        finish_reason: Any = None
        if final_resp is not None:
            finish_reason = getattr(final_resp, "finish_reason", None)
            content = getattr(final_resp, "content", None)
            if content is not None and getattr(content, "parts", None):
                try:
                    response_message = await lite_llm._content_to_message_param(content, model=effective_model)
                except Exception as exc:
                    response_message = {"capture_error": f"{type(exc).__name__}: {exc}"}
        record = {
            "messages": _jsonable(messages),
            "tools": _jsonable(tools),
            "response_message": _jsonable(response_message),
            "finish_reason": str(getattr(finish_reason, "name", finish_reason)) if finish_reason is not None else None,
            "is_agent_call": bool(tools),
            "prompt_token_ids": holder.get("prompt_token_ids"),
            "completion_token_ids": holder.get("completion_token_ids"),
            "completion_logprobs": holder.get("completion_logprobs"),
            "policy_version": holder.get("policy_version"),
        }
        if holder.get("server"):
            record["server"] = holder["server"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    _captured._v2_request_capture = True  # type: ignore[attr-defined]
    cls.generate_content_async = _captured
    return True
