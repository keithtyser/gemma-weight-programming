"""Cheap free-running degeneration probe for a LoRA: does greedy decoding fall into repetition loops?

  python -m v2.eval.lora_loop_probe --model-path M --adapter-dir A --request-logs R*.jsonl --out OUT \
      [--scales 1.0 0.5 0.25] [--n-prompts 48] [--max-tokens 1536]

Builds served contexts from captured request logs (one per log, mid-trajectory), writes copies of the adapter with
`lora_B` scaled by each `--scales` value, and greedy-decodes every context with the base model and each scaled
adapter in one vLLM process. Reports per arm: share of outputs that hit `max_tokens` (the golden symptom), share
with heavy line repetition, and mean output tokens. Not a decision metric (that is golden_v1); a fast filter.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import shutil
import sys
from typing import Any


def line_repeat_fraction(text: str) -> float:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 6:
        return 0.0
    counts = collections.Counter(lines)
    return sum(v - 1 for v in counts.values() if v > 1) / len(lines)


def write_scaled_adapter(src: Path, dst: Path, scale: float) -> None:
    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "adapter_config.json", dst / "adapter_config.json")
    sd = load_file(str(src / "adapter_model.safetensors"))
    save_file({k: (v * scale if ".lora_B." in k else v).contiguous() for k, v in sd.items()},
              str(dst / "adapter_model.safetensors"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.eval.lora_loop_probe")
    p.add_argument("--model-path", required=True)
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--request-logs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    p.add_argument("--n-prompts", type=int, default=48)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--max-prompt-tokens", type=int, default=14000)
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    from v2.eval.train_serve_parity import build_prompt_ids

    args.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    prompts = build_prompt_ids(tok, args.request_logs, args.n_prompts, args.max_prompt_tokens)
    arms: dict[str, LoRARequest | None] = {"base": None}
    for i, s in enumerate(args.scales, start=1):
        d = args.out / f"adapter_x{s:g}"
        write_scaled_adapter(args.adapter_dir, d, s)
        arms[f"lora_x{s:g}"] = LoRARequest(f"x{s:g}", i, str(d.resolve()))

    llm = LLM(model=args.model_path, dtype="bfloat16", max_model_len=args.max_prompt_tokens + args.max_tokens + 64,
              gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=64, max_loras=len(args.scales), seed=0,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    tp = [TokensPrompt(prompt_token_ids=x) for x in prompts]
    report: dict[str, Any] = {"n_prompts": len(prompts), "max_tokens": args.max_tokens, "arms": {}}
    samples: dict[str, list[str]] = {}
    for name, lr in arms.items():
        outs = llm.generate(tp, sp, lora_request=lr) if lr else llm.generate(tp, sp)
        texts = [o.outputs[0].text for o in outs]
        hit_cap = [o.outputs[0].finish_reason == "length" for o in outs]
        reps = [line_repeat_fraction(t) for t in texts]
        report["arms"][name] = {
            "hit_max_tokens": round(sum(hit_cap) / len(outs), 3),
            "heavy_repeat": round(sum(r > 0.5 for r in reps) / len(outs), 3),
            "mean_tokens": round(sum(len(o.outputs[0].token_ids) for o in outs) / len(outs), 1),
        }
        samples[name] = [t[:600] for t in texts]
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
