"""Check that an arena edit is a valid adapter: same modules as the zero adapter, rank <= 32, alpha = 2 r, finite, non-zero.

Usage: check_adapter.py EDIT_DIR
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ZERO = Path("/path/to/gemma-v2/results/steer/edits/zero")
d = Path(sys.argv[1])
keys = ("target_modules", "peft_type", "base_model_name_or_path", "bias", "use_rslora", "use_dora")
cz = json.loads((ZERO / "adapter_config.json").read_text())
cd = json.loads((d / "adapter_config.json").read_text())
bad = [k for k in keys if cz.get(k) != cd.get(k)]
if bad:
    sys.exit(f"FAIL config differs from the zero adapter: {bad}")
r = cd["r"]
if not (1 <= r <= 32 and cd["lora_alpha"] == 2 * r):
    sys.exit(f"FAIL need 1 <= r <= 32 and lora_alpha == 2 r, got r={r} alpha={cd['lora_alpha']}")
z = load_file(str(ZERO / "adapter_model.safetensors"))
e = load_file(str(d / "adapter_model.safetensors"))
if set(z) != set(e):
    sys.exit(f"FAIL tensor names differ: {len(set(z) ^ set(e))} keys")
changed = 0
big = 0.0
for k, t in e.items():
    want = tuple(r if n == 16 and i == (0 if ".lora_A" in k else 1) else n for i, n in enumerate(z[k].shape))
    if tuple(t.shape) != want or t.dtype != z[k].dtype:
        sys.exit(f"FAIL {k}: {tuple(t.shape)} {t.dtype} vs {want} {z[k].dtype}")
    if not torch.isfinite(t.float()).all():
        sys.exit(f"FAIL {k}: non-finite values")
    if ".lora_B" in k and float(t.float().abs().max()) > 0:
        changed += 1
        big = max(big, float(t.float().abs().max()))
if changed == 0:
    sys.exit("FAIL every lora_B is zero: the edit does nothing")
print(f"OK {d.name}: {changed} modules changed, max |B| {big:.4g}")
