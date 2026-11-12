"""Which MLP neurons does each edit overwrite? Reports neuron indices shared by two or more edits per layer.

    python stack_overlap.py editA editB ...      (run in $S/edits)
A neuron edit writes column j of down_proj (LoRA A row is one-hot at j) and row j of gate_proj / up_proj
(LoRA B column is one-hot at j). Two edits that overwrite the same neuron do not add as two neurons.
"""
import re
import sys
from collections import defaultdict

from safetensors.torch import load_file

names = sys.argv[1:]
touch = defaultdict(lambda: defaultdict(set))       # layer -> edit -> neuron indices
for n in names:
    t = load_file(f"{n}/adapter_model.safetensors")
    for k, v in t.items():
        m = re.search(r"layers\.(\d+)\..*?(gate_proj|up_proj|down_proj)\.lora_([AB])", k)
        if not m:
            continue
        layer, mod, ab = int(m.group(1)), m.group(2), m.group(3)
        v = v.float()
        if mod == "down_proj" and ab == "A":        # rows over neurons
            for row in v:
                if row.abs().max() > 0:
                    idx = (row.abs() > 0.5 * row.abs().max()).nonzero().flatten().tolist()
                    if len(idx) <= 4:
                        touch[layer][n].update(idx)
        if mod != "down_proj" and ab == "B":        # columns over neurons
            for col in v.T:
                if col.abs().max() > 0:
                    idx = (col.abs() > 0.5 * col.abs().max()).nonzero().flatten().tolist()
                    if len(idx) <= 4:
                        touch[layer][n].update(idx)
for layer in sorted(touch):
    d = touch[layer]
    line = " | ".join(f"{n}: {sorted(d[n])}" for n in names if d[n])
    shared = [j for j in set().union(*d.values()) if sum(j in s for s in d.values()) > 1]
    print(f"L{layer} {line}" + (f"  SHARED {sorted(shared)}" if shared else ""))
