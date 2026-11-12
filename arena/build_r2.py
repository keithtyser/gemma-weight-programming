"""Round-2 edits by author `claude`: dose, depth and stacking variants of the before-first-edit gate (xp_pre).

    python build_r2.py --edit NAME        writes ../edits/NAME/   (same closed-form construction as build.py)
    python build_r2.py --merge NAME       layer-disjoint union of two existing adapters (see MERGES)
"""
import json, sys
from pathlib import Path
from safetensors.torch import load_file, save_file
import build

PRE_ON, PRE_OFF = dict(tags=build.PRE + ["bg@prompt"]), dict(tags=build.POST)
EARLY_ON, EARLY_OFF = dict(tags=build.PRE, call_max=4), [dict(tags=build.PRE, call_min=8), dict(tags=build.POST)]


def pre(t):
    return dict(write=("edit@0", -1.0), target=t, on=PRE_ON, off=PRE_OFF)


def early(t):
    return dict(write=("edit@0", -1.0), target=t, on=EARLY_ON, off=EARLY_OFF)


build.EDITS.update({
    "pre_d50":      dict(layers=[18, 20, 22], neurons=[pre(0.5)]),
    "pre_d100":     dict(layers=[18, 20, 22], neurons=[pre(1.0)]),
    "pre_d200":     dict(layers=[18, 20, 22], neurons=[pre(2.0)]),
    "pre_L5":       dict(layers=[16, 18, 20, 22, 24], neurons=[pre(0.25)]),
    "pre_L5_d50":   dict(layers=[16, 18, 20, 22, 24], neurons=[pre(0.5)]),
    "pre_early":    dict(layers=[18, 20, 22], neurons=[pre(0.25), early(0.3)]),
    "early_d60":    dict(layers=[18, 20, 22], neurons=[early(0.6)]),
})
E = Path("/path/to/gemma-v2/results/steer/edits")
HERE = Path(__file__).resolve().parent.parent / "edits"
# name: (adapter that supplies every layer by default, adapter that supplies the listed layers, layers)
MERGES = {
    "pre_concise":     (HERE / "xp_pre", E / "am_damped_dual_explore_concise", [24, 26, 28]),
    "pre_d50_concise": (HERE / "pre_d50", E / "am_damped_dual_explore_concise", [24, 26, 28]),
}


def merge(name):
    a, b, layers = MERGES[name]
    ta, tb = load_file(str(a / "adapter_model.safetensors")), load_file(str(b / "adapter_model.safetensors"))
    assert ta.keys() == tb.keys()
    out, n = {}, 0
    for k, v in ta.items():
        li = int(k.split(".layers.")[1].split(".")[0])
        if li in layers:
            assert ".lora_B" not in k or float(v.float().abs().max()) == 0.0, k   # first adapter is silent there
            v = tb[k]
            n += int(float(v.float().abs().max()) > 0)
        out[k] = v
    od = HERE / name
    od.mkdir(parents=True, exist_ok=True)
    (od / "adapter_config.json").write_text((a / "adapter_config.json").read_text())
    save_file(out, str(od / "adapter_model.safetensors"))
    (od / "report.json").write_text(json.dumps({"edit": name, "merge_of": [str(a), str(b)], "layers_from_second": layers,
                                                "nonzero_tensors_from_second": n}, indent=1))
    print("wrote", od, "nonzero tensors from second:", n)


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge(sys.argv[sys.argv.index("--merge") + 1])
    else:
        build.main()
