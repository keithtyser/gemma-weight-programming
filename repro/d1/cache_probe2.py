import json, sys, time, warnings, inspect
import torch
warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)
import transformers
from transformers import StaticCache
from transformers.utils.quantization_config import CompressedTensorsConfig
M = "/path/to/gemma-v2/models/gemma-4-e4b-it-qat-w4a16-ct"
print(inspect.signature(StaticCache.__init__))
model = transformers.AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, quantization_config=CompressedTensorsConfig(run_compressed=False)).cuda().eval()
pt = json.loads(open("/path/to/gemma-v2/d1/points.jsonl").readline())
B = 2
ids = torch.tensor(pt["prompt"], device="cuda")[None, :].repeat(B, 1)
P = ids.shape[1]
res = {}
for kind in ("dynamic",):
    try:
        past = None if kind == "dynamic" else StaticCache(config=model.config, max_cache_len=P + 64)
        t = time.time()
        for c in range(0, P, 2048):
            out = model(input_ids=ids[:, c:c + 2048], past_key_values=past, use_cache=True)
            past = out.past_key_values
        torch.cuda.synchronize(); t1 = time.time()
        seq = []
        lg = out.logits[:, -1].float()
        import cProfile, pstats
        pr = cProfile.Profile(); pr.enable()
        for _ in range(40):
            nxt = lg.argmax(-1); seq.append(nxt.tolist())
            out = model(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
            past = out.past_key_values
            lg = out.logits[:, -1].float()
        pr.disable(); pstats.Stats(pr).sort_stats("tottime").print_stats(14)
        torch.cuda.synchronize()
        res[kind] = seq
        print(kind, type(past).__name__, "prefill", round(t1 - t, 2), "ms/step", round((time.time() - t1) / 40 * 1000, 1), torch.cuda.max_memory_allocated() >> 20, flush=True)
        try:
            print([type(l).__name__ for l in past.layers][:8], len(past.layers))
        except Exception as e:
            print("nolayers", e)
        del past, out
    except Exception as e:
        print(kind, "FAIL", type(e).__name__, str(e)[:600], flush=True)
if len(res) == 2:
    print("same tokens", res["dynamic"] == res["static"])
