import json, sys, time, warnings, cProfile, pstats, io
import torch
warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)
import transformers
from transformers.utils.quantization_config import CompressedTensorsConfig
M = "/path/to/gemma-v2/models/gemma-4-e4b-it-qat-w4a16-ct"
model = transformers.AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, quantization_config=CompressedTensorsConfig(run_compressed=False)).cuda().eval()
pt = json.loads(open("/path/to/gemma-v2/d1/points.jsonl").readline())
B = int(sys.argv[1])
ids = torch.tensor(pt["prompt"], device="cuda")[None, :].repeat(B, 1)
P = ids.shape[1]
past = None
for c in range(0, P, 2048):
    out = model(input_ids=ids[:, c:c + 2048], past_key_values=past, use_cache=True)
    past = out.past_key_values
lg = out.logits[:, -1].float()
torch.cuda.synchronize()
def steps(n):
    global past, lg
    for _ in range(n):
        nxt = lg.argmax(-1)
        out = model(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
        past = out.past_key_values
        lg = out.logits[:, -1].float()
    torch.cuda.synchronize()
steps(3)
t = time.time(); steps(20); print("B", B, "P", P, "ms/step", round((time.time() - t) / 20 * 1000, 1), flush=True)
pr = cProfile.Profile(); pr.enable(); steps(20); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(22); print(s.getvalue()[:5000])
