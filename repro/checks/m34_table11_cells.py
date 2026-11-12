"""m34: Table 11 (tab:cost) cells from src/cost_per_arm_b21_b28.json (scripts/cost_per_arm.py). Run from repro/."""
import json
d = json.load(open("src/cost_per_arm_b21_b28.json"))["summary"]
for a in ["ac_xp_pre", "ac_xp_early", "ag_explore_then_rotate", "am_damped_dual_explore_concise",
          "ac_xp_const_ctrl", "ag_think_start_explore_ctrl", "am_damped_dual_explore_concise_ctrl"]:
    v = d[a]; tc, ds = v["tool_calls"], v["duration_seconds"]
    print(f"{a}: batches={len(v['batches'])} calls {tc['arm_mean']} vs {tc['base_mean']} p={tc['p_batch_t']} | seconds {ds['arm_mean']} vs {ds['base_mean']} p={ds['p_batch_t']}")
