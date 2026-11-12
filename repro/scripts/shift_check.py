"""Predicted (Eq. 3) versus realised layer-output shift along the target direction, in class gaps, for every
edit x layer whose build report records both.

    python -I scripts/shift_check.py <dir>     ->  src/shift_check.json
<dir> holds read-only copies of the build reports from evalhost:/path/to/gemma-v2/wsp_arena/
    claude/edits/xp_*/report.json, gpt/edits/*/build_report.json, gemini/edits/*/build_report.json
Pure python.

Eq. 3:  Delta_l / gap_l = s_l * beta / sqrt(1 + beta^2) * sqrt(n) * eta_l / gap_l = K_l * beta / sqrt(1 + beta^2).
GPT and Gemini set beta from Eq. 3 for a target (dose) in class gaps, uncapped in every layer here, so the
Eq. 3 prediction equals the target. Claude's builder set the write size by bisection on the exact recomputed shift
(not by Eq. 3), so the Eq. 3 prediction for Claude's edits is computed here from the builder's recorded beta_on
and K_l. K_l = t / (beta_t / sqrt(1 + beta_t^2)) is taken from Table tab:params (app_a.tex: the first builder's
Eq. 3 betas for the same direction d_l, same class gap and same layer; capped entry excluded); Claude's report
has the same gaps (9.375, 11.827, 8.365) and direction AUCs (0.773, 0.813, 0.833) as that table.
"""
import json, math, statistics as st, sys

D = sys.argv[1]
OUT_PATH = "src/shift_check.json"
f = lambda b: b / math.sqrt(1 + b * b)  # noqa: E731

# Table tab:params (app_a.tex): beta at strengths 0.125, 0.25, 0.5, 1.0; layer 22 at 1.0 is the cap (0.6), excluded
TAB = {"18": [0.029, 0.057, 0.115, 0.234], "20": [0.030, 0.060, 0.121, 0.248], "22": [0.077, 0.154, 0.320, None]}
T = [0.125, 0.25, 0.5, 1.0]
K = {}
for li, bs in TAB.items():
    ks = [t / f(b) for t, b in zip(T, bs) if b is not None]
    K[li] = {"from_table_each_strength": [round(k, 4) for k in ks], "K": st.mean(ks)}

rows = []


def add(author, edit, layer, kind, positions, predicted, realised, beta, note="", fire=None):
    r = {"author": author, "edit": edit, "layer": int(layer), "kind": kind, "positions": positions, "beta": round(beta, 5),
         "predicted_gaps": round(predicted, 4), "realised_gaps": round(realised, 4), "ratio": round(realised / predicted, 4)}
    if fire is not None:
        r["fire_over_reference"] = round(fire, 4)
        r["ratio_per_unit_fire"] = round(realised / predicted / fire, 4)
    if note:
        r["note"] = note
    rows.append(r)


# ---------------------------------------------------------------- Claude (bisection builder; Eq. 3 from beta_on)
for e in ["xp_const", "xp_pre", "xp_think", "xp_early"]:
    r = json.load(open(f"{D}/claude/edits/{e}/report.json", encoding="utf-8"))
    target = r["spec"]["neurons"][0]["target"]
    for li, L in r["layers"].items():
        n = L["neurons"][0]
        pred = K[li]["K"] * f(n["beta_on"])
        kind = "always-on" if n["mode"] == "const" else "gated"
        add("claude", e, li, kind, "held-out ON positions (fp32 exact recompute; bisection target %.3f)" % target,
            pred, n["shift_on_heldout"], n["beta_on"], note="realised bf16 at decision positions edit@0: %.4f" %
            -L["verified_bf16_shift_over_gap"]["along_edit@0_dir"]["edit@0"])

# ---------------------------------------------------------------- GPT (Eq. 3 builder, neuron edits)
for e in ["think_start_explore", "error_recheck"]:
    r = json.load(open(f"{D}/gpt/edits/{e}/build_report.json", encoding="utf-8"))
    for li, L in r["layers"].items():
        for kind, I in L.items():
            if not isinstance(I, dict) or I.get("dose_target_class_gaps") is None:
                continue
            assert I["beta"] < 0.35  # uncapped -> Eq. 3 prediction == dose
            real = -I["local_bf16"]["heldout|event_positive"]["shift_over_class_gap"]  # write is minus d
            add("gpt", e, li, "gated", "held-out gate-positive positions (bf16)", I["dose_target_class_gaps"], real, I["beta"],
                fire=I["event_fire"]["heldout_positive"])

# ---------------------------------------------------------------- Gemini (Eq. 3 builder with damped write; eta = projection)
r = json.load(open(f"{D}/gemini/edits/damped_dual_explore_concise/build_report.json", encoding="utf-8"))
LS = {"18": 0.53515625, "20": 0.439453125, "22": 0.1728515625}  # layer scalars from Claude's reports (same model)
check = {}
for li, L in r["layers"].items():
    assert L["beta"] < 0.25
    real = (L["realized_shift_over_gap_ho_pos"] + L["realized_shift_over_gap_ho_neg"]) / 2
    if li in LS:  # recompute Eq. 3 from the recorded beta and eta to confirm prediction == target
        check[li] = round(LS[li] * math.sqrt(2560) * L["eff_proj_main"] / L["gap"] * f(L["beta"]), 4)
    add("gemini", "damped_dual_explore_concise", li, "always-on", "held-out decision positions, mean of both classes (bf16)",
        L["target_gap_frac"], real, L["beta"], note="background positions: %.4f" % L["realized_shift_over_gap_ho_bg"])
r = json.load(open(f"{D}/gemini/edits/hybrid_explore_verify_guard/build_report.json", encoding="utf-8"))
for li, L in r["layers"].items():
    assert L["beta"] < 0.25
    if L["stage"] == "pre_edit_explore":
        add("gemini", "hybrid_explore_verify_guard", li, "always-on", "held-out decision positions, mean of both classes (bf16)",
            0.13, (L["realized_shift_over_gap_ho_pos"] + L["realized_shift_over_gap_ho_neg"]) / 2, L["beta"])
    else:
        add("gemini", "hybrid_explore_verify_guard", li, "gated", "held-out gate-positive positions (unverified submit; bf16)",
            0.10, L["realized_shift_over_gap_ho_unv_sub"], L["beta"], fire=L["ho_fire_unv_sub"])
r = json.load(open(f"{D}/gemini/edits/whitened_cond_rushed_read/build_report.json", encoding="utf-8"))
for li, L in r["layers"].items():
    assert L["beta"] < 0.25
    add("gemini", "whitened_cond_rushed_read", li, "gated", "held-out gate-positive positions (rushed; bf16)", 0.18,
        L["realized_shift_over_gap_ho_rush"], L["beta"], fire=L["ho_fire_rush"])


def summ(rs, key="ratio"):
    v = [x[key] for x in rs if key in x]
    return {"n": len(v), "mean": round(st.mean(v), 3), "min": round(min(v), 3), "max": round(max(v), 3)} if v else None


S = {"all": summ(rows)}
for a in ["claude", "gpt", "gemini"]:
    S[a] = summ([x for x in rows if x["author"] == a])
for k in ["always-on", "gated"]:
    S[k] = summ([x for x in rows if x["kind"] == k])
S["gated_per_unit_fire"] = summ(rows, "ratio_per_unit_fire")
S["all_except_gemini_hybrid_gated"] = summ([x for x in rows if not (x["edit"] == "hybrid_explore_verify_guard" and x["kind"] == "gated")])

# not tabulated: no Eq. 3 prediction exists
excluded = {
    "gpt decision_rotation, explore_then_rotate rotation layers 12/16/20": "rank-2 rotation of the down projection, sized by a linearised derivative "
        "toward -0.2 gaps with the angle capped at 0.15 rad (cap hit at all three layers); class gap not recorded, so the capped prediction "
        "cannot be recomputed. Realised at held-out edit@te: -0.012, -0.101, -0.119 gaps.",
    "gemini oproj_onesided_commit_damper": "attention output projection attenuation (gamma 0.14-0.16); no Eq. 3 prediction. Realised at held-out "
        "positive positions -0.11 to -0.16 gaps.",
    "controls (claude xp_const_ctrl, gpt think_start_explore_ctrl, gemini damped_dual_explore_concise_ctrl)": "write orthogonal/random to d_l, "
        "predicted shift 0. Realised: claude 0.0016/0.0040/0.0074; gpt +0.015/-0.019/+0.009; gemini |x| <= 0.0015.",
    "first builder's always-on explore-more edits (bneg_t*, Table tab:params)": "Eq. 3 prediction = t gaps by construction, but their build "
        "reports with realised held-out shifts are not in src/ (remote_diag.json has no build-report fields) nor under the two permitted "
        "evalhost paths; not tabulated. Claude's xp_const reproduces the same edit (same d_l, neurons 3359/2972 at layers 18/20) at t=0.125.",
}
out = {"eq3": "Delta_l/gap = K_l * beta/sqrt(1+beta^2), K_l = s_l sqrt(n) eta_l / gap_l, n = 2560",
       "K_from_table_params": K, "gemini_eq3_recomputed_from_beta_eta_layers_18_20_22": check,
       "source_reports": "evalhost:/path/to/gemma-v2/wsp_arena/{claude/edits/xp_*/report.json, gpt/edits/*/build_report.json, gemini/edits/*/build_report.json}",
       "rows": rows, "summary_ratio_realised_over_predicted": S, "excluded": excluded}
json.dump(out, open(OUT_PATH, "w", encoding="utf-8"), indent=1)
print(json.dumps(K))
print("gemini check", check)
for x in rows:
    print(f'{x["author"]:7s} {x["edit"]:30s} L{x["layer"]:2d} {x["kind"]:9s} beta {x["beta"]:.4f} pred {x["predicted_gaps"]:.3f} real {x["realised_gaps"]:.4f} '
          f'ratio {x["ratio"]:.3f}' + (f' fire {x["fire_over_reference"]:.3f} r/fire {x["ratio_per_unit_fire"]:.3f}' if "fire_over_reference" in x else ""))
print(json.dumps(S, indent=0))
