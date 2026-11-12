"""M2 / s19: eta, natural-output shrinkage and layer-22 saturation from the first builder's build reports.

Inputs: read-only copies (scp) of evalhost:/path/to/gemma-v2/results/steer/edits/bneg_t{0.125,0.25,0.5,1.0}/neuron_report.json
in ../results/edit_reports/bneg_t*/.   Run: python -I m2_eta.py  (prints to stdout; output saved as m2_eta_output.txt)
"""
import json, math, os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "edit_reports")
n = 2560
for t in ["0.125", "0.25", "0.5", "1.0"]:
    r = json.load(open(os.path.join(D, f"bneg_t{t}", "neuron_report.json")))
    print(f"strength {t}  n_train {r['n_train']}  ridge {r['args']['ridge']}  beta cap {r['args']['beta']}")
    for li, L in r["layers"].items():
        b = L["beta"]
        shrink_exact = 1 - 1 / math.sqrt(1 + b * b)
        kappa = float(t) * L["gap"] / (L["layer_scalar"] * math.sqrt(n) * L["write_eff"])
        print(f"  L{li} neuron {L['neuron']} s_l {L['layer_scalar']:.4f} gap {L['gap']:.3f} rho {L['rho']:.4f} "
              f"eta(write_eff) {L['write_eff']:.4f} realised {L['write_eff_realised']:.4f} beta {b:.4f} "
              f"natural_write_scale {L['natural_write_scale']:.4f}  1-1/sqrt(1+b^2) {shrink_exact*100:.2f}%  b^2/2 {b*b/2*100:.2f}%  "
              f"kappa {kappa:.4f}  pred_shift/gap {L['pred_shift_over_gap']:.4f}  q mean/sd {L['const_heldout_mean']:.3f}/{L['const_heldout_sd']:.3f}")
# saturation of layer 22: kappa = t * gap / (s sqrt(n) eta) = 1  ->  t_sat; beta cap 0.6 -> kappa_cap = 0.6/sqrt(1.36)
r = json.load(open(os.path.join(D, "bneg_t0.25", "neuron_report.json")))
for li, L in r["layers"].items():
    k1 = L["gap"] / (L["layer_scalar"] * math.sqrt(n) * L["write_eff"])   # kappa per unit strength
    kcap = 0.6 / math.sqrt(1 + 0.36)
    print(f"L{li}: kappa per unit strength {k1:.4f}; kappa=1 at t={1/k1:.3f}; beta cap 0.6 reached at t={kcap/k1:.3f}")
