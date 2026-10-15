import json, pathlib
d = json.loads(pathlib.Path("results/sensitivity_v2/param_sensitivity_mmvp_qwen3b.json").read_text())
R = d["results"]
HELD = {"lambda_sem": 2.0, "lambda_bg": 0.2, "lambda_sys": 0.3,
        "tau": 0.2, "k": 0.2, "sigma": 20.0, "layers": [6, 31]}
SYM = {"lambda_sem": (r"$\lambda_{\mathrm{sem}}$", "amplify relevant"),
       "lambda_bg":  (r"$\lambda_{\mathrm{bg}}$",  "attenuate background"),
       "lambda_sys": (r"$\lambda_{\mathrm{sys}}$", "suppress system prompt"),
       "tau":        (r"$\tau$",                   "presence threshold"),
       "k":          (r"$k$",                      "head fraction per layer"),
       "sigma":      (r"$\sigma$",                 "foveal blur (px)"),
       "layers":     (r"$[\ell_s,\ell_e]$",        "decoder layer band")}
ORDER = ["lambda_sem", "lambda_bg", "lambda_sys", "tau", "k", "sigma", "layers"]
NCOL = max(len(R[p]) for p in ORDER)

def fmt_v(p, v):
    if p == "layers":
        return r"$[%d,%d]$" % (int(v[0]), int(v[1]))
    f = float(v)
    return ("%g" % f)

lines = []
for p in ORDER:
    rows = R[p]
    sym, role = SYM[p]
    vals, accs = [], []
    for r in rows:
        v, a = r["value"], 100 * r["pair_acc"]
        held = (list(v) == HELD[p]) if p == "layers" else (float(v) == float(HELD[p]))
        vs, as_ = fmt_v(p, v), "%.1f" % a
        if held:
            vs = (r"$\mathbf{[%d,%d]}$" % (int(v[0]), int(v[1]))) if p == "layers" \
                 else (r"\textbf{%s}" % vs)
        vals.append(vs)
        accs.append(r"\textbf{%s}" % as_ if held else as_)
    pad = [""] * (NCOL - len(rows))
    spread = max(r["pair_acc"] for r in rows) - min(r["pair_acc"] for r in rows)
    lines.append(r"\multirow{2}{*}{%s} & \multirow{2}{*}{%s} & value & %s & \\" %
                 (sym, role, " & ".join(vals + pad)))
    lines.append(r" & & pair acc. & %s & %.2f \\" % (" & ".join(accs + pad), 100 * spread))

body = "\n\\addlinespace\n".join("\n".join(lines[i:i+2]) for i in range(0, len(lines), 2))


HDR = (r"\toprule" "\n"
       r"\textbf{Param} & \textbf{Role} & & \multicolumn{NC}{c}{\textbf{Swept grid}} & \textbf{Spread} \\" "\n"
       r"\cmidrule(lr){4-A} \cmidrule(lr){B-B}")
HDR = HDR.replace("NC", str(NCOL)).replace("A", str(3 + NCOL)).replace("B", str(4 + NCOL))

PRE = r"""% Auto-generated from results/sensitivity_v2/param_sensitivity_mmvp_qwen3b.json
% Source: srf/param_sensitivity.py --anchor current  (2026-09-21). Do not hand-edit.
% Anchor: lambda_sem=2.0 lambda_bg=0.2 lambda_sys=0.3 tau=0.20 k=0.20 sigma=20 layers=[6,31]
%         head_mode=ratio_topk, 78 active slots, MMVP pair 45.33 / img 70.00
\begin{table}[t]
\centering
\caption{Parameter sensitivity of SRF on MMVP with Qwen2.5-VL-3B-Instruct. Each
parameter is swept on its own while every other parameter is held at the
configuration used for our reported result, so each block isolates a single
parameter. The value used in our configuration is shown in bold. Image accuracy
follows pair accuracy closely and is omitted for compactness. MMVP contains 150
pairs, so one pair corresponds to $0.67$ percentage points and differences below
roughly two pairs should not be read as meaningful. The unmodified baseline
scores $40.0$.}
\label{tab:param_sensitivity_mmvp}
\resizebox{\linewidth}{!}{%
\begin{tabular}{l l l cccccc c}
"""
POST = r"""
\bottomrule
\end{tabular}}
\end{table}
"""
tex = PRE + HDR + "\n" + body + POST
pathlib.Path("/volumes2/mllm/PAPER/ICLR/tables/param_sensitivity_mmvp.tex").write_text(tex)
print(tex)
