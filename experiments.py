"""experiments.py -- Experimentos reproducibles.  Ejecutar:  python experiments.py

Genera en ./results:
  results.csv            malla principal (3 tamaños x 3 tolerancias x 3 contenidos)
  extra_results.csv      dimensiones no potencia de 2, criterio 'var' y escalamiento
  plots/*.png            gráficas
  evidence/*.png, .md    3 ejemplos (original, reconstruida, partición, métricas)
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from images import TYPES, make_image, worst_case_image, save_image
from quadtree import (build_quadtree, reconstruct, tree_stats, leaves_of, storage_bits,
                      serialize, mse, psnr, max_abs_error, compression_ratio, original_bits)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
os.makedirs(os.path.join(OUT, "evidence"), exist_ok=True)

SIZES = (64, 128, 256)
TOLS = (0, 16, 48)          # 0 = uniformidad exacta
REPEATS = 5                 # el tiempo reportado es la mediana de REPEATS corridas


def timed(fn, repeats=REPEATS):
    ts, res = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn()
        ts.append(time.perf_counter() - t0)
    return res, float(np.median(ts)) * 1000.0  # ms


def run_one(img, name, kind, tol, criterion="range"):
    h, w, c = img.shape
    root, t_build = timed(lambda: build_quadtree(img, tol, criterion))
    stats, t_walk = timed(lambda: tree_stats(root))
    rec, t_rec = timed(lambda: reconstruct(root, c))
    bits = storage_bits(stats, c)
    nbytes = len(serialize(root, c))
    assert (bits + 7) // 8 == nbytes
    m = mse(img, rec)
    return dict(
        image=name, content=kind, height=h, width=w, pixels=h * w, tol=tol, criterion=criterion,
        build_ms=round(t_build, 3), traverse_ms=round(t_walk, 3), reconstruct_ms=round(t_rec, 3),
        nodes=stats["nodes"], leaves=stats["leaves"], max_depth=stats["depth"],
        storage_bits=bits, storage_bytes=nbytes, original_bytes=original_bits(h, w, c) // 8,
        compression_ratio=round(compression_ratio(h, w, c, bits), 4),
        space_saving_pct=round(100 * (1 - bits / original_bits(h, w, c)), 2),
        mse=round(m, 4), psnr_db=(round(psnr(m), 3) if m > 0 else float("inf")),
        max_abs_error=max_abs_error(img, rec),
    ), root, rec


def main():
    # ---------------- Malla principal ----------------
    rows = []
    for kind in TYPES:
        for n in SIZES:
            img = make_image(kind, n)
            for tol in TOLS:
                r, _, _ = run_one(img, f"{kind}_{n}x{n}", kind, tol)
                rows.append(r)
                print(f"{r['image']:>18} tol={tol:<3} nodes={r['nodes']:<7} CR={r['compression_ratio']:<8} "
                      f"MSE={r['mse']:<9} build={r['build_ms']} ms")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)

    # ---------------- Experimentos extra ----------------
    extra = []
    # (a) dimensiones que no son potencia de 2
    for kind in TYPES:
        for (h, w) in [(100, 75), (127, 129)]:
            img = make_image(kind, h, w)
            for tol in TOLS:
                r, _, _ = run_one(img, f"{kind}_{h}x{w}", kind, tol)
                r["experiment"] = "non_power_of_two"
                extra.append(r)
    # (b) criterio 'var' (imágenes integrales) vs 'range' en 'detail'
    for n in SIZES:
        img = make_image("detail", n)
        for tol in TOLS:
            for crit in ("range", "var"):
                r, _, _ = run_one(img, f"detail_{n}x{n}", "detail", tol, crit)
                r["experiment"] = "criterion_comparison"
                extra.append(r)
    # (b2) regiones planas con ruido leve: efecto de la tolerancia sobre contenido "casi uniforme"
    for n in SIZES:
        img = make_image("noisy_regions", n)
        for tol in (0, 4, 8, 16, 48):
            r, _, _ = run_one(img, f"noisy_regions_{n}x{n}", "noisy_regions", tol)
            r["experiment"] = "noisy_regions"
            extra.append(r)
    # (c) escalamiento en el peor caso (ruido, tol=0) y mejor caso (uniforme)
    for n in (16, 32, 64, 128, 256, 512):
        for label, img in (("worst_noise", worst_case_image(n)), ("best_uniform", make_image("uniform", n))):
            for crit in ("range", "var"):
                r, _, _ = run_one(img, f"{label}_{n}x{n}", label, 0, crit)
                r["experiment"] = "scaling"
                extra.append(r)
    edf = pd.DataFrame(extra)
    edf.to_csv(os.path.join(OUT, "extra_results.csv"), index=False)

    make_plots(df, edf)
    make_evidence()
    print("\nListo. Resultados en", OUT)


# ------------------------------------------------------------------ gráficas
def make_plots(df, edf):
    colors = {"uniform": "tab:green", "regions": "tab:blue", "detail": "tab:red"}
    marks = {0: "o", 16: "s", 48: "^"}

    # 1) tiempo de construcción vs tamaño
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for i, tol in enumerate(TOLS):
        for kind in TYPES:
            d = df[(df.tol == tol) & (df.content == kind)].sort_values("height")
            ax[i].plot(d.height, d.build_ms, marker="o", color=colors[kind], label=kind)
        ax[i].set(xscale="log", yscale="log", xlabel="lado n (px)", title=f"tol = {tol}")
        ax[i].set_xticks(SIZES); ax[i].set_xticklabels(SIZES)
        ax[i].grid(alpha=.3)
    ax[0].set_ylabel("tiempo de construcción (ms)"); ax[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "plots", "time_vs_size.png"), dpi=150); plt.close(fig)

    # 2) hojas vs tolerancia
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for i, n in enumerate(SIZES):
        for kind in TYPES:
            d = df[(df.height == n) & (df.content == kind)].sort_values("tol")
            ax[i].plot(d.tol, d.leaves, marker="o", color=colors[kind], label=kind)
        ax[i].set(yscale="log", xlabel="tolerancia", title=f"{n}x{n}")
        ax[i].set_xticks(TOLS); ax[i].grid(alpha=.3)
    ax[0].set_ylabel("nº de hojas"); ax[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "plots", "leaves_vs_tol.png"), dpi=150); plt.close(fig)

    # 3) razón de compresión y MSE vs tolerancia
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for kind in TYPES:
        for n, ls in zip(SIZES, ("-", "--", ":")):
            d = df[(df.height == n) & (df.content == kind)].sort_values("tol")
            ax[0].plot(d.tol, d.compression_ratio, ls, marker="o", color=colors[kind],
                       label=f"{kind} {n}")
            ax[1].plot(d.tol, d.mse.clip(lower=1e-3), ls, marker="o", color=colors[kind])
    ax[0].axhline(1, color="k", lw=.8); ax[0].set(yscale="log", xlabel="tolerancia", ylabel="razón de compresión")
    ax[1].set(yscale="log", xlabel="tolerancia", ylabel="MSE (0 se muestra como 1e-3)")
    ax[0].legend(fontsize=6, ncol=2); [a.grid(alpha=.3) for a in ax]
    ax[0].set_xticks(TOLS); ax[1].set_xticks(TOLS)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "plots", "ratio_mse_vs_tol.png"), dpi=150); plt.close(fig)

    # 4) compromiso tasa-distorsión: CR vs PSNR
    fig, ax = plt.subplots(figsize=(5, 3.4))
    fin = df[np.isfinite(df.psnr_db)]
    for kind in TYPES:
        d = fin[fin.content == kind]
        ax.scatter(d.psnr_db, d.compression_ratio, color=colors[kind], label=kind)
    ax.axhline(1, color="k", lw=.8); ax.set(yscale="log", xlabel="PSNR (dB)", ylabel="razón de compresión")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "plots", "rate_distortion.png"), dpi=150); plt.close(fig)

    # 5) escalamiento: peor caso con 'range' vs 'var'
    sc = edf[edf.experiment == "scaling"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for crit, st in (("range", "-"), ("var", "--")):
        d = sc[(sc.content == "worst_noise") & (sc.criterion == crit)].sort_values("height")
        ax[0].plot(d.height, d.build_ms, st, marker="o", label=f"peor caso (ruido), {crit}")
        d = sc[(sc.content == "best_uniform") & (sc.criterion == crit)].sort_values("height")
        ax[0].plot(d.height, d.build_ms, st, marker="s", label=f"mejor caso (uniforme), {crit}")
    n = np.array(sorted(sc.height.unique()), float)
    ref = sc[(sc.content == "worst_noise") & (sc.criterion == "range") & (sc.height == 64)].build_ms.iloc[0]
    ax[0].plot(n, ref * (n / 64) ** 2 * np.log2(n) / np.log2(64), "k:", lw=.8, label=r"$\propto n^2\log n$")
    ax[0].plot(n, ref * (n / 64) ** 2, "k-.", lw=.8, label=r"$\propto n^2$")
    ax[0].set(xscale="log", yscale="log", xlabel="lado n (px)", ylabel="construcción (ms)")
    ax[0].legend(fontsize=6); ax[0].grid(alpha=.3)
    w = sc[(sc.content == "worst_noise") & (sc.criterion == "range")].sort_values("height")
    ax[1].plot(w.height, w.nodes / w.pixels, marker="o")
    ax[1].axhline(4 / 3, color="k", ls=":", lw=.8); ax[1].set_ylim(1.30, 1.36)
    ax[1].set(xscale="log", xlabel="lado n (px)", ylabel="nodos / píxeles (peor caso)")
    ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "plots", "scaling.png"), dpi=150); plt.close(fig)


# ------------------------------------------------------------------ evidencias
def draw_partition(img, root):
    out = img.copy()
    for l in leaves_of(root):
        out[l.y, l.x:l.x + l.w] = (0, 0, 0)
        out[l.y:l.y + l.h, l.x] = (0, 0, 0)
    return out


def make_evidence():
    cases = [("regions", 256, 16), ("detail", 256, 48), ("detail", 128, 0)]
    md = ["# Evidencias de funcionamiento\n"]
    for i, (kind, n, tol) in enumerate(cases, 1):
        img = make_image(kind, n)
        r, root, rec = run_one(img, f"{kind}_{n}x{n}", kind, tol)
        tag = f"ejemplo{i}_{kind}_{n}_tol{tol}"
        for suffix, a in (("original", img), ("reconstruida", rec), ("particion", draw_partition(rec, root))):
            save_image(a, os.path.join(OUT, "evidence", f"{tag}_{suffix}.png"))
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
        for a, t, im in zip(ax, ("Original", "Reconstruida", "Partición Quadtree"), (img, rec, draw_partition(rec, root))):
            a.imshow(im, interpolation="nearest"); a.set_title(t, fontsize=9); a.axis("off")
        fig.suptitle(f"{kind} {n}x{n}, tol={tol} | hojas={r['leaves']} | CR={r['compression_ratio']:.2f} | "
                     f"MSE={r['mse']:.2f} | PSNR={r['psnr_db']} dB", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "evidence", f"{tag}.png"), dpi=130); plt.close(fig)
        md.append(f"## Ejemplo {i}: `{kind}` {n}x{n}, tol={tol}\n")
        md.append(f"![{tag}]({tag}.png)\n")
        md.append("| Métrica | Valor |\n|---|---|")
        for k in ("height", "width", "tol", "criterion", "nodes", "leaves", "max_depth", "build_ms",
                  "storage_bytes", "original_bytes", "compression_ratio", "space_saving_pct",
                  "mse", "psnr_db", "max_abs_error"):
            md.append(f"| {k} | {r[k]} |")
        md.append("")
    with open(os.path.join(OUT, "evidence", "EVIDENCIAS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()
