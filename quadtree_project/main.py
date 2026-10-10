"""main.py -- Interfaz de consola.

Uso:
  python main.py compress  imagen.png --tol 16 [--criterion range|var] [--gray] [--out salida/]
  python main.py decompress salida/imagen.qt --out reconstruida.png
"""
import argparse
import os
import time

import numpy as np

from experiments import draw_partition
from images import load_image, save_image
from quadtree import (build_quadtree, reconstruct, tree_stats, serialize, deserialize,
                      storage_bits, mse, psnr, max_abs_error, compression_ratio, original_bits)


def cmd_compress(a):
    img = load_image(a.image, a.gray)
    h, w, c = img.shape
    t0 = time.perf_counter()
    root = build_quadtree(img, a.tol, a.criterion)
    dt = (time.perf_counter() - t0) * 1000
    st = tree_stats(root)
    rec = reconstruct(root, c)
    data = serialize(root, c)
    bits = storage_bits(st, c)
    m = mse(img, rec)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(a.image))[0]
    with open(os.path.join(a.out, base + ".qt"), "wb") as f:
        f.write(data)
    save_image(rec, os.path.join(a.out, base + "_reconstruida.png"))
    save_image(draw_partition(rec, root) if c == 3 else rec, os.path.join(a.out, base + "_particion.png"))
    print(f"Imagen: {w}x{h}, {c} canal(es), {h*w} píxeles")
    print(f"Tolerancia: {a.tol} (criterio {a.criterion})")
    print(f"Nodos: {st['nodes']} | hojas: {st['leaves']} | profundidad máx.: {st['depth']}")
    print(f"Tiempo de construcción: {dt:.2f} ms")
    print(f"Tamaño original: {original_bits(h, w, c)//8} B | comprimido: {len(data)} B")
    print(f"Razón de compresión: {compression_ratio(h, w, c, bits):.3f}")
    print(f"MSE: {m:.4f} | PSNR: {psnr(m):.2f} dB | error máx.: {max_abs_error(img, rec)}")
    print(f"Archivos escritos en: {a.out}")


def cmd_decompress(a):
    with open(a.file, "rb") as f:
        root, c = deserialize(f.read())
    save_image(reconstruct(root, c), a.out)
    print("Imagen reconstruida guardada en", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("compress")
    c.add_argument("image"); c.add_argument("--tol", type=float, default=16)
    c.add_argument("--criterion", choices=["range", "var"], default="range")
    c.add_argument("--gray", action="store_true"); c.add_argument("--out", default="salida")
    c.set_defaults(fn=cmd_compress)
    d = sp.add_parser("decompress")
    d.add_argument("file"); d.add_argument("--out", default="reconstruida.png")
    d.set_defaults(fn=cmd_decompress)
    args = p.parse_args()
    args.fn(args)
