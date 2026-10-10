"""tests.py -- Pruebas de correctitud.  Ejecutar:  python tests.py"""
import numpy as np

from images import TYPES, make_image, worst_case_image
from quadtree import (build_quadtree, reconstruct, tree_stats, leaves_of, split_region,
                      serialize, deserialize, storage_bits, max_abs_error, mse)


def test_partition_covers_exactly():
    """Las particiones cubren cada píxel exactamente una vez (sin traslapes ni huecos)."""
    for h, w in [(1, 1), (1, 7), (7, 1), (2, 3), (5, 5), (13, 9), (64, 64), (75, 100)]:
        cover = np.zeros((h, w), int)
        for (y, x, hh, ww) in split_region(0, 0, h, w):
            assert hh > 0 and ww > 0
            assert hh * ww < h * w or h * w == 1  # cada hijo es estrictamente menor
            cover[y:y + hh, x:x + ww] += 1
        if h * w > 1:
            assert (cover == 1).all(), (h, w)


def test_leaves_tile_image():
    """Las hojas forman una teselación exacta de la imagen (también con dims. impares)."""
    for (h, w) in [(33, 17), (100, 75), (64, 64), (1, 20), (20, 1)]:
        img = worst_case_image(h, w)
        root = build_quadtree(img, tol=40)
        cover = np.zeros((h, w), int)
        for l in leaves_of(root):
            cover[l.y:l.y + l.h, l.x:l.x + l.w] += 1
        assert (cover == 1).all(), (h, w)


def test_exact_reconstruction_tol0():
    """Con tol=0 la reconstrucción es idéntica a la original (compresión sin pérdida)."""
    for crit in ("range", "var"):
        for kind in TYPES:
            for (h, w) in [(64, 64), (50, 37)]:
                img = make_image(kind, h, w)
                rec = reconstruct(build_quadtree(img, 0, crit))
                assert np.array_equal(img, rec), (crit, kind, h, w)


def test_error_bound_range():
    """Criterio 'range': error por píxel y canal <= tol; criterio 'var': MSE <= tol^2."""
    for tol in (4, 16, 48):
        for kind in TYPES:
            img = make_image(kind, 96, 80)
            rec = reconstruct(build_quadtree(img, tol, "range"))
            assert max_abs_error(img, rec) <= tol
            rec = reconstruct(build_quadtree(img, tol, "var"))
            assert mse(img, rec) <= tol * tol + 1.0  # +1: redondeo del color medio


def test_uniform_image_single_leaf():
    root = build_quadtree(make_image("uniform", 128), 0)
    s = tree_stats(root)
    assert s == {"nodes": 1, "leaves": 1, "depth": 0}


def test_serialization_roundtrip_and_size():
    for kind in TYPES:
        img = make_image(kind, 64, 48)
        root = build_quadtree(img, 16)
        data = serialize(root, 3)
        root2, ch = deserialize(data)
        assert ch == 3
        assert np.array_equal(reconstruct(root, 3), reconstruct(root2, 3))
        bits = storage_bits(tree_stats(root), 3)
        assert len(data) == (bits + 7) // 8  # tamaño real == tamaño teórico (a bytes)


def test_grayscale():
    img = make_image("regions", 64)[:, :, :1]
    root = build_quadtree(img, 0)
    assert np.array_equal(img, reconstruct(root, 1))


def test_criteria_agree_at_tol0():
    img = make_image("detail", 64)
    assert tree_stats(build_quadtree(img, 0, "range")) == tree_stats(build_quadtree(img, 0, "var"))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} pruebas pasaron.")
