"""images.py -- Carga de imágenes y generación reproducible de imágenes de prueba."""
from __future__ import annotations

import numpy as np
from PIL import Image

TYPES = ("uniform", "regions", "detail")


def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    """Carga una imagen desde disco como matriz (H, W, C) uint8."""
    im = Image.open(path).convert("L" if grayscale else "RGB")
    a = np.asarray(im, dtype=np.uint8)
    return a[:, :, None] if a.ndim == 2 else a


def save_image(arr: np.ndarray, path: str) -> None:
    a = arr[:, :, 0] if arr.shape[2] == 1 else arr
    Image.fromarray(a).save(path)


def make_image(kind: str, h: int, w: int | None = None, seed: int = 0) -> np.ndarray:
    """Genera una imagen RGB sintética determinista.

    uniform : un solo color.
    regions : bloques de colores planos definidos (rectángulos y un círculo).
    detail  : degradado suave + formas + ruido gaussiano (variaciones y detalles).
    """
    w = h if w is None else w
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]

    if kind == "uniform":
        img = np.empty((h, w, 3), np.uint8)
        img[:] = (90, 140, 200)
        return img

    if kind == "regions":
        img = np.empty((h, w, 3), np.uint8)
        img[:] = (240, 240, 240)                                   # fondo
        img[: h // 2, : w // 3] = (200, 40, 40)                    # rect. rojo
        img[h // 4: 3 * h // 4, w // 2: 7 * w // 8] = (30, 90, 200)  # rect. azul
        img[int(h * 0.7):, : w // 4] = (40, 160, 70)               # rect. verde
        cy, cx, r = int(h * 0.72), int(w * 0.62), max(2, min(h, w) // 6)
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = (250, 200, 20)  # círculo
        return img

    if kind == "noisy_regions":
        # mismas regiones que 'regions' + ruido gaussiano leve (sigma=2): como un escaneo/foto
        base = make_image("regions", h, w).astype(np.float64)
        base += rng.normal(0, 2, base.shape)
        return np.clip(np.rint(base), 0, 255).astype(np.uint8)

    if kind == "detail":
        base = np.stack([
            60 + 120 * xx / max(w - 1, 1),
            80 + 100 * yy / max(h - 1, 1),
            140 + 60 * np.sin(2 * np.pi * (xx + yy) / max(w, h)),
        ], axis=-1)
        cy, cx, r = h // 2, w // 2, min(h, w) // 4
        base[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = (230, 90, 60)
        base[h // 8: h // 8 + max(2, h // 16), :] = (20, 20, 20)    # línea fina
        base += rng.normal(0, 6, base.shape)                         # ruido
        return np.clip(np.rint(base), 0, 255).astype(np.uint8)

    raise ValueError(f"tipo desconocido: {kind!r}")


def worst_case_image(h: int, w: int | None = None, seed: int = 0) -> np.ndarray:
    """Peor caso: ruido uniforme de alta amplitud (nada es uniforme salvo 1 píxel)."""
    w = h if w is None else w
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
