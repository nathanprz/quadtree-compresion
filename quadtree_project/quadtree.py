"""
quadtree.py -- Compresión de imágenes con Quadtree (Divide y Vencerás).

Contenido
---------
* Node                 : nodo del árbol (región + color representativo + hijos).
* build_quadtree       : construcción recursiva (DIVIDIR / RESOLVER / COMBINAR).
* reconstruct          : reconstrucción recorriendo el árbol.
* serialize/deserialize: formato binario real (bits) para medir almacenamiento.
* metrics              : MSE, PSNR, error máximo, razón de compresión, etc.

Convenciones
------------
* Imagen: ndarray uint8 de forma (H, W, C) con C = 1 (grises) o 3 (RGB).
* Región: (y, x, h, w) = fila inicial, columna inicial, alto y ancho (h, w >= 1).
* La construcción es implementada íntegramente aquí; NumPy solo se usa para
  manejar matrices (permitido por el enunciado).

Criterios de uniformidad (definición matemática)
------------------------------------------------
Sea R una región de N = h*w píxeles y c un canal.

* "range"  : R es uniforme  <=>  max_c ( max_R I_c - min_R I_c ) <= tol
             Consecuencia: el color medio de R queda dentro de [min, max], por lo
             que el error por píxel y canal es <= tol (cota garantizada).
             Costo de la verificación: Theta(N) (recorre la región).

* "var"    : R es uniforme  <=>  max_c Var_R(I_c) <= tol^2
             Consecuencia: el MSE de la región respecto de su media es <= tol^2.
             Se evalúa en Theta(1) con imágenes integrales (sumas y sumas de
             cuadrados acumuladas, en aritmética entera exacta):
                 N*S2 - S^2  <=  tol^2 * N^2          (Var = (N*S2 - S^2) / N^2)
             El preprocesamiento cuesta Theta(n^2) una sola vez.

Partición para dimensiones arbitrarias
--------------------------------------
Una región (h, w) se parte en:  h1 = h//2, h2 = h - h1;  w1 = w//2, w2 = w - w1.
Los 4 cuadrantes son (y,x,h1,w1) (y,x+w1,h1,w2) (y+h1,x,h2,w1) (y+h1,x+w1,h2,w2).
Los cuadrantes con alto o ancho 0 se descartan (ocurre solo si h==1 o w==1), por
lo que un nodo interno tiene 2 o 4 hijos. Esto cubre todos los píxeles sin
traslapes ni omisiones y garantiza que cada hijo es estrictamente más pequeño
que el padre (h*w decrece), de modo que la recursión termina.
"""
from __future__ import annotations

import struct
import numpy as np

# --------------------------------------------------------------------------
# Estructura de datos
# --------------------------------------------------------------------------


class Node:
    """Nodo del Quadtree.

    Atributos
    ---------
    y, x, h, w : región cubierta (fila, columna, alto, ancho)
    color      : tupla de enteros (color medio) si es hoja, None si es interno
    children   : lista de 2 o 4 nodos si es interno, [] si es hoja
    """

    __slots__ = ("y", "x", "h", "w", "color", "children")

    def __init__(self, y, x, h, w, color=None, children=None):
        self.y, self.x, self.h, self.w = y, x, h, w
        self.color = color
        self.children = children if children is not None else []

    @property
    def is_leaf(self) -> bool:
        return not self.children


def split_region(y: int, x: int, h: int, w: int):
    """DIVIDIR: devuelve la lista de subregiones no vacías (2 o 4) de (y,x,h,w)."""
    h1, w1 = h // 2, w // 2
    h2, w2 = h - h1, w - w1
    quads = [
        (y, x, h1, w1),
        (y, x + w1, h1, w2),
        (y + h1, x, h2, w1),
        (y + h1, x + w1, h2, w2),
    ]
    return [q for q in quads if q[2] > 0 and q[3] > 0]


# --------------------------------------------------------------------------
# Criterios de uniformidad
# --------------------------------------------------------------------------


class _RangeCriterion:
    """max_c(max - min) <= tol, evaluado recorriendo la región: Theta(h*w)."""

    name = "range"

    def __init__(self, img: np.ndarray, tol: float):
        self.img, self.tol = img, tol

    def is_uniform(self, y, x, h, w) -> bool:
        reg = self.img[y:y + h, x:x + w]
        spread = reg.max(axis=(0, 1)).astype(np.int32) - reg.min(axis=(0, 1)).astype(np.int32)
        return int(spread.max()) <= self.tol


class _VarCriterion:
    """Var <= tol^2 por canal con imágenes integrales: Theta(1) por consulta."""

    name = "var"

    def __init__(self, img: np.ndarray, tol: float):
        self.tol2 = int(np.floor(tol * tol))  # cota entera (exacta para tol entero)
        a = img.astype(np.int64)
        pad = ((1, 0), (1, 0), (0, 0))
        self.S = np.pad(a.cumsum(0).cumsum(1), pad)
        self.S2 = np.pad((a * a).cumsum(0).cumsum(1), pad)

    @staticmethod
    def _box(T, y, x, h, w):
        return T[y + h, x + w] - T[y, x + w] - T[y + h, x] + T[y, x]

    def is_uniform(self, y, x, h, w) -> bool:
        n = h * w
        s = self._box(self.S, y, x, h, w)
        s2 = self._box(self.S2, y, x, h, w)
        lhs = n * s2 - s * s                       # = n^2 * Var, entero exacto
        return bool((lhs <= self.tol2 * n * n).all())


def _make_criterion(img, tol, criterion):
    if criterion == "range":
        return _RangeCriterion(img, tol)
    if criterion == "var":
        return _VarCriterion(img, tol)
    raise ValueError(f"criterio desconocido: {criterion!r}")


# --------------------------------------------------------------------------
# Construcción (Divide y Vencerás)
# --------------------------------------------------------------------------


def build_quadtree(img: np.ndarray, tol: float = 0, criterion: str = "range") -> Node:
    """Construye el Quadtree de `img` (H, W, C uint8) con tolerancia `tol`.

    DIVIDIR   : si la región no es uniforme y tiene más de un píxel, se parte en
                2 o 4 subregiones (split_region).
    RESOLVER  : llamada recursiva sobre cada subregión. Casos base:
                  (a) región uniforme  -> hoja con el color medio,
                  (b) región de 1x1    -> hoja (un píxel siempre es uniforme).
    COMBINAR  : los nodos hijos se enlazan a un nodo padre que conserva la región
                (la distribución espacial se recupera del árbol).
    """
    if img.ndim == 2:
        img = img[:, :, None]
    H, W, _ = img.shape
    crit = _make_criterion(img, tol, criterion)

    def mean_color(y, x, h, w):
        m = img[y:y + h, x:x + w].reshape(-1, img.shape[2]).mean(axis=0)
        return tuple(int(v) for v in np.rint(m))

    def build(y, x, h, w) -> Node:
        # Caso base: región uniforme (un píxel lo es siempre) -> hoja
        if h * w == 1 or crit.is_uniform(y, x, h, w):
            return Node(y, x, h, w, color=mean_color(y, x, h, w))
        # DIVIDIR + RESOLVER recursivamente
        children = [build(*q) for q in split_region(y, x, h, w)]
        # COMBINAR en un nodo padre
        return Node(y, x, h, w, children=children)

    return build(0, 0, H, W)


# --------------------------------------------------------------------------
# Recorridos: reconstrucción y estadísticas
# --------------------------------------------------------------------------


def reconstruct(root: Node, channels: int = 3) -> np.ndarray:
    """Reconstruye la imagen (H, W, C) recorriendo el árbol: Theta(n_pixeles)."""
    out = np.zeros((root.h, root.w, channels), dtype=np.uint8)

    def paint(node: Node):
        if node.is_leaf:
            out[node.y:node.y + node.h, node.x:node.x + node.w] = node.color
        else:
            for c in node.children:
                paint(c)

    paint(root)
    return out


def tree_stats(root: Node) -> dict:
    """Nodos totales, hojas y profundidad máxima (raíz = profundidad 0)."""
    def rec(n, d):
        if n.is_leaf:
            return 1, 1, d
        tot, leaves, depth = 1, 0, d
        for c in n.children:
            t, l, dd = rec(c, d + 1)
            tot += t
            leaves += l
            depth = max(depth, dd)
        return tot, leaves, depth

    total, leaves, depth = rec(root, 0)
    return {"nodes": total, "leaves": leaves, "depth": depth}


def leaves_of(root: Node):
    """Lista de hojas (para visualizar la partición)."""
    res = []

    def rec(n):
        if n.is_leaf:
            res.append(n)
        else:
            for c in n.children:
                rec(c)

    rec(root)
    return res


# --------------------------------------------------------------------------
# Formato de almacenamiento (bits reales)
# --------------------------------------------------------------------------
#
# Cabecera (6 bytes = 48 bits):  magic 'Q' (8) | canales (8) | alto (16) | ancho (16)
# Cuerpo: recorrido en preorden; por cada nodo:
#     1 bit  : 1 = hoja, 0 = interno
#     8*C bits (solo hojas): color medio (C canales de 8 bits)
# No se guardan coordenadas ni punteros: el decodificador re-deriva las regiones
# con la misma función split_region a partir de (alto, ancho). El número de hijos
# (2 o 4) también es implícito (depende de si h==1 o w==1).
# Se redondea el total a bytes (relleno con ceros).

HEADER_BITS = 48


def storage_bits(stats: dict, channels: int = 3) -> int:
    """Tamaño exacto en bits (sin relleno) del formato anterior."""
    return HEADER_BITS + stats["nodes"] + stats["leaves"] * 8 * channels


def serialize(root: Node, channels: int = 3) -> bytes:
    bits = []

    def rec(n):
        if n.is_leaf:
            bits.append(1)
            for v in n.color:
                bits.extend((v >> k) & 1 for k in range(7, -1, -1))
        else:
            bits.append(0)
            for c in n.children:
                rec(c)

    rec(root)
    pad = (-len(bits)) % 8
    bits.extend([0] * pad)
    body = np.packbits(np.array(bits, dtype=np.uint8)).tobytes()
    header = struct.pack(">cBHH", b"Q", channels, root.h, root.w)
    return header + body


def deserialize(data: bytes) -> tuple[Node, int]:
    """Reconstruye el árbol desde bytes. Devuelve (raíz, canales)."""
    magic, channels, H, W = struct.unpack(">cBHH", data[:6])
    assert magic == b"Q", "formato inválido"
    bits = np.unpackbits(np.frombuffer(data[6:], dtype=np.uint8))
    pos = [0]

    def read(k):
        v = 0
        for _ in range(k):
            v = (v << 1) | int(bits[pos[0]])
            pos[0] += 1
        return v

    def rec(y, x, h, w):
        if read(1) == 1:
            return Node(y, x, h, w, color=tuple(read(8) for _ in range(channels)))
        return Node(y, x, h, w, children=[rec(*q) for q in split_region(y, x, h, w)])

    return rec(0, 0, H, W), channels


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d * d))


def psnr(m: float) -> float:
    return float("inf") if m == 0 else float(10 * np.log10(255.0 ** 2 / m))


def max_abs_error(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max())


def original_bits(h: int, w: int, channels: int = 3) -> int:
    """Imagen cruda sin compresión: 8 bits por canal y por píxel."""
    return h * w * channels * 8


def compression_ratio(h, w, channels, bits_compressed) -> float:
    """Razón de compresión CR = bits_original / bits_comprimido (>1 comprime)."""
    return original_bits(h, w, channels) / bits_compressed
