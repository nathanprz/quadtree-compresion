# Compresión de imágenes con Quadtree (Divide y Vencerás)

Prototipo de compresión/reconstrucción de imágenes con un Quadtree implementado desde cero.
Python 3.9+; dependencias en `requirements.txt` (NumPy, Pillow, Matplotlib, pandas; ReportLab solo para el PDF).

## Instalación
```bash
pip install -r requirements.txt
```

## Estructura
| Archivo | Contenido |
|---|---|
| `quadtree.py` | `Node`, `build_quadtree` (D&C), `reconstruct`, serialización binaria, métricas (MSE, PSNR, CR) |
| `images.py` | Carga/guardado de imágenes y generador reproducible de imágenes de prueba (`uniform`, `regions`, `detail`) |
| `tests.py` | 8 pruebas de correctitud (teselación, error ≤ τ, sin pérdida con τ=0, ida y vuelta binaria, dims. impares) |
| `experiments.py` | Malla 3 tamaños × 3 tolerancias × 3 contenidos + experimentos extra; produce CSV, gráficas y evidencias |
| `main.py` | CLI para comprimir/descomprimir imágenes propias |
| `generate_report.py` | Regenera `reporte_tecnico.pdf` desde los CSV y gráficas |
| `results/` | `results.csv`, `extra_results.csv`, `plots/`, `evidence/` (3 ejemplos + `EVIDENCIAS.md`) |
| `images/` | Imágenes de prueba 256×256 (se pueden regenerar con `images.make_image`) |

## Ejecución
```bash
python tests.py                 # pruebas de correctitud
python experiments.py           # todos los experimentos (~1 min) -> results/
python generate_report.py       # reporte_tecnico.pdf

# Comprimir una imagen propia (RGB; --gray para escala de grises)
python main.py compress images/detail_256.png --tol 48 --out salida/
python main.py decompress salida/detail_256.qt --out reconstruida.png
```
Salida de `compress`: archivo `.qt` (formato binario propio), imagen reconstruida, imagen con la partición y métricas en consola.

## Criterios de uniformidad (`--criterion`)
* `range` (por defecto): max por canal de (máx − mín) ≤ τ. Garantiza error por píxel ≤ τ. τ=0 es sin pérdida. Verificación Θ(área).
* `var` (opcional): varianza por canal ≤ τ² con imágenes integrales. Verificación Θ(1); acota el MSE de la región, no el error por píxel.

## Formato de almacenamiento
Cabecera de 48 bits (marca `Q`, canales, alto, ancho) + preorden del árbol: 1 bit por nodo (1 = hoja) + 8·C bits de color por hoja.
`CR = 8·C·H·W / bits_comprimidos`.

## Reproducibilidad
Las imágenes se generan con semilla fija. Los tiempos son la mediana de 5 ejecuciones y varían según la máquina; el resto de las métricas son deterministas.
