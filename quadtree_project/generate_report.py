"""generate_report.py -- Genera reporte_tecnico.pdf a partir de results/*.csv y results/plots/*.png.
Ejecutar después de experiments.py:   python generate_report.py
"""
import os

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table,
                                TableStyle, Preformatted, KeepTogether)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# ---- fuentes con soporte Unicode (Θ, ≤, ≈, ×, →) incluidas con matplotlib
fd = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(fd, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", os.path.join(fd, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", os.path.join(fd, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", os.path.join(fd, "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", os.path.join(fd, "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

BODY = ParagraphStyle("body", fontName="DV", fontSize=8, leading=10.1, alignment=TA_JUSTIFY, spaceAfter=3)
H1 = ParagraphStyle("h1", fontName="DV-B", fontSize=10, leading=12, spaceBefore=5, spaceAfter=3,
                    textColor=colors.HexColor("#1a3c6e"))
H2 = ParagraphStyle("h2", fontName="DV-B", fontSize=8.4, leading=10.4, spaceBefore=3, spaceAfter=2)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7, leading=8.6, alignment=TA_CENTER,
                     textColor=colors.HexColor("#333333"), spaceAfter=4)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=9, bulletIndent=0, spaceAfter=1.5)
CODE = ParagraphStyle("code", fontName="DVM", fontSize=6.8, leading=8.2, backColor=colors.HexColor("#f3f3f3"),
                      borderPadding=3, spaceAfter=5, spaceBefore=2)


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def fig(name, width_cm, caption):
    path = os.path.join(RES, name)
    from PIL import Image as PI
    w, h = PI.open(path).size
    return KeepTogether([Image(path, width=width_cm * cm, height=width_cm * cm * h / w), P(caption, CAP)])


def tbl(data, colw, font=6.8, header_rows=1, zebra=True):
    t = Table(data, colWidths=colw, repeatRows=header_rows)
    st = [("FONT", (0, 0), (-1, -1), "DV", font), ("FONT", (0, 0), (-1, header_rows - 1), "DV-B", font),
          ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
          ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#dfe7f3")),
          ("TOPPADDING", (0, 0), (-1, -1), 1.3), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.3),
          ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]
    if zebra:
        for r in range(header_rows, len(data)):
            if (r - header_rows) % 2 == 1:
                st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f6f6f6")))
    t.setStyle(TableStyle(st))
    return t


# ------------------------------------------------------------------ datos
df = pd.read_csv(os.path.join(RES, "results.csv"))
ex = pd.read_csv(os.path.join(RES, "extra_results.csv"))


def g(content, n, tol, col):
    r = df[(df.content == content) & (df.height == n) & (df.tol == tol)].iloc[0]
    return r[col]


def x(content, n, tol, col, crit="range", exp=None):
    q = ex[(ex.content == content) & (ex.height == n) & (ex.tol == tol) & (ex.criterion == crit)]
    if exp:
        q = q[q.experiment == exp]
    return q.iloc[0][col]


def fmt_ms(v):
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}"


# ------------------------------------------------------------------ documento
story = []

# ---------- Portada
cover = ParagraphStyle("c", fontName="DV-B", fontSize=20, leading=26, alignment=TA_CENTER, textColor=colors.HexColor("#1a3c6e"))
sub = ParagraphStyle("s", fontName="DV", fontSize=11, leading=16, alignment=TA_CENTER)
story += [Spacer(1, 6 * cm), P("Compresión de imágenes con Quadtree<br/>mediante Divide y Vencerás", cover), Spacer(1, 0.6 * cm),
          P("Reporte técnico — Diseño, análisis teórico y evaluación experimental", sub), Spacer(1, 2 * cm),
          P("Curso: ____________________________", sub), P("Equipo / integrantes: ____________________________", sub),
          P("Fecha: ____________________________", sub), PageBreak()]

# ---------- A. Modelado
story.append(P("A. Modelado del problema", H1))
story.append(P(
    "<b>Entrada.</b> Una imagen I ∈ {0,…,255}<super>H×W×C</super> (C = 3 para RGB o C = 1 para grises), cargada desde archivo y "
    "guardada como matriz <i>uint8</i>, más una tolerancia τ ≥ 0 y un criterio de uniformidad. El tamaño de entrada es "
    "N = H·W píxeles; para imágenes cuadradas, n = H = W y N = n². "
    "<b>Salida.</b> (i) Un árbol Quadtree T; (ii) su codificación binaria; (iii) la imagen reconstruida Î con las mismas "
    "dimensiones; (iv) métricas (nodos, hojas, profundidad, bytes, razón de compresión CR, MSE, PSNR, error máximo)."))
story.append(P(
    "<b>Similitud de color.</b> Una región R es <i>uniforme</i> si <b>max<sub>c</sub> ( max<sub>R</sub> I<sub>c</sub> − min<sub>R</sub> I<sub>c</sub> ) ≤ τ</b> "
    "(criterio «range», norma L<sub>∞</sub> por canal). Se eligió porque da una <b>cota garantizada</b> del error: el color "
    "representativo es la media redondeada de R, que cae dentro de [min, max]; por lo tanto |I<sub>c</sub>(p) − Î<sub>c</sub>(p)| ≤ τ para todo píxel p "
    "y canal c, y MSE ≤ τ². Con τ = 0 la región debe ser de color exactamente constante y la compresión es <b>sin pérdida</b>. "
    "Como criterio alternativo (opcional) se implementó «var»: Var<sub>R</sub>(I<sub>c</sub>) ≤ τ² para todo c, evaluable en Θ(1) con "
    "imágenes integrales [5]; en aritmética entera exacta la condición es N<sub>R</sub>·ΣI² − (ΣI)² ≤ τ²·N<sub>R</sub>²."))
story.append(P(
    "<b>Almacenamiento y métricas.</b> Formato binario propio: cabecera de 48 bits (marca, C, alto, ancho) y recorrido en preorden con "
    "1 bit por nodo (1 = hoja, 0 = interno) más 8·C bits de color por hoja. No se guardan coordenadas ni punteros: el decodificador "
    "re-deriva las regiones con la misma regla de partición. Tamaño = 48 + nodos + 8·C·hojas bits (redondeado a bytes; "
    "verificado contra los bytes reales escritos). Razón de compresión <b>CR = 8·C·H·W / bits_comprimidos</b> (CR &gt; 1 comprime; "
    "CR &lt; 1 expande). MSE = media de (I − Î)² sobre todos los píxeles y canales; PSNR = 10·log<sub>10</sub>(255²/MSE)."))

# ---------- B. Diseño
story.append(P("B. Diseño algorítmico", H1))
story.append(P(
    "<b>Estrategia.</b> <i>Dividir</i>: si la región no es uniforme y tiene más de un píxel, se parte en sus cuadrantes. "
    "<i>Resolver</i>: se procesa recursivamente cada cuadrante; casos base: región uniforme (hoja con su color medio) o región de 1×1 "
    "(siempre uniforme). <i>Combinar</i>: los hijos se enlazan a un nodo padre que conserva la región (y, x, h, w), de modo que "
    "el árbol conserva la distribución espacial y basta recorrerlo para reconstruir la imagen."))
story.append(Preformatted(
    "CONSTRUIR(I, y, x, h, w, τ):\n"
    "    si h·w = 1  ó  UNIFORME(I, y, x, h, w, τ):             # casos base\n"
    "        devolver HOJA(y, x, h, w, color = redondear(media(I[región])))\n"
    "    para cada (y', x', h', w') en PARTIR(y, x, h, w):       # DIVIDIR (2 ó 4 subregiones no vacías)\n"
    "        hijos.añadir( CONSTRUIR(I, y', x', h', w', τ) )     # RESOLVER\n"
    "    devolver NODO(y, x, h, w, hijos)                        # COMBINAR", CODE))
story.append(P(
    "<b>Dimensiones arbitrarias.</b> PARTIR usa h₁ = ⌊h/2⌋, h₂ = h − h₁ (igual para el ancho) y descarta los cuadrantes con alto o "
    "ancho 0 (solo ocurre si h = 1 o w = 1; entonces el nodo tiene 2 hijos). Así los cuadrantes cubren la región sin traslapes ni huecos "
    "y cada hijo tiene estrictamente menos píxeles que el padre: no se subdividen regiones vacías y la recursión siempre termina. "
    "Se probó con 100×75, 127×129, 1×20 y 20×1. "
    "<b>Estructura.</b> <i>Node</i> = (y, x, h, w, color | None, hijos[0..4]); una hoja tiene color y no tiene hijos. "
    "No se usó ninguna biblioteca que construya o comprima Quadtrees (NumPy solo maneja matrices; Pillow solo lee/escribe archivos). "
    "<b>Reconstrucción.</b> Recorrido recursivo que pinta el rectángulo de cada hoja con su color."))

# ---------- C. Correctitud y complejidad
story.append(P("C. Análisis de correctitud y complejidad", H1))
story.append(P(
    "<b>Correctitud.</b> (1) <i>Terminación:</i> el número de píxeles de cada hijo es estrictamente menor que el del padre y es ≥ 1; "
    "con área 1 se alcanza el caso base. (2) <i>Teselación:</i> por inducción, si los cuadrantes de una región la cubren exactamente una vez "
    "(propiedad de PARTIR), las hojas del árbol cubren la imagen exactamente una vez; verificado por prueba automática. "
    "(3) <i>Cota de error:</i> en cada hoja uniforme el color medio está en [min, max], de modo que el error por píxel es ≤ τ; "
    "con τ = 0 la reconstrucción es idéntica (verificado en 3 tipos de imagen y 2 criterios). (4) <i>Codificación:</i> la "
    "serialización → deserialización devuelve el mismo árbol (prueba <font name='DVM'>tests.py</font>, 8 pruebas)."))
story.append(P(
    "<b>Complejidad temporal.</b> Operación fundamental: lectura/comparación de un píxel en UNIFORME (y creación de un nodo). "
    "Para n×n (potencia de 2), con el criterio «range» la verificación cuesta Θ(n²) en una región de lado n, y sea k ∈ {0,…,4} el número de "
    "cuadrantes que resultan no uniformes: <b>T(n) = k·T(n/2) + Θ(n²)</b>, T(1) = Θ(1)."))
story += bullets([
    "<b>Peor caso</b> (k = 4 en todos los niveles; p. ej. ruido con τ = 0): a = 4, b = 2, f(n) = Θ(n²) = Θ(n<super>log<sub>2</sub>4</super>) → caso 2 del Teorema Maestro: "
    "<b>T(n) = Θ(n² log n)</b>. Hay ≤ (4n² − 1)/3 nodos (medido: 87 381 para n = 256), pero cada nivel re-examina todos los píxeles y hay log n niveles.",
    "<b>Mejor caso</b> (imagen uniforme): una sola verificación, sin recursión, <b>Θ(n²)</b> (hay que leer todos los píxeles al menos una vez).",
    "<b>Caso intermedio</b> (fronteras de longitud O(n), p. ej. «regions»): a profundidad d solo O(2<super>d</super>) regiones son no uniformes, con área total O(n²/2<super>d</super>); "
    "la suma geométrica de las verificaciones es dominada por la raíz → <b>Θ(n²)</b> (equivale a k &lt; 4 en la recurrencia, caso 3 del Teorema Maestro).",
    "<b>Criterio «var»</b> con imágenes integrales: la verificación es Θ(1) y el preproceso Θ(n²): T(n) = 4T(n/2) + Θ(1) = Θ(n²) en el peor caso y Θ(n²) por el preproceso en el mejor.",
    "El cálculo del color medio de las hojas suma Θ(Σ áreas de hojas) = Θ(n²) en total; no altera el orden.",
])
story.append(P("<b>Complejidad espacial y costos por separado.</b> El árbol ocupa Θ(#nodos): Θ(1) en el mejor caso y Θ(n²) en el peor (≤ 4n²/3 nodos). "
               "La pila de recursión llega a ⌈log₂ n⌉ + 1 marcos: <b>O(log n)</b> (profundidades medidas: 6, 7, 8 para n = 64, 128, 256). "
               "El criterio «var» añade 2·(n+1)²·C enteros de 64 bits. Ver tabla:"))
t_cost = [["Operación", "Tiempo (peor)", "Tiempo (mejor)", "Espacio extra"],
          ["Construir (range)", "Θ(n² log n)", "Θ(n²)", "árbol Θ(n²) + pila O(log n)"],
          ["Construir (var)", "Θ(n²)", "Θ(n²)", "árbol + tablas integrales Θ(n²)"],
          ["Recorrer (estadísticas)", "Θ(#nodos) = O(n²)", "Θ(1)", "pila O(log n)"],
          ["Reconstruir", "Θ(n² + #nodos) = Θ(n²)", "Θ(n²) (se escribe cada píxel)", "imagen de salida Θ(n²)"]]
story += [KeepTogether([tbl(t_cost, [3.4 * cm, 4.4 * cm, 4.2 * cm, 5.4 * cm], font=6.8), Spacer(1, 3)])]

# ---------- D. Resultados
story.append(P("D. Resultados experimentales", H1))
story.append(P(
    "<b>Diseño.</b> 3 tamaños (64², 128², 256²) × 3 tolerancias (τ = 0 exacta, 16, 48) × 3 contenidos: <i>uniform</i> (un color), "
    "<i>regions</i> (4 regiones planas: 3 rectángulos y un círculo) y <i>detail</i> (degradado + círculo + línea fina + ruido gaussiano σ = 6). "
    "Imágenes sintéticas con semilla fija (reproducible); tiempo = mediana de 5 ejecuciones (Python 3, NumPy, un solo hilo). "
    "Todos los registros (también tiempos de recorrido y reconstrucción, bytes, PSNR, error máximo) están en <font name='DVM'>results/results.csv</font>."))

hdr1 = ["Imagen"] + ["Hojas", "CR", "MSE", "ms"] * 3
hdr0 = [""] + ["τ = 0", "", "", ""] + ["τ = 16", "", "", ""] + ["τ = 48", "", "", ""]
rows = [hdr0, hdr1]
for c in ("uniform", "regions", "detail"):
    for n in (64, 128, 256):
        r = [f"{c} {n}²"]
        for tol in (0, 16, 48):
            cr = g(c, n, tol, "compression_ratio")
            r += [f"{int(g(c, n, tol, 'leaves'))}", (f"{cr:.0f}" if cr >= 100 else f"{cr:.2f}"),
                  f"{g(c, n, tol, 'mse'):.1f}", fmt_ms(g(c, n, tol, "build_ms"))]
        rows.append(r)
t1 = tbl(rows, [2.2 * cm] + [1.1 * cm, 1.2 * cm, 0.95 * cm, 1.0 * cm] * 3, font=6.4, header_rows=2)
t1.setStyle(TableStyle([("SPAN", (1, 0), (4, 0)), ("SPAN", (5, 0), (8, 0)), ("SPAN", (9, 0), (12, 0))]))
story += [t1, P("Tabla 1. Hojas, razón de compresión CR, MSE y tiempo de construcción (ms). Nodos = (4·hojas − 1)/3 (árbol 4-ario completo en potencias de 2); "
                "profundidad máxima = log₂ n en todos los casos salvo «uniform» (0). Detalle completo en el CSV.", CAP)]

story.append(fig("plots/time_vs_size.png", 13.5, "Figura 1. Tiempo de construcción vs. lado n (escala log-log) para cada tolerancia y tipo de contenido."))

story.append(P("<b>¿Cómo influye el tamaño?</b> "
               f"En «uniform» el tiempo crece ×{g('uniform',128,0,'build_ms')/g('uniform',64,0,'build_ms'):.1f} y ×{g('uniform',256,0,'build_ms')/g('uniform',128,0,'build_ms'):.1f} al duplicar n "
               "(≈ ×4 = Θ(n²), una sola pasada). En «detail» con τ = 0 (peor caso) crece "
               f"×{g('detail',128,0,'build_ms')/g('detail',64,0,'build_ms'):.1f} y ×{g('detail',256,0,'build_ms')/g('detail',128,0,'build_ms'):.1f}; en «regions» solo "
               f"×{g('regions',128,0,'build_ms')/g('regions',64,0,'build_ms'):.1f} y ×{g('regions',256,0,'build_ms')/g('regions',128,0,'build_ms'):.1f}, porque el número de nodos sigue el perímetro de las regiones "
               f"(Θ(n)): {int(g('regions',64,0,'nodes'))}, {int(g('regions',128,0,'nodes'))}, {int(g('regions',256,0,'nodes'))} nodos. "
               "En la Fig. 3 (ruido, τ = 0, n hasta 512) cada duplicación multiplica el tiempo por ≈ 4, consistente con Θ(n²). "
               "<i>Nota honesta:</i> el factor log n del peor caso teórico no se distingue de Θ(n²) en estas mediciones, porque en Python el costo por nodo "
               "(≈ 4n²/3 llamadas) domina sobre las operaciones vectorizadas de NumPy que ejecutan la verificación; la recurrencia describe el modelo de costo de lectura de píxeles, no el tiempo de pared de esta implementación."))
story.append(P("<b>¿Cómo afecta τ al número de regiones?</b> "
               f"En «detail» 256² las hojas pasan de {int(g('detail',256,0,'leaves'))} (τ = 0) a {int(g('detail',256,16,'leaves'))} (τ = 16) y {int(g('detail',256,48,'leaves'))} (τ = 48). "
               "El efecto es de umbral: mientras τ sea menor que la amplitud del ruido y del degradado en un bloque, solo se fusionan bloques diminutos; al superarla, bloques grandes se vuelven uniformes. "
               "En «uniform» y «regions» (colores planos) τ no cambia nada porque los bordes son nítidos. Con ruido leve (σ = 2) sobre las mismas regiones (Tabla 2), τ = 0 y τ = 4 apenas comprimen "
               "y la compresión se dispara cuando τ supera el rango del ruido (≈ 8–16)."))

rows2 = [["τ", "0", "4", "8", "16", "48"],
         ["Hojas"] + [str(int(x("noisy_regions", 256, t, "leaves", exp="noisy_regions"))) for t in (0, 4, 8, 16, 48)],
         ["CR"] + [f"{x('noisy_regions',256,t,'compression_ratio',exp='noisy_regions'):.2f}" for t in (0, 4, 8, 16, 48)],
         ["MSE"] + [f"{x('noisy_regions',256,t,'mse',exp='noisy_regions'):.2f}" for t in (0, 4, 8, 16, 48)],
         ["Error máx."] + [str(int(x("noisy_regions", 256, t, "max_abs_error", exp="noisy_regions"))) for t in (0, 4, 8, 16, 48)]]
story += [tbl(rows2, [2.2 * cm] + [1.9 * cm] * 5, font=6.8),
          P("Tabla 2. «regions» 256² con ruido gaussiano σ = 2 (experimento <font name='DVM'>noisy_regions</font> en extra_results.csv).", CAP)]

story.append(fig("plots/ratio_mse_vs_tol.png", 11.5, "Figura 2. Razón de compresión (izq.) y MSE (der.) vs. tolerancia (MSE = 0 se dibuja como 10⁻³)."))
story.append(P("<b>Compresión vs. calidad.</b> "
               f"En «detail» 256²: τ = 16 da CR = {g('detail',256,16,'compression_ratio'):.2f} con MSE = {g('detail',256,16,'mse'):.1f} (PSNR = {g('detail',256,16,'psnr_db'):.1f} dB) y τ = 48 da "
               f"CR = {g('detail',256,48,'compression_ratio'):.1f} con MSE = {g('detail',256,48,'mse'):.1f} (PSNR = {g('detail',256,48,'psnr_db'):.1f} dB). "
               f"El error máximo observado ({int(g('detail',256,16,'max_abs_error'))} y {int(g('detail',256,48,'max_abs_error'))}) respeta la cota τ. "
               "La relación no es lineal: el salto de CR ≈ 1.5 a ≈ 43 cuesta ≈ 7 dB de PSNR. "
               f"<b>Casos sin ahorro:</b> con τ = 0 y contenido ruidoso «detail» (y el ruido uniforme del peor caso) CR = {g('detail',256,0,'compression_ratio'):.3f} &lt; 1: "
               "todas las hojas son píxeles, y el árbol añade 1 bit por nodo (4N/3 bits) a los 24N bits de datos, así que CR → 18/19 ≈ 0.947, es decir, <b>+5.6 % de almacenamiento</b>. "
               "También con τ = 16 en «detail» el ahorro es mínimo (CR ≈ 1.4–1.5)."))

story.append(fig("plots/scaling.png", 11.5, "Figura 3. Izq.: escalamiento del tiempo (peor caso = ruido, τ = 0; mejor caso = uniforme) con ambos criterios y referencias n² y n² log n. "
                                              "Der.: nodos/píxeles en el peor caso → 4/3."))
story.append(P(
    f"<b>Criterio «var» (opcional).</b> Con τ = 0 construye exactamente el mismo árbol que «range». Con τ = 16 en «detail» 256² tarda {fmt_ms(x('detail',256,16,'build_ms','var','criterion_comparison'))} ms frente a "
    f"{fmt_ms(x('detail',256,16,'build_ms','range','criterion_comparison'))} ms, pero no son comparables a igual τ: «var» acota la varianza de la región, no el error por píxel "
    f"(CR = {x('detail',256,16,'compression_ratio','var','criterion_comparison'):.1f}, MSE = {x('detail',256,16,'mse','var','criterion_comparison'):.1f}, error máx. = {int(x('detail',256,16,'max_abs_error','var','criterion_comparison'))}). "
    "<b>Dimensiones no potencia de 2</b> (100×75 y 127×129): todas las pruebas pasan y el comportamiento es análogo (p. ej. «regions» 127×129: CR = "
    f"{x('regions',127,0,'compression_ratio',exp='non_power_of_two'):.1f}, MSE = 0)."))
story.append(fig("evidence/ejemplo2_detail_256_tol48.png", 12.5,
                 "Figura 4. Evidencia (ejemplo 2): original, reconstruida y partición Quadtree. Los ejemplos 1 y 3 están en results/evidence/."))

# ---------- E. Conclusiones
story.append(P("E. Conclusiones", H1))
story += bullets([
    "<b>Ventajas.</b> Algoritmo simple y correcto (cota de error ≤ τ demostrada y comprobada); con τ = 0 es sin pérdida; en imágenes con regiones planas grandes la compresión es "
    f"muy alta (CR = {g('regions',256,0,'compression_ratio'):.0f} sin pérdida en «regions» 256²; {g('uniform',256,0,'compression_ratio'):.0f} en «uniform»); los tiempos de construcción son de milisegundos; "
    "la recursión usa pila O(log n); τ es un control directo del compromiso tasa–distorsión.",
    "<b>Limitaciones.</b> La partición es rígida (por mitades), así que bordes diagonales o curvos obligan a subdividir hasta el píxel (círculo en «regions», Fig. 4); "
    "con ruido o texturas y τ pequeño el árbol <b>expande</b> los datos (hasta +5.6 %); el peor caso con «range» es Θ(n² log n) en el modelo de lectura de píxeles; "
    "el color medio por bloque produce artefactos de bloques (Fig. 4) y el error por región es acotado, no el error global perceptual.",
    "<b>Cuándo conviene.</b> Gráficos, diagramas, logotipos, mapas, máscaras y píxel-art (regiones planas); también como estructura jerárquica para búsqueda espacial. "
    "No conviene en fotografías ruidosas con τ ≈ 0. Mejoras posibles: usar «var» o tablas de rango (O(1)) para la verificación, poda posterior de hijos similares, "
    "codificación entrópica de colores y partición adaptativa. "
    "<b>Limitaciones del estudio:</b> imágenes sintéticas, una sola máquina y tiempos de Python dominados por el costo por nodo.",
])

# ---------- Referencias
story.append(PageBreak())
story.append(P("Referencias y bibliotecas", H1))
refs = [
    "[1] R. A. Finkel y J. L. Bentley, «Quad trees: a data structure for retrieval on composite keys», <i>Acta Informatica</i> 4(1), 1–9, 1974.",
    "[2] H. Samet, «The Quadtree and Related Hierarchical Data Structures», <i>ACM Computing Surveys</i> 16(2), 187–260, 1984.",
    "[3] T. H. Cormen, C. E. Leiserson, R. L. Rivest y C. Stein, <i>Introduction to Algorithms</i>, 3.ª/4.ª ed., MIT Press (Divide y Vencerás, Teorema Maestro).",
    "[4] Z. Wang, A. C. Bovik, H. R. Sheikh y E. P. Simoncelli, «Image quality assessment: from error visibility to structural similarity», <i>IEEE Trans. Image Processing</i> 13(4), 600–612, 2004 (MSE/PSNR como métricas de referencia).",
    "[5] F. C. Crow, «Summed-area tables for texture mapping», <i>SIGGRAPH Computer Graphics</i> 18(3), 207–212, 1984 (imágenes integrales).",
    "<b>Bibliotecas auxiliares:</b> NumPy (matrices), Pillow (lectura/escritura de imágenes), Matplotlib (gráficas), pandas (CSV), ReportLab (este PDF). "
    "Ninguna implementa Quadtrees ni compresión: la construcción, recorrido, serialización y métricas son código propio del equipo.",
]
story += [P(r) for r in refs]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7)
    canvas.setFillColor(colors.grey)
    if doc.page > 1:
        canvas.drawCentredString(letter[0] / 2, 0.9 * cm, f"Compresión con Quadtree — Divide y Vencerás | pág. {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(os.path.join(HERE, "reporte_tecnico.pdf"), pagesize=letter, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                        topMargin=1.5 * cm, bottomMargin=1.6 * cm, title="Compresión de imágenes con Quadtree",
                        author="Equipo")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("reporte_tecnico.pdf generado")
