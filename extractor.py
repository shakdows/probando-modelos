#!/usr/bin/env python3
"""
NATTIVA · Extractor plano → scene_graph.json
Convierte una lámina de tipología (JPEG rasterizado de un deck de venta)
en un grafo de escena metrico, y se autovalida contra el área techada.

Uso:
    export ANTHROPIC_API_KEY=sk-ant-...
    python extractor.py --pdf Presentacion_Grau_10_2025.pdf --paginas 30-44 --out ./salida

Principio de diseño:
    El extractor NUNCA ve el área techada. La escala se calibra SOLO con las
    cotas impresas de los ambientes. El área techada queda libre como variable
    de control. Si el extractor la usara, la validación sería circular y no
    mediría nada.
"""

import argparse, base64, json, os, re, subprocess, sys, math
from pathlib import Path
from statistics import mean, pstdev

import anthropic  # pip install anthropic

MODELO = "claude-sonnet-4-6"
ALTURA_TECHO_M = 2.40  # RNE Perú: mínimo 2.30 m; Edifica usa 2.40 típico. Parametrizable.


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONTRATO: el esquema del grafo de escena
# ─────────────────────────────────────────────────────────────────────────────

ESQUEMA = {
    "ambientes": [
        {
            "nombre": "str — etiqueta EXACTA impresa en el plano (ej. 'SALA COMEDOR')",
            "tipo": "sala|dormitorio|cocina|bano|terraza|lavanderia|closet|circulacion",
            "cota_impresa_m": "[ancho, largo] o null si el plano no la imprime",
            "poligono_px": "[[x,y], ...] contorno del ambiente en píxeles, sentido horario",
            "muebles": ["cama|sofa|tv|mesa_comedor|encimera|lavadora|inodoro|ducha|lavatorio|..."],
        }
    ],
    "poligono_exterior_px": "[[x,y], ...] contorno de muros exteriores del departamento",
    "vanos": [
        {
            "tipo": "puerta|ventana|mampara",
            "extremos_px": "[[x1,y1],[x2,y2]] los dos extremos del vano sobre el muro",
        }
    ],
    "notas": "str — cualquier ambigüedad que encontraste al leer el plano",
}

PROMPT = """Eres un extractor de geometría arquitectónica. Recibes la lámina de una tipología
de departamento de un catálogo de venta peruano (imagen rasterizada, planta amoblada, vista cenital).

Tu trabajo es devolver la geometría en ESPACIO DE PÍXELES, más las cotas que estén impresas.

REGLAS DURAS:
1. NO inventes cotas. Si un ambiente no tiene medidas impresas en el plano, pon cota_impresa_m: null.
   El baño y la terraza casi nunca las traen. Eso es esperado y correcto.
2. Las etiquetas de ambiente van EXACTAS como están impresas, en mayúsculas.
   ("DORMITORIO PRINCIPAL", no "Dormitorio 1").
3. El polígono exterior sigue la cara EXTERIOR de los muros perimetrales (la línea negra gruesa).
4. Identifica el tipo de ambiente por los MUEBLES dibujados, no solo por la etiqueta:
   cama → dormitorio; inodoro/ducha/lavatorio → bano; encimera/campana/fregadero → cocina;
   sofá + TV + mesa → sala; piso exterior + plantas → terraza.
5. Si NO puedes trazar el polígono con confianza, dilo en "notas" en vez de adivinar.
6. NO calcules el área total. No es tu trabajo. Solo geometría en píxeles y cotas impresas.

Devuelve SOLO JSON válido, sin markdown, sin preámbulo, con este esquema:
""" + json.dumps(ESQUEMA, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Extracción de la verdad de terreno (capa de texto — sin IA, determinista)
# ─────────────────────────────────────────────────────────────────────────────

def leer_metadata(pdf: str, pagina: int) -> dict:
    """Área techada, área ocupada y dormitorios salen limpios del text layer."""
    txt = subprocess.run(
        ["pdftotext", "-layout", "-f", str(pagina), "-l", str(pagina), pdf, "-"],
        capture_output=True, text=True,
    ).stdout
    plano = re.sub(r"\s+", " ", txt)
    tipo = re.search(r"TIPO\s*(\d+)", plano)
    areas = re.findall(r"([\d.]+)\s*m\s*2", plano)
    dorm = re.search(r"DORMITORIOS\s*(\d+)", plano)
    return {
        "tipo": f"TIPO{tipo.group(1)}" if tipo else f"PAG{pagina}",
        "area_techada_m2": float(areas[0]) if areas else None,
        "area_ocupada_m2": float(areas[1]) if len(areas) > 1 else None,
        "dormitorios": int(dorm.group(1)) if dorm else None,
        "pagina": pagina,
    }


def extraer_imagen(pdf: str, pagina: int, destino: Path) -> Path:
    """El plano es un JPEG incrustado. Lo sacamos sin recomprimir."""
    destino.mkdir(parents=True, exist_ok=True)
    pref = destino / f"p{pagina}"
    subprocess.run(["pdfimages", "-png", "-f", str(pagina), "-l", str(pagina), pdf, str(pref)],
                   check=True, capture_output=True)
    cands = sorted(destino.glob(f"p{pagina}-*.png"), key=lambda p: p.stat().st_size, reverse=True)
    if not cands:
        raise RuntimeError(f"No se extrajo imagen de la página {pagina}")
    return cands[0]  # la más pesada es el plano; el resto son logos


# ─────────────────────────────────────────────────────────────────────────────
# 3. El extractor (VLM)
# ─────────────────────────────────────────────────────────────────────────────

def extraer_geometria(cliente, imagen: Path) -> dict:
    b64 = base64.standard_b64encode(imagen.read_bytes()).decode()
    r = cliente.messages.create(
        model=MODELO,
        max_tokens=8000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    crudo = "".join(b.text for b in r.content if b.type == "text")
    crudo = re.sub(r"^```(?:json)?|```$", "", crudo.strip(), flags=re.M).strip()
    return json.loads(crudo)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Calibración de escala — SOLO con cotas impresas
# ─────────────────────────────────────────────────────────────────────────────

def bbox(poli):
    xs = [p[0] for p in poli]; ys = [p[1] for p in poli]
    return max(xs) - min(xs), max(ys) - min(ys)


def calibrar(geo: dict) -> dict:
    """
    Cada ambiente con cota impresa da una estimación independiente de px/m.
    Si las estimaciones no concuerdan, la lectura del plano es mala — y lo sabemos
    ANTES de mirar el área techada.
    """
    est = []
    for amb in geo.get("ambientes", []):
        cota, poli = amb.get("cota_impresa_m"), amb.get("poligono_px")
        if not cota or not poli or len(poli) < 3:
            continue
        w_px, h_px = bbox(poli)
        a, b = sorted(cota)          # la cota puede venir en cualquier orden
        c, d = sorted([w_px, h_px])  # y el ambiente rotado en el plano
        if a > 0 and b > 0:
            est.append({"ambiente": amb["nombre"], "px_por_m": c / a})
            est.append({"ambiente": amb["nombre"], "px_por_m": d / b})

    if not est:
        return {"px_por_m": None, "estimaciones": [], "cv_pct": None,
                "confiable": False, "motivo": "ningún ambiente trae cota impresa"}

    vals = [e["px_por_m"] for e in est]
    m = mean(vals)
    cv = (pstdev(vals) / m * 100) if m else None
    return {
        "px_por_m": m,
        "estimaciones": est,
        "cv_pct": cv,
        # Si las cotas discrepan más de 8% entre sí, el trazado está mal.
        # Esto se detecta SIN usar el área techada.
        "confiable": cv is not None and cv < 8.0,
    }


def area_m2(poli, px_por_m):
    """Shoelace en píxeles, convertida a m²."""
    n = len(poli)
    s = sum(poli[i][0] * poli[(i + 1) % n][1] - poli[(i + 1) % n][0] * poli[i][1] for i in range(n))
    return abs(s) / 2.0 / (px_por_m ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Validación — aquí, y solo aquí, aparece el área techada
# ─────────────────────────────────────────────────────────────────────────────

def validar(geo, cal, meta):
    ext = geo.get("poligono_exterior_px")
    if not ext or not cal["px_por_m"]:
        return {"ok": False, "motivo": "sin polígono exterior o sin escala"}

    recon = area_m2(ext, cal["px_por_m"])
    real = meta["area_techada_m2"]
    err = (recon - real) / real * 100

    return {
        "area_reconstruida_m2": round(recon, 2),
        "area_techada_real_m2": real,
        "error_pct": round(err, 2),
        "escala_confiable": cal["confiable"],
        "cv_calibracion_pct": round(cal["cv_pct"], 2) if cal["cv_pct"] is not None else None,
        # Umbral: 5% de error de área ≈ ±10 cm en un muro de 6 m. Aceptable para un recorrido.
        "ok": abs(err) < 5.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Grafo de escena final (en METROS — listo para el visor 3D)
# ─────────────────────────────────────────────────────────────────────────────

def construir_grafo(geo, cal, meta, acabados):
    k = cal["px_por_m"]
    if not k:
        return None
    a_m = lambda poli: [[round(x / k, 3), round(y / k, 3)] for x, y in poli]

    return {
        "id": f"GRAU10-{meta['tipo']}",
        "proyecto": "Grau 10 · Av. Miguel Grau 1020, Barranco",
        "desarrolladora": "Edifica",
        "arquitecto": "Lima 1007 Arquitectos",
        "area_techada_m2": meta["area_techada_m2"],
        "dormitorios": meta["dormitorios"],
        "altura_techo_m": ALTURA_TECHO_M,
        "escala_px_por_m": round(k, 3),
        "poligono_exterior_m": a_m(geo["poligono_exterior_px"]),
        "ambientes": [
            {
                "nombre": a["nombre"],
                "tipo": a["tipo"],
                "cota_impresa_m": a.get("cota_impresa_m"),
                "poligono_m": a_m(a["poligono_px"]),
                "muebles": a.get("muebles", []),
            }
            for a in geo.get("ambientes", []) if a.get("poligono_px")
        ],
        "vanos_m": [
            {"tipo": v["tipo"], "extremos_m": a_m(v["extremos_px"])}
            for v in geo.get("vanos", []) if v.get("extremos_px")
        ],
        "acabados": acabados,
        "notas_extractor": geo.get("notas", ""),
    }


# Página 28 del deck. Esto NO se adivina: el desarrollador lo dicta.
ACABADOS_GRAU10 = {
    "piso_sala": "SPC",
    "piso_cocina": "SPC",
    "tablero_cocina": "cuarzo",
    "tablero_bano": "cuarzo",
    "cocina": "equipada: encimera, horno, campana",
    "ventanas": "perfilería de vidrio con reducción acústica",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--paginas", required=True, help="ej. 30-44")
    ap.add_argument("--out", default="./salida")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Falta ANTHROPIC_API_KEY")

    ini, fin = (int(x) for x in args.paginas.split("-"))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cliente = anthropic.Anthropic()
    tabla = []

    for pag in range(ini, fin + 1):
        meta = leer_metadata(args.pdf, pag)
        print(f"→ {meta['tipo']} (p{pag}, {meta['area_techada_m2']} m²) ... ", end="", flush=True)
        try:
            img = extraer_imagen(args.pdf, pag, out / "planos")
            geo = extraer_geometria(cliente, img)
            cal = calibrar(geo)
            val = validar(geo, cal, meta)
            grafo = construir_grafo(geo, cal, meta, ACABADOS_GRAU10)
            if grafo:
                (out / f"{meta['tipo']}.json").write_text(
                    json.dumps(grafo, indent=2, ensure_ascii=False))
            print(f"{val.get('area_reconstruida_m2','—')} m²  "
                  f"err {val.get('error_pct','—')}%  "
                  f"{'OK' if val.get('ok') else 'REVISAR'}")
        except Exception as e:
            val = {"ok": False, "motivo": str(e)[:80]}
            print(f"FALLO: {e}")
        tabla.append({**meta, **val})

    (out / "validacion.json").write_text(json.dumps(tabla, indent=2, ensure_ascii=False))

    # Resumen honesto
    con_err = [t for t in tabla if t.get("error_pct") is not None]
    print("\n" + "=" * 64)
    print(f"tipologías procesadas : {len(tabla)}")
    print(f"con geometría válida  : {len(con_err)}")
    if con_err:
        errs = [abs(t["error_pct"]) for t in con_err]
        print(f"error absoluto medio  : {mean(errs):.2f} %")
        print(f"error máximo          : {max(errs):.2f} %")
        print(f"bajo umbral 5%        : {sum(1 for t in con_err if t['ok'])}/{len(con_err)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
