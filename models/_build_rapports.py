"""Build visually designed RAPPORT_MODELE.pdf for every model folder."""
from __future__ import annotations

import base64
import html
import json
import mimetypes
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_css() -> str:
    return (ROOT / "_rapport_style.css").read_text(encoding="utf-8")


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    raw = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def esc(s: str) -> str:
    return html.escape(s or "")


def inline_md(text: str) -> str:
    text = esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def parse_sections(md: str) -> tuple[str, list[tuple[str, str]]]:
    lines = md.splitlines()
    title = "Rapport modèle"
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break
    sections: list[tuple[str, str]] = []
    cur_title = None
    buf: list[str] = []
    for line in lines:
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            if cur_title is not None:
                sections.append((cur_title, "\n".join(buf).strip()))
            cur_title = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    if cur_title is not None:
        sections.append((cur_title, "\n".join(buf).strip()))
    return title, sections


def table_html(rows: list[str]) -> str:
    parsed = []
    for r in rows:
        if re.match(r"^\|?\s*-+", r):
            continue
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        parsed.append(cells)
    if not parsed:
        return ""
    head, body = parsed[0], parsed[1:]
    thead = "<thead><tr>" + "".join(f"<th>{inline_md(c)}</th>" for c in head) + "</tr></thead>"
    tb = []
    for row in body:
        while len(row) < len(head):
            row.append("")
        tb.append("<tr>" + "".join(f"<td>{inline_md(c)}</td>" for c in row[: len(head)]) + "</tr>")
    return f"<table>{thead}<tbody>{''.join(tb)}</tbody></table>"


def render_block(body: str) -> str:
    out: list[str] = []
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("### "):
            out.append(f"<h3>{inline_md(line[4:].strip())}</h3>")
            i += 1
            continue
        if line.strip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(table_html(rows))
            continue
        if re.match(r"^\d+\.\s", line.strip()):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i].strip()):
                items.append(f"<li>{inline_md(re.sub(r'^\\d+\\.\\s', '', lines[i].strip()))}</li>")
                i += 1
            out.append("<ol>" + "".join(items) + "</ol>")
            continue
        if line.strip().startswith(("- ", "* ")):
            items = []
            while i < len(lines) and lines[i].strip().startswith(("- ", "* ")):
                items.append(f"<li>{inline_md(lines[i].strip()[2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        if line.strip().startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(esc(lines[i]))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(code_lines) + "</code></pre>")
            continue
        paras = [line]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].startswith("#")
            and not lines[i].strip().startswith("|")
            and not lines[i].strip().startswith(("- ", "* "))
            and not re.match(r"^\d+\.\s", lines[i].strip())
            and not lines[i].startswith("### ")
        ):
            paras.append(lines[i])
            i += 1
        text = " ".join(p.strip() for p in paras)
        if text.strip().startswith(("⚠️", "⚠")):
            out.append(f'<div class="callout warn">{inline_md(text)}</div>')
        else:
            out.append(f"<p>{inline_md(text)}</p>")
    return "\n".join(out)


def status_class(statut: str) -> str:
    s = (statut or "").upper()
    if "SAMPLE" in s:
        return "sample"
    if "SYNTH" in s:
        return "synth"
    if "MISSING" in s or "NOT_COMPUTED" in s or "UNAVAILABLE" in s:
        return "missing"
    return ""


JUNK_IMG_TOKENS = (
    "banner", "placeholder", "local_status", "sample_banner",
    "dataset_missing", "metrics_placeholder", "sample_gt_box",
)


def is_junk_image(path: Path) -> bool:
    name = path.name.lower()
    return any(tok in name for tok in JUNK_IMG_TOKENS)


def collect_images(folder: Path) -> list[Path]:
    found: list[Path] = []
    for sub in ("illustrations", "metrics"):
        d = folder / sub
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS and p.is_file() and not is_junk_image(p):
                found.append(p)
    # de-dupe by name preferring illustrations/
    by_name: dict[str, Path] = {}
    for p in found:
        by_name.setdefault(p.name.lower(), p)
        if "illustrations" in str(p).replace("\\", "/"):
            by_name[p.name.lower()] = p
    return list(by_name.values())


def rank_images(paths: list[Path]) -> list[Path]:
    keys = (
        "confusion", "results", "pr_curve", "boxpr", "boxf1", "feature",
        "class_dist", "metric", "author", "synthetic", "split", "dataset",
        "overlay", "pred", "visdrone", "cylinder", "tank", "fire_", "smoke_",
    )
    scored = []
    for p in paths:
        name = p.name.lower()
        if is_junk_image(p):
            continue
        score = 0
        for i, k in enumerate(keys):
            if k in name:
                score = 100 - i
                break
        if p.suffix.lower() == ".png":
            score += 2
        scored.append((score, p.name.lower(), p))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [p for _, _, p in scored]


def status_notice_html(meta: dict, folder: Path) -> str:
    """Inline note matching rapport style — never a giant fake PNG banner."""
    statut = (meta.get("statut") or "").upper()
    msg = None
    if "SAMPLE" in statut:
        msg = (
            "<strong>SAMPLE</strong> — aucune performance REAL revendiquée. "
            "Slot thermique / UI uniquement (<code>metrics_valid=false</code>)."
        )
        cls = "callout warn"
    elif "MISSING" in statut or "DATASET_MISSING" in statut:
        msg = (
            "<strong>DATASET_MISSING</strong> — pas de paires indoor licenciées ; "
            "métriques volontairement <code>null</code>."
        )
        cls = "callout warn"
    elif "SYNTH" in statut:
        msg = (
            "<strong>SYNTHETIC_MODEL</strong> — scores = accord avec règles jouet, "
            "pas de physique incendie."
        )
        cls = "callout"
    else:
        # floorplan local status
        tm = folder / "metrics" / "test_metrics.json"
        if tm.exists():
            data = json.loads(tm.read_text(encoding="utf-8"))
            if data.get("status") == "NOT_COMPUTED":
                msg = (
                    "<strong>Éval locale NOT_COMPUTED</strong> — CubiCasa non préparé ici ; "
                    "mIoU upstream auteur cité seulement, pas comme mesure held-out."
                )
                cls = "callout warn"
    if not msg:
        return ""
    return f'<div class="{cls}">{msg}</div>'


def fig_html(path: Path, height_note: str = "") -> str:
    return (
        f'<figure><img src="{data_uri(path)}" alt="{esc(path.name)}"/>'
        f"<figcaption>{esc(path.name)}{esc(height_note)}</figcaption></figure>"
    )


def visual_band_html(images: list[Path]) -> str:
    if not images:
        return ""
    ranked = rank_images(images)
    main = ranked[0]
    side = ranked[1:3]
    side_html = "".join(fig_html(p) for p in side) if side else fig_html(main)
    return f"""
    <div class="visual-band">
      <div class="visual-main">{fig_html(main)}</div>
      <div class="visual-side">{side_html}</div>
    </div>
    """


def metric_figures_html(images: list[Path], limit: int = 4) -> str:
    ranked = rank_images(images)
    chartish = [
        p for p in ranked
        if any(k in p.name.lower() for k in (
            "confusion", "results", "pr", "f1", "metric", "feature",
            "class", "author", "synthetic", "map", "split", "pred", "visdrone",
        ))
    ]
    chosen = (chartish or ranked)[:limit]
    if not chosen:
        return ""
    return '<div class="metric-figures">' + "".join(fig_html(p) for p in chosen) + "</div>"


def kpis_from_folder(folder: Path, meta: dict) -> list[tuple[str, str, str]]:
    """Prefer metrics.json numbers; fall back to caracteristiques string."""
    kpis: list[tuple[str, str, str]] = []

    for cand in (folder / "metrics" / "metrics.json", folder / "metrics" / "test_metrics.json"):
        if not cand.exists():
            continue
        data = json.loads(cand.read_text(encoding="utf-8"))
        # RGB person
        if isinstance(data.get("test"), dict) and "mAP50" in data["test"]:
            t = data["test"]
            return [
                ("mAP50", f"{t['mAP50']:.3f}", "test"),
                ("Precision", f"{t['precision']:.3f}", "test"),
                ("Recall", f"{t['recall']:.3f}", "test"),
                ("mAP50-95", f"{t['mAP50_95']:.3f}", "test"),
            ]
        # equipment
        if "test_map50" in data:
            return [
                ("mAP50", f"{data['test_map50']:.3f}", "test"),
                ("mAP50-95", f"{data['test_map50_95']:.3f}", "test"),
                ("Precision", f"{data['test_precision_macro']:.3f}", "macro"),
                ("Recall", f"{data['test_recall_macro']:.3f}", "macro"),
            ]
        # gas / generic accuracy
        if "accuracy" in data and "macro_f1" in data:
            return [
                ("Accuracy", f"{float(data['accuracy']):.3f}", "primary"),
                ("Macro F1", f"{float(data['macro_f1']):.3f}", "primary"),
                ("Macro P", f"{float(data.get('macro_precision', 0) or 0):.3f}" if data.get("macro_precision") else "—", "primary"),
                ("Macro R", f"{float(data.get('macro_recall', 0) or 0):.3f}" if data.get("macro_recall") else "—", "primary"),
            ]
        # fire smoke author
        ar = data.get("author_reported_metrics")
        if isinstance(ar, dict) and "mAP50" in ar:
            return [
                ("mAP50", str(ar["mAP50"]), "auteur"),
                ("mAP50-95", str(ar.get("mAP50-95", "—")), "auteur"),
                ("Precision", str(ar.get("precision", "—")), "auteur"),
                ("Recall", str(ar.get("recall", "—")), "auteur"),
            ]
        # floorplan / damage / sample — no fake KPI circus; empty list → builder skips strip
        if data.get("status") in {"NOT_COMPUTED", "DATASET_MISSING"} or data.get("performance_claim") is False:
            return []
        if data.get("status"):
            return []

    # parse caracteristiques metriques string
    raw = re.sub(r"\([^)]*\)", "", meta.get("metriques") or "")
    parts = re.split(r"\s*[·•|]\s*", raw)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.match(r"^(.+?)\s+([0-9]+(?:\.[0-9]+)?)\s*$", p)
        if m:
            kpis.append((m.group(1).strip(), m.group(2).strip(), ""))
    while len(kpis) < 4:
        fillers = [
            ("Algo", (meta.get("algo") or "—")[:16], ""),
            ("Statut", (meta.get("statut") or "—")[:14], ""),
            ("Dataset", "ok" if meta.get("dataset") else "—", ""),
            ("Weights", "oui" if meta.get("checkpoint") else "—", ""),
        ]
        kpis.append(fillers[len(kpis)])
    return kpis[:4]


def build_html(folder: Path) -> str:
    md_path = folder / "RAPPORT_MODELE.md"
    meta_path = folder / "caracteristiques.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    title, sections = parse_sections(md_path.read_text(encoding="utf-8"))

    hero_sub = meta.get("tache") or ""
    for name, body in sections:
        if name.lower().startswith("objectif"):
            first = re.split(r"[.\n]", body.strip())[0].strip()
            if first:
                hero_sub = first[:200]
            break

    statut = meta.get("statut", "")
    kpis = kpis_from_folder(folder, meta)
    images = collect_images(folder)

    if kpis:
        kpi_html = (
            '<div class="kpis">'
            + "".join(
                f'<div class="kpi"><div class="label">{esc(a)}</div>'
                f'<div class="value">{esc(b)}</div>'
                f'<div class="hint">{esc(c)}</div></div>'
                for a, b, c in kpis
            )
            + "</div>"
        )
    else:
        kpi_html = ""

    chips = [
        ("ID", meta.get("id", folder.name)),
        ("Algo", meta.get("algo", "")),
        ("Dataset", meta.get("dataset", "")),
        ("Checkpoint", meta.get("checkpoint", "")),
    ]
    chip_html = "".join(
        f'<span class="chip"><strong>{esc(k)}</strong> {esc(v)}</span>' for k, v in chips if v
    )

    # split intro sections into 2-col when present
    intro_names = {"objectif", "architecture", "pourquoi ce modèle", "pourquoi yolov8n", "pourquoi random forest", "pourquoi ce choix", "pourquoi resnet-18 pairé"}
    intro: list[tuple[str, str]] = []
    rest: list[tuple[str, str]] = []
    for name, body in sections:
        key = name.lower().strip()
        if key in intro_names or key.startswith("pourquoi"):
            intro.append((name, body))
        else:
            rest.append((name, body))

    intro_html = ""
    if intro:
        cards = []
        for idx, (name, body) in enumerate(intro[:2], start=1):
            cards.append(
                f'<section class="section"><h2><span class="num">{idx:02d}</span>{esc(name)}</h2>{render_block(body)}</section>'
            )
        # leftover intro sections as normal
        extra = ""
        for name, body in intro[2:]:
            extra += f'<section class="section"><h2>{esc(name)}</h2>{render_block(body)}</section>'
        intro_html = f'<div class="intro-grid">{"".join(cards)}</div>{extra}'

    section_html = []
    n0 = 1 + min(2, len(intro))
    for i, (name, body) in enumerate(rest):
        block = render_block(body)
        lower = name.lower()
        if any(k in lower for k in ("métrique", "metrique", "évaluation", "evaluation", "résultat")):
            block = metric_figures_html(images, 4) + block
        elif "illustration" in lower:
            block = metric_figures_html(images, 4) + block
        section_html.append(
            f'<section class="section"><h2><span class="num">{n0 + i:02d}</span>{esc(name)}</h2>{block}</section>'
        )

    # if no dedicated metrics section got figures, add a visual results section
    has_metric_section = any(
        any(k in n.lower() for k in ("métrique", "metrique", "illustration")) for n, _ in rest
    )
    results_section = ""
    if images and not has_metric_section:
        results_section = (
            f'<section class="section"><h2><span class="num">{n0 + len(rest):02d}</span>Résultats visuels</h2>'
            f"{metric_figures_html(images, 4)}</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="fr"><head>
<meta charset="utf-8"/>
<title>{esc(title)}</title>
<style>{load_css()}</style>
</head><body>
<div class="page">
  <header class="hero">
    <div class="brand-row">
      <div class="brand">AeroShield · FireTwin AI</div>
      <div class="model-id">{esc(meta.get('id', folder.name))}</div>
    </div>
    <h1>{esc(meta.get('nom') or title)}</h1>
    <p class="hero-sub">{esc(hero_sub)}</p>
    <div class="hero-meta">
      <span class="status-pill {status_class(statut)}">{esc(statut or 'Rapport modèle')}</span>
    </div>
    <span class="hero-rule" aria-hidden="true"></span>
  </header>

  {kpi_html}
  {status_notice_html(meta, folder)}
  {visual_band_html(images)}
  <div class="meta">{chip_html}</div>

  {intro_html}
  {''.join(section_html)}
  {results_section}
  <p class="footer-note">Prototype hackathon FireTwin / AeroShield — ne remplace pas une certification incendie ni une inspection humaine.</p>
</div>
</body></html>"""


def write_pdf(html_doc: str, pdf_path: Path) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html_doc, wait_until="load")
        page.wait_for_timeout(600)
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={"top": "10mm", "bottom": "12mm", "left": "9mm", "right": "9mm"},
        )
        browser.close()


def main() -> None:
    folders = sorted(
        d for d in ROOT.iterdir()
        if d.is_dir() and (d / "RAPPORT_MODELE.md").exists() and re.match(r"^\d{2}_", d.name)
    )
    for folder in folders:
        html_doc = build_html(folder)
        (folder / "RAPPORT_MODELE.html").write_text(html_doc, encoding="utf-8")
        pdf_path = folder / "RAPPORT_MODELE.pdf"
        try:
            write_pdf(html_doc, pdf_path)
            print("OK", folder.name, f"{pdf_path.stat().st_size // 1024} KB")
        except Exception as e:
            print("FAIL", folder.name, e)


if __name__ == "__main__":
    main()
