"""
API-compatible Model Card Toolkit shim for AeroShield FireTwin notebooks.

Official `model-card-toolkit` pulls TensorFlow Data Validation wheels that are
often unavailable (e.g. Python 3.13). This shim implements the subset used by
our Model Card notebooks: scaffold → populate → JSON/HTML export.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Owner:
    name: str = ""
    contact: str = ""


@dataclass
class Version:
    name: str = ""
    date: str = ""
    diff: str = ""


@dataclass
class Reference:
    reference: str = ""


@dataclass
class Graphic:
    name: str = ""
    image: str = ""


@dataclass
class GraphicCollection:
    name: str = ""
    graphics: list[Graphic] = field(default_factory=list)


@dataclass
class Dataset:
    name: str = ""
    description: str = ""
    link: str = ""
    graphics: GraphicCollection | None = None


@dataclass
class PerformanceMetric:
    type: str = ""
    value: str = ""
    slice: str = "overall"


@dataclass
class User:
    description: str = ""


@dataclass
class UseCase:
    description: str = ""


@dataclass
class Limitation:
    description: str = ""


@dataclass
class EthicalConsideration:
    name: str = ""
    description: str = ""
    mitigation_strategy: str = ""


@dataclass
class Risk:
    name: str = ""
    mitigation_strategy: str = ""


@dataclass
class ModelDetails:
    name: str = ""
    overview: str = ""
    owners: list[Owner] = field(default_factory=list)
    version: Version | None = None
    references: list[Reference] = field(default_factory=list)
    path: str = ""


@dataclass
class ModelParameters:
    model_architecture: str = ""
    input_format: str = ""
    output_format: str = ""
    data: list[Dataset] = field(default_factory=list)


@dataclass
class QuantitativeAnalysis:
    performance_metrics: list[PerformanceMetric] = field(default_factory=list)
    graphics: GraphicCollection | None = None


@dataclass
class Considerations:
    users: list[User] = field(default_factory=list)
    use_cases: list[UseCase] = field(default_factory=list)
    limitations: list[Limitation] = field(default_factory=list)
    ethical_considerations: list[EthicalConsideration] = field(default_factory=list)
    risks: list[Risk] = field(default_factory=list)


@dataclass
class ModelCard:
    model_details: ModelDetails = field(default_factory=ModelDetails)
    model_parameters: ModelParameters = field(default_factory=ModelParameters)
    quantitative_analysis: QuantitativeAnalysis = field(default_factory=QuantitativeAnalysis)
    considerations: Considerations = field(default_factory=Considerations)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_html(card: ModelCard) -> str:
    md = card.model_details
    mp = card.model_parameters
    qa = card.quantitative_analysis
    co = card.considerations

    def esc(x: Any) -> str:
        return html.escape(str(x or ""))

    metrics_rows = "".join(
        f"<tr><td>{esc(m.type)}</td><td>{esc(m.value)}</td><td>{esc(m.slice)}</td></tr>"
        for m in qa.performance_metrics
    )
    limits = "".join(f"<li>{esc(x.description)}</li>" for x in co.limitations)
    risks = "".join(
        f"<li><b>{esc(r.name)}</b> — {esc(r.mitigation_strategy)}</li>" for r in co.risks
    )
    refs = "".join(f"<li>{esc(r.reference)}</li>" for r in md.references)
    owners = ", ".join(esc(o.name) for o in md.owners) or "—"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{esc(md.name)}</title>
<style>
body{{font-family:Segoe UI,system-ui,sans-serif;margin:24px;background:#0b1220;color:#f3f6fb;}}
h1,h2{{color:#e8eef7}} table{{border-collapse:collapse;width:100%;margin:12px 0}}
td,th{{border:1px solid #374151;padding:8px;text-align:left}} th{{background:#111827}}
.card{{background:#111827;border:1px solid #374151;border-radius:12px;padding:16px;margin:12px 0}}
a{{color:#93c5fd}}
</style></head><body>
<h1>{esc(md.name)}</h1>
<div class="card"><p>{esc(md.overview)}</p>
<p><b>Owners:</b> {owners}<br/>
<b>Version:</b> {esc(md.version.name if md.version else '')}
 ({esc(md.version.date if md.version else '')})<br/>
<b>Artifact:</b> {esc(md.path)}</p></div>
<div class="card"><h2>Model parameters</h2>
<p><b>Architecture:</b> {esc(mp.model_architecture)}</p>
<p><b>Input:</b> {esc(mp.input_format)}</p>
<p><b>Output:</b> {esc(mp.output_format)}</p></div>
<div class="card"><h2>Quantitative analysis</h2>
<table><thead><tr><th>Metric</th><th>Value</th><th>Slice</th></tr></thead>
<tbody>{metrics_rows}</tbody></table></div>
<div class="card"><h2>Limitations</h2><ul>{limits}</ul>
<h2>Risks</h2><ul>{risks}</ul>
<h2>References</h2><ul>{refs}</ul></div>
</body></html>"""


class ModelCardToolkit:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "model_cards").mkdir(parents=True, exist_ok=True)
        self._card: ModelCard | None = None

    def scaffold_assets(self) -> ModelCard:
        self._card = ModelCard()
        return self._card

    def update_model_card_json(self, model_card: ModelCard | None = None) -> Path:
        card = model_card or self._card
        if card is None:
            raise ValueError("No model card to serialize")
        path = self.output_dir / "model_cards" / "model_card.json"
        path.write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")
        return path

    def export_format(self, model_card: ModelCard | None = None) -> str:
        card = model_card or self._card
        if card is None:
            raise ValueError("No model card to export")
        html_doc = _to_html(card)
        path = self.output_dir / "model_cards" / "model_card.html"
        path.write_text(html_doc, encoding="utf-8")
        return html_doc
