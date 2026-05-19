from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DOMAIN_COLORS = {
    "java": "#2563eb",
    "mysql": "#16a34a",
    "JVM": "#dc2626",
    "os": "#7c3aed",
    "cn": "#0891b2",
    "data_structure": "#ea580c",
}

DOMAIN_LABELS = {
    "cn": "Computer Network",
    "data_structure": "Data Structure",
    "java": "Java",
    "JVM": "JVM",
    "mysql": "MySQL",
    "os": "Operating System",
}

RELATION_LABELS = {
    "depends_on": "depends_on",
    "has_part": "has_part",
    "part_of": "part_of",
    "used_for": "used_for",
    "is_a": "is_a",
    "implements": "implements",
    "alias_of": "alias_of",
    "related_to": "related_to",
    "extends": "extends",
    "causes": "causes",
}

RELATION_COLORS = {
    "depends_on": "#475569",
    "has_part": "#ea580c",
    "part_of": "#f97316",
    "used_for": "#0891b2",
    "is_a": "#16a34a",
    "implements": "#7c3aed",
    "alias_of": "#dc2626",
    "related_to": "#64748b",
    "extends": "#b45309",
    "causes": "#be123c",
}


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("interview-guide/app/src/main/resources/fonts/ZhuqueFangsong-Regular.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def find_graph_json(project_root: Path) -> Path:
    matches = list(project_root.glob("**/graph_store/graph.json"))
    if not matches:
        raise FileNotFoundError("graph.json not found")
    return matches[0]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def short_text(text: str, limit: int = 18) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def wrap_lines(text: str, width: int) -> list[str]:
    text = str(text or "")
    if not text:
        return []
    lines: list[str] = []
    current = ""
    for ch in text:
        trial = current + ch
        if len(trial) <= width:
            current = trial
        else:
            lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


def draw_text_block(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill: str, line_chars: int, line_gap: int = 4) -> int:
    x, y = xy
    lines = wrap_lines(text, line_chars)
    step = (font.size if hasattr(font, "size") else 16) + line_gap
    for index, line in enumerate(lines):
        draw.text((x, y + index * step), line, font=font, fill=fill)
    return len(lines) * step


def build_domain_cards(domains: list[str], image_w: int, top_y: int, bottom_margin: int) -> dict[str, tuple[int, int, int, int]]:
    cols = 3
    rows = math.ceil(len(domains) / cols)
    outer_x = 40
    gap_x = 28
    gap_y = 28
    card_w = (image_w - outer_x * 2 - gap_x * (cols - 1)) // cols
    usable_h = 3200 - top_y - bottom_margin
    card_h = (usable_h - gap_y * (rows - 1)) // rows
    cards: dict[str, tuple[int, int, int, int]] = {}
    for index, domain in enumerate(domains):
        col = index % cols
        row = index // cols
        x1 = outer_x + col * (card_w + gap_x)
        y1 = top_y + row * (card_h + gap_y)
        cards[domain] = (x1, y1, x1 + card_w, y1 + card_h)
    return cards


def choose_domain_nodes(payload: dict, domain: str, limit: int = 18) -> list[dict]:
    nodes = [node for node in payload["nodes"] if node["domain"] == domain]
    nodes.sort(key=lambda item: (-item.get("mentionCount", 0), item["name"]))
    return nodes[:limit]


def choose_domain_edges(payload: dict, domain: str, node_ids: set[str], limit: int = 18) -> list[dict]:
    edges = [
        edge for edge in payload["edges"]
        if edge["domain"] == domain and edge["source"] in node_ids and edge["target"] in node_ids
    ]
    edges.sort(key=lambda item: (-float(item.get("confidence") or 0.0), item["relation"], item["source"]))
    return edges[:limit]


def choose_domain_attributes(payload: dict, domain: str, entity_names: set[str], limit: int = 6) -> list[dict]:
    attrs = [
        attr for attr in payload.get("attributes", [])
        if attr["domain"] == domain and attr["entity"] in entity_names
    ]
    attrs.sort(key=lambda item: (-(item.get("valueCount") or 1), -(item.get("confidence") or 0.0), item["entity"]))
    return attrs[:limit]


def compute_positions(nodes: list[dict], graph_box: tuple[int, int, int, int]) -> dict[str, tuple[float, float, float]]:
    x1, y1, x2, y2 = graph_box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    rx = (x2 - x1) / 2 - 36
    ry = (y2 - y1) / 2 - 36
    positions: dict[str, tuple[float, float, float]] = {}
    for index, node in enumerate(nodes):
        radius = clamp(9 + math.log(node.get("mentionCount", 0) + 1, 2) * 2.5, 9, 26)
        if index == 0:
            positions[node["id"]] = (cx, cy, radius)
            continue
        angle = index * 2.399963229728653
        ring_scale = min(1.0, 0.18 + math.sqrt(index) / max(math.sqrt(len(nodes)), 1))
        px = cx + math.cos(angle) * rx * ring_scale
        py = cy + math.sin(angle) * ry * ring_scale
        positions[node["id"]] = (px, py, radius)
    return positions


def draw_arrow(draw: ImageDraw.ImageDraw, p1: tuple[float, float], p2: tuple[float, float], color: str) -> None:
    draw.line((p1[0], p1[1], p2[0], p2[1]), fill=color, width=2)
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    size = 8
    left = (
        p2[0] - size * math.cos(angle) + size * 0.6 * math.sin(angle),
        p2[1] - size * math.sin(angle) - size * 0.6 * math.cos(angle),
    )
    right = (
        p2[0] - size * math.cos(angle) - size * 0.6 * math.sin(angle),
        p2[1] - size * math.sin(angle) + size * 0.6 * math.cos(angle),
    )
    draw.polygon([p2, left, right], fill=color)


def draw_metric_chip(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, value: str, color: str, font, small_font) -> None:
    x, y = xy
    draw.rounded_rectangle((x, y, x + 180, y + 56), radius=18, fill="#ffffff", outline=color, width=2)
    draw.text((x + 14, y + 10), label, font=small_font, fill="#64748b")
    draw.text((x + 14, y + 28), value, font=font, fill="#0f172a")


def render_domain_card(draw: ImageDraw.ImageDraw, payload: dict, domain: str, card: tuple[int, int, int, int], fonts: dict[str, object]) -> None:
    x1, y1, x2, y2 = card
    color = DOMAIN_COLORS.get(domain, "#64748b")
    title = DOMAIN_LABELS.get(domain, domain)
    card_nodes = choose_domain_nodes(payload, domain, limit=18)
    node_ids = {node["id"] for node in card_nodes}
    edges = choose_domain_edges(payload, domain, node_ids, limit=18)
    attrs = choose_domain_attributes(payload, domain, {node["name"] for node in card_nodes}, limit=5)

    draw.rounded_rectangle((x1, y1, x2, y2), radius=30, fill="#ffffff", outline=color, width=3)
    draw.rounded_rectangle((x1 + 16, y1 + 16, x2 - 16, y1 + 80), radius=22, fill=color + "1A", outline=color + "55", width=2)
    draw.text((x1 + 30, y1 + 28), title, font=fonts["h2"], fill=color)

    relation_counter = Counter(edge["relation"] for edge in edges)
    top_relation = relation_counter.most_common(1)[0][0] if relation_counter else "-"
    draw_metric_chip(draw, (x2 - 580, y1 + 22), "Nodes", str(len(card_nodes)), color, fonts["metric"], fonts["small"])
    draw_metric_chip(draw, (x2 - 390, y1 + 22), "Edges", str(len(edges)), color, fonts["metric"], fonts["small"])
    draw_metric_chip(draw, (x2 - 200, y1 + 22), "Top Rel", top_relation, color, fonts["small"], fonts["tiny"])

    graph_box = (x1 + 20, y1 + 100, x1 + 420, y2 - 24)
    list_box = (x1 + 440, y1 + 100, x2 - 20, y2 - 24)
    draw.rounded_rectangle(graph_box, radius=24, fill="#f8fafc", outline="#dbeafe", width=2)
    positions = compute_positions(card_nodes, graph_box)

    for edge in edges:
        if edge["source"] not in positions or edge["target"] not in positions:
            continue
        sx, sy, sr = positions[edge["source"]]
        tx, ty, tr = positions[edge["target"]]
        color_line = RELATION_COLORS.get(edge["relation"], "#94a3b8")
        dx = tx - sx
        dy = ty - sy
        dist = max(math.hypot(dx, dy), 1)
        start = (sx + dx / dist * sr, sy + dy / dist * sr)
        end = (tx - dx / dist * tr, ty - dy / dist * tr)
        draw_arrow(draw, start, end, color_line)

    for node in card_nodes:
        px, py, radius = positions[node["id"]]
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline="#ffffff", width=3)
        label = short_text(node["name"], 10)
        label_y = py + radius + 6
        draw.text((px - radius, label_y), label, font=fonts["label"], fill="#0f172a")

    list_x1, list_y1, list_x2, list_y2 = list_box
    draw.rounded_rectangle(list_box, radius=24, fill="#f8fafc", outline="#e2e8f0", width=2)
    draw.text((list_x1 + 18, list_y1 + 14), "Top Nodes", font=fonts["h3"], fill="#0f172a")

    cursor_y = list_y1 + 52
    top_nodes = card_nodes[:8]
    for index, node in enumerate(top_nodes, start=1):
        text = f"{index}. {short_text(node['name'], 16)} ({node.get('mentionCount', 0)})"
        draw.text((list_x1 + 20, cursor_y), text, font=fonts["small"], fill="#334155")
        cursor_y += 26

    cursor_y += 8
    draw.text((list_x1 + 18, cursor_y), "Core Edges", font=fonts["h3"], fill="#0f172a")
    cursor_y += 38
    for edge in edges[:5]:
        rel = RELATION_LABELS.get(edge["relation"], edge["relation"])
        text = f"{short_text(edge['sourceName'], 12)} -> {rel} -> {short_text(edge['targetName'], 12)}"
        draw.text((list_x1 + 20, cursor_y), text, font=fonts["small"], fill=RELATION_COLORS.get(edge["relation"], "#334155"))
        cursor_y += 24

    cursor_y += 8
    draw.text((list_x1 + 18, cursor_y), "Representative Attributes", font=fonts["h3"], fill="#0f172a")
    cursor_y += 38
    if not attrs:
        draw.text((list_x1 + 20, cursor_y), "No attributes extracted", font=fonts["small"], fill="#64748b")
    for attr in attrs:
        head = f"{short_text(attr['entity'], 10)} / {short_text(attr['attributeName'], 8)}"
        body = short_text(attr["attributeValue"], 22)
        draw.text((list_x1 + 20, cursor_y), head, font=fonts["small"], fill=color)
        cursor_y += 22
        draw.text((list_x1 + 28, cursor_y), body, font=fonts["tiny"], fill="#334155")
        cursor_y += 26


def render_summary(draw: ImageDraw.ImageDraw, payload: dict, fonts: dict[str, object], image_w: int) -> int:
    draw.rounded_rectangle((24, 24, image_w - 24, 220), radius=34, fill="#ffffff", outline="#e2e8f0", width=2)
    draw.text((46, 38), "Java Knowledge Graph Poster", font=fonts["h1"], fill="#0f172a")
    draw.text((48, 92), "Static overview with domain-level subgraphs, key relations, and representative attributes.", font=fonts["body"], fill="#475569")

    stats = payload.get("stats", {})
    chips = [
        ("Nodes", str(stats.get("nodeCount", 0)), "#2563eb"),
        ("Edges", str(stats.get("edgeCount", 0)), "#dc2626"),
        ("Attributes", str(stats.get("attributeCount", 0)), "#16a34a"),
        ("Domains", str(stats.get("domainCount", 0)), "#7c3aed"),
    ]
    start_x = 48
    for index, (label, value, color) in enumerate(chips):
        draw_metric_chip(draw, (start_x + index * 196, 132), label, value, color, fonts["metric"], fonts["small"])

    relation_counter = Counter(edge["relation"] for edge in payload["edges"])
    top_relations = relation_counter.most_common(5)
    legend_x = image_w - 650
    draw.rounded_rectangle((legend_x, 34, image_w - 42, 190), radius=26, fill="#f8fafc", outline="#e2e8f0", width=2)
    draw.text((legend_x + 20, 48), "Top Relation Types", font=fonts["h3"], fill="#0f172a")
    for index, (relation, count) in enumerate(top_relations):
        y = 88 + index * 20
        draw.line((legend_x + 22, y + 8, legend_x + 54, y + 8), fill=RELATION_COLORS.get(relation, "#64748b"), width=3)
        draw.text((legend_x + 66, y), f"{relation}: {count}", font=fonts["small"], fill="#334155")
    return 244


def render_footer(draw: ImageDraw.ImageDraw, payload: dict, fonts: dict[str, object], image_w: int, start_y: int) -> None:
    box = (24, start_y, image_w - 24, start_y + 250)
    draw.rounded_rectangle(box, radius=28, fill="#ffffff", outline="#e2e8f0", width=2)
    draw.text((46, start_y + 20), "Global Highlights", font=fonts["h2"], fill="#0f172a")

    top_nodes = sorted(payload["nodes"], key=lambda item: (-item.get("mentionCount", 0), item["name"]))[:12]
    top_attrs = sorted(payload.get("attributes", []), key=lambda item: (-(item.get("valueCount") or 1), -(item.get("confidence") or 0.0)))[:8]

    draw.text((48, start_y + 64), "Most Frequent Nodes", font=fonts["h3"], fill="#334155")
    x = 48
    y = start_y + 98
    for index, node in enumerate(top_nodes, start=1):
        draw.text((x, y), f"{index}. [{node['domain']}] {short_text(node['name'], 16)} ({node.get('mentionCount', 0)})", font=fonts["small"], fill=DOMAIN_COLORS.get(node["domain"], "#334155"))
        y += 24
        if index == 6:
            x = 690
            y = start_y + 98

    draw.text((1320, start_y + 64), "Representative Attributes", font=fonts["h3"], fill="#334155")
    y = start_y + 98
    for attr in top_attrs:
        line = f"[{attr['domain']}] {short_text(attr['entity'], 12)} / {short_text(attr['attributeName'], 8)} = {short_text(attr['attributeValue'], 26)}"
        draw.text((1320, y), line, font=fonts["small"], fill="#334155")
        y += 24


def render_graph_image(payload: dict, output_path: Path) -> None:
    image_w, image_h = 4200, 3200
    image = Image.new("RGB", (image_w, image_h), "#f1f5f9")
    draw = ImageDraw.Draw(image, "RGBA")

    fonts = {
        "h1": load_font(52),
        "h2": load_font(30),
        "h3": load_font(22),
        "metric": load_font(26),
        "body": load_font(22),
        "small": load_font(18),
        "tiny": load_font(16),
        "label": load_font(15),
    }

    top_y = render_summary(draw, payload, fonts, image_w)
    domains = sorted({node["domain"] for node in payload["nodes"]}, key=lambda item: DOMAIN_LABELS.get(item, item))
    cards = build_domain_cards(domains, image_w, top_y, bottom_margin=300)
    for domain in domains:
        render_domain_card(draw, payload, domain, cards[domain], fonts)

    render_footer(draw, payload, fonts, image_w, start_y=image_h - 276)
    image.save(output_path)


def main() -> None:
    project_root = Path.cwd()
    graph_path = find_graph_json(project_root)
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    output_path = graph_path.with_name("knowledge_graph_poster.png")
    render_graph_image(payload, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
