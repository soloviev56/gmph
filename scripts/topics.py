"""Собрать карту тем пособия из заголовков статей.

Каждая статья в math/ и physics/ (кроме страниц разделов index.qmd) обязана иметь в заголовке поля
``topic`` (уникальный идентификатор), ``title``, ``level`` (курс, 1–4)
и ``requires`` (список идентификаторов тем, которые нужно знать заранее).

Скрипт проверяет, что ссылки ведут на существующие темы и в графе нет
циклов, и пишет:
  _generated/topics.json — данные для фильтра filters/topics.lua;
  _generated/map.md      — карта тем для страницы map.qmd.
Запускается Quarto перед каждой сборкой (project.pre-render).
"""

import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
SECTIONS = ("math", "physics")
OUT = ROOT / "_generated"


def front_matter(path):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    return yaml.safe_load(text.split("---", 2)[1]) or {}


def collect():
    topics, errors = {}, []
    for section in SECTIONS:
        for path in sorted((ROOT / section).rglob("*.qmd")):
            if path.name.startswith("_") or path.name == "index.qmd":
                continue
            meta = front_matter(path) or {}
            rel = path.relative_to(ROOT).as_posix()
            missing = [k for k in ("topic", "title", "level", "requires") if k not in meta]
            if missing:
                errors.append(f"{rel}: нет полей {', '.join(missing)}")
                continue
            tid = meta["topic"]
            if tid in topics:
                errors.append(f"{rel}: тема '{tid}' уже есть в {topics[tid]['href']}")
                continue
            topics[tid] = {
                "title": meta["title"],
                "href": "/" + rel,
                "level": int(meta["level"]),
                "requires": list(meta["requires"] or []),
                "required_by": [],
            }
    for tid, t in topics.items():
        for req in t["requires"]:
            if req not in topics:
                errors.append(f"{t['href']}: требуется неизвестная тема '{req}'")
            else:
                topics[req]["required_by"].append(tid)
    errors += find_cycles(topics)
    return topics, errors


def find_cycles(topics):
    state, errors = {}, []

    def visit(tid, path):
        state[tid] = "open"
        for req in topics[tid]["requires"]:
            if req not in topics:
                continue
            if state.get(req) == "open":
                cycle = path[path.index(req):] + [req]
                errors.append("цикл зависимостей: " + " → ".join(cycle))
            elif req not in state:
                visit(req, path + [req])
        state[tid] = "done"

    for tid in topics:
        if tid not in state:
            visit(tid, [tid])
    return errors


def map_markdown(topics):
    lines = ["```{mermaid}", "flowchart TD"]
    for tid, t in topics.items():
        lines.append(f'  {tid}["{t["title"]}<br/><small>{t["level"]} курс</small>"]')
        lines.append(f'  click {tid} "{t["href"].lstrip("/").removesuffix(".qmd")}.html"')
    for tid, t in topics.items():
        for req in t["requires"]:
            lines.append(f"  {req} --> {tid}")
    lines.append("```")
    lines.append("")
    for level in sorted({t["level"] for t in topics.values()}):
        lines.append(f"### {level} курс")
        lines.append("")
        for t in sorted((t for t in topics.values() if t["level"] == level), key=lambda t: t["title"]):
            lines.append(f"- [{t['title']}]({t['href']})")
        lines.append("")
    return "\n".join(lines)


def main():
    topics, errors = collect()
    if errors:
        print("Ошибки в карте тем:", *errors, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    OUT.mkdir(exist_ok=True)
    (OUT / "topics.json").write_text(json.dumps(topics, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "map.md").write_text(map_markdown(topics), encoding="utf-8")
    print(f"Карта тем: {len(topics)} статей")


if __name__ == "__main__":
    main()
