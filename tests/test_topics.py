"""Проверки карты тем: настоящая карта корректна, а ошибки ловятся."""

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("topics", ROOT / "scripts" / "topics.py")
topics_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(topics_mod)
build = topics_mod.build


def topic(id, level=1, requires=()):
    return {"id": id, "title": id, "level": level, "requires": list(requires), "about": "—"}


def section(*items, id="math/test"):
    return {"id": id, "title": "Тест", "topics": list(items)}


def test_real_map_is_valid():
    _, errors = build(topics_mod.load_plan(), topics_mod.scan_articles())
    assert errors == []


def test_unknown_requirement():
    _, errors = build([section(topic("a", requires=["b"]))], [])
    assert any("неизвестная тема 'b'" in e for e in errors)


def test_requirement_from_later_course():
    _, errors = build([section(topic("a", level=1, requires=["b"]), topic("b", level=3))], [])
    assert any("требует 'b'" in e for e in errors)


def test_cycle():
    _, errors = build([section(topic("a", requires=["b"]), topic("b", requires=["a"]))], [])
    assert any("цикл" in e for e in errors)


def test_article_in_wrong_section():
    _, errors = build([section(topic("a"))], [("physics/x/a.qmd", {"topic": "a"})])
    assert any("относится к разделу math/test" in e for e in errors)


def test_article_must_not_duplicate_plan_fields():
    _, errors = build([section(topic("a"))], [("math/test/a.qmd", {"topic": "a", "level": 1})])
    assert any("задаются в topics.yml" in e for e in errors)


def test_written_topic_gets_link_and_reverse_edges():
    t, errors = build(
        [section(topic("a"), topic("b", requires=["a"]))],
        [("math/test/a.qmd", {"topic": "a"})],
    )
    assert errors == []
    assert t["a"]["href"] == "/math/test/a.qmd"
    assert "href" not in t["b"]
    assert t["a"]["required_by"] == ["b"]
