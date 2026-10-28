from src.voice_control import llm_brain


def test_unfilled_template_is_skipped():
    assert "FILL IN" not in llm_brain._KNOWLEDGE_TEXT
    assert "PRIMARY CONTEXT" not in llm_brain._KNOWLEDGE_TEXT


def test_event_pack_first(monkeypatch, tmp_path):
    assert llm_brain._KNOWLEDGE_FILES[0] == "event_knowledge.md"
    src = llm_brain._KNOWLEDGE_DIR
    for name in llm_brain._KNOWLEDGE_FILES[1:]:
        (tmp_path / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "event_knowledge.md").write_text(
        "## Event\n- Name: Test Fair\n- Date: [FILL IN]\n", encoding="utf-8")
    monkeypatch.setattr(llm_brain, "_KNOWLEDGE_DIR", tmp_path)
    text = llm_brain._load_knowledge_packs()
    assert "PRIMARY CONTEXT" in text
    assert "Test Fair" in text and "FILL IN" not in text
    assert text.index("event_knowledge.md") < text.index("thayer_knowledge.md")
