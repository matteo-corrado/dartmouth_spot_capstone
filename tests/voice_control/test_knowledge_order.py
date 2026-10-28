def test_event_pack_first():
    from src.voice_control import llm_brain
    assert llm_brain._KNOWLEDGE_FILES[0] == "event_knowledge.md"
    text = llm_brain._KNOWLEDGE_TEXT
    assert text.index("event_knowledge.md") < text.index("thayer_knowledge.md")
    assert "PRIMARY CONTEXT" in text
