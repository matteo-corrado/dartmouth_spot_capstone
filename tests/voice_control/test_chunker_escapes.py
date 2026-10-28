"""B2: TTSChunker decodes JSON string escapes before speaking."""
from src.voice_control.spot_tts import TTSChunker


def test_chunker_decodes_escapes():
    out = []; c = TTSChunker(out.append); c.feed('Hi\\nthere \\u2014 friend. '); c.flush()
    assert "u2014" not in " ".join(out) and "\\" not in " ".join(out)
    assert "—" in " ".join(out)


def test_chunker_decodes_quotes_in_json_stream():
    out = []; c = TTSChunker(out.append)
    c.accept('{"actions": [], "response": "She said \\"hi\\" to me."}'); c.flush()
    assert out == ['She said "hi" to me.']
