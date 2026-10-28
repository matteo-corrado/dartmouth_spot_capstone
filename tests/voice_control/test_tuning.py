import logging

from src.voice_control import tuning
from src.voice_control.tuning import load_tuning


def test_defaults_when_file_missing(tmp_path):
    t = load_tuning(tmp_path / "nope.yaml")
    assert t.lock_window_deg == 35.0 and t.mic_channel == 0 and t.lock_timeout_s == 20.0


def test_yaml_overrides(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("mic_channel: 1\nmin_snr_db: 8\nbeam_steering: true\n")
    t = load_tuning(p)
    assert t.mic_channel == 1 and t.min_snr_db == 8.0 and t.onset_frames == 3
    assert t.beam_steering is True


def test_unknown_key_warns_and_is_ignored(tmp_path, caplog):
    p = tmp_path / "t.yaml"
    p.write_text("lock_windw_deg: 20\nmic_channel: 1\n")
    with caplog.at_level(logging.WARNING):
        t = load_tuning(p)
    assert "lock_windw_deg" in caplog.text
    assert t.lock_window_deg == 35.0 and t.mic_channel == 1


def test_shipped_yaml_is_all_defaults():
    assert load_tuning() == tuning.Tuning()


def test_get_tuning_cached():
    assert tuning.get_tuning() is tuning.get_tuning()
