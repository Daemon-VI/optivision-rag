"""Bundled config presets make the CLI usable from a pip install."""

import pytest

from optivision.config import Config, available_presets


def test_presets_are_bundled():
    assert {"synthetic", "colsmol", "colpali", "qdrant"} <= set(available_presets())


@pytest.mark.parametrize("name", ["synthetic", "synthetic.yaml"])
def test_load_preset_by_name(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no repo checkout around
    assert Config.load(name).encoder.backend == "synthetic"


def test_file_on_disk_wins_over_preset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "synthetic.yaml").write_text("encoder:\n  backend: colpali\n", encoding="utf-8")
    assert Config.load("synthetic.yaml").encoder.backend == "colpali"


def test_unknown_config_lists_presets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="bundled presets"):
        Config.load("nope")
