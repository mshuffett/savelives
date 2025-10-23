import json
from pathlib import Path

from omegaconf import OmegaConf


def test_config_loads():
    cfg = OmegaConf.load("config/salm_lora.yaml")
    assert cfg.name == "canary_qwen_medical_lora"
    assert cfg.model.restore_from_path
    assert cfg.data.train_ds.batch_size > 0


def test_manifest_fields(tmp_path: Path):
    # Create a small dummy manifest line and validate keys exist
    d = tmp_path / "val_manifest.json"
    obj = {"audio_filepath": "x.wav", "duration": 1.0, "text": "hi"}
    d.write_text(json.dumps(obj) + "\n", encoding="utf-8")
    # Read back
    with d.open("r", encoding="utf-8") as f:
        line = f.readline()
        loaded = json.loads(line)
    assert set(loaded.keys()) == {"audio_filepath", "duration", "text"}

