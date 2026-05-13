"""Smoke tests for the public-bucket data loader."""
from pathlib import Path

from plasmidrl import data


def test_load_manifest_returns_existing_path():
    p = data.load("manifests/paper_v2_camera_ready.json")
    assert isinstance(p, Path)
    assert p.exists()
    assert p.stat().st_size > 0


def test_load_json_returns_dict():
    obj = data.load_json("manifests/paper_v2_camera_ready.json")
    assert isinstance(obj, dict)
    assert "claims" in obj


def test_load_csv_returns_dataframe():
    df = data.load_csv("evaluation/eight_prompt/RL/qc/qc_summary.csv")
    assert len(df) > 0


def test_list_tree_finds_manifests():
    items = data.list_tree("manifests/")
    assert "manifests/paper_v2_camera_ready.json" in items
