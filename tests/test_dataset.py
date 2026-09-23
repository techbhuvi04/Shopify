"""Catalog validation, preprocessing, leakage-safe splits and config validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import DatasetError, load_catalog, split_indices
from src.data.preprocessing import clean_title, load_image
from src.utils.config import ConfigError, load_config


def test_clean_title():
    assert clean_title("[PROMO] Red  T-Shirt &amp; Cap free shipping") == "red t-shirt & cap"
    assert clean_title(None) == "" and clean_title(float("nan")) == ""
    assert clean_title("ＲＥＤ Shirt") == "red shirt"  # NFKC full-width -> ASCII


def test_load_image_handles_bad_input(tmp_path):
    assert load_image(tmp_path / "nope.jpg") is None
    assert load_image(b"not an image") is None
    assert load_image(b"") is None


def test_catalog_flags_missing_image_and_empty_title(tiny_catalog_csv):
    df = load_catalog(tiny_catalog_csv).df
    row = df.set_index("product_id")
    assert not row.loc["T900", "has_image"] and row.loc["T900", "has_text"]
    assert row.loc["T901", "has_image"] and not row.loc["T901", "has_text"]
    assert row.loc["T001", "has_image"] and row.loc["T001", "has_text"]


def test_catalog_validation_errors(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_catalog(tmp_path / "missing.csv")
    pd.DataFrame({"product_id": ["a"], "title": ["x"]}).to_csv(tmp_path / "a.csv", index=False)
    with pytest.raises(DatasetError, match="missing required columns"):
        load_catalog(tmp_path / "a.csv")
    pd.DataFrame({"product_id": ["a", "a"], "title": ["x", "y"], "image_path": ["", ""]}).to_csv(
        tmp_path / "b.csv", index=False)
    with pytest.raises(DatasetError, match="duplicate"):
        load_catalog(tmp_path / "b.csv")


@pytest.mark.parametrize("strategy", ["group_shuffle", "group_kfold"])
def test_group_split_has_no_group_leakage(tiny_catalog_csv, strategy):
    catalog = load_catalog(tiny_catalog_csv)
    train, test = split_indices(catalog, strategy, test_size=0.4, n_splits=4, fold=1, seed=3)
    groups = catalog.df["label_group"].to_numpy()
    assert len(train) and len(test)
    assert set(groups[train]).isdisjoint(groups[test])
    assert sorted(np.concatenate([train, test]).tolist()) == list(range(len(catalog)))


def test_config_validation(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("fusion:\n  image_weight: -1\n")
    with pytest.raises(ConfigError):
        load_config(bad)
    bad.write_text("unknown_section: 1\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(bad)
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_default_config_values():
    config = load_config()
    assert config.model.projection_dim == 256
    assert (config.fusion.image_weight, config.fusion.text_weight) == (0.6, 0.4)
    assert config.matching.threshold == 0.75
    assert config.retrieval.index_type == "inner_product"
