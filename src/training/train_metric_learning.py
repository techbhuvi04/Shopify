"""OPTIONAL: fine-tune the projection heads with metric learning.

The baseline works without any training. This module shows how the
pretrained baseline can be adapted to a marketplace: the CLIP / MiniLM
backbones stay frozen (their cached raw embeddings are reused), and only the
two small projection heads are trained so that listings of the same
``label_group`` move together and different products move apart.

    raw image emb -> image head -> ┐
                                   ├-> loss(image) + loss(text) + loss(fused)
    raw text  emb -> text head  -> ┘

Training uses train-split groups only; before/after retrieval metrics are
reported on the held-out test groups.

Usage:
    python -m src.training.train_metric_learning [--config configs/config.yaml]

Then set ``model.projection_weights`` in the config to the saved file and
rebuild embeddings + index to use the fine-tuned heads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import ProductCatalog, split_indices
from src.embeddings.store import EmbeddingStore, embeddings_dir
from src.evaluation.evaluate import ranking_metrics, retrieve
from src.models.multimodal_encoder import ProjectionHead, save_projection_heads
from src.retrieval.faiss_index import FaissIndex
from src.training.losses import build_loss
from src.utils.config import Config, load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def pk_batches(labels: np.ndarray, rows: np.ndarray, p: int, k: int,
               rng: np.random.Generator) -> list[np.ndarray]:
    """One epoch of P×K batches: P groups per batch, K listings per group.

    Only groups with at least two listings are used (a positive pair is needed).
    """
    by_group: dict[int, np.ndarray] = {}
    for g in np.unique(labels[rows]):
        members = rows[labels[rows] == g]
        if len(members) >= 2:
            by_group[int(g)] = members
    groups = rng.permutation(list(by_group))
    batches = []
    for start in range(0, len(groups), p):
        chunk = groups[start:start + p]
        if len(chunk) < 2:
            continue
        batch = [rng.choice(by_group[g], size=min(k, len(by_group[g])), replace=False) for g in chunk]
        batches.append(np.concatenate(batch))
    return batches


def _evaluate(store: EmbeddingStore, image_head: ProjectionHead, text_head: ProjectionHead,
              labels: np.ndarray, query_rows: np.ndarray, config: Config) -> dict[str, dict]:
    """Held-out retrieval metrics for image, text and weighted-score fusion."""
    img = image_head.project(store.image_raw, store.image_mask)
    txt = text_head.project(store.text_raw, store.text_mask)
    wi, wt = config.fusion.normalized_weights
    fused = np.concatenate([np.sqrt(wi) * img, np.sqrt(wt) * txt], 1)
    fused /= np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12)
    out = {}
    for name, emb in {"image": img, "text": txt, "multimodal": fused}.items():
        res = retrieve(FaissIndex.build(emb), emb, query_rows, labels, config.evaluation.candidate_k)
        out[name] = ranking_metrics(res, config.evaluation.k_values)
    return out


def train(config: Config) -> dict:
    """Train projection heads and return before/after metrics.

    Raises:
        ValueError: If the catalog has no ``label_group`` column.
    """
    set_seed(config.seed)
    store = EmbeddingStore.load(embeddings_dir(config))
    catalog = ProductCatalog(store.metadata)
    labels = catalog.labels
    train_rows, test_rows = split_indices(catalog, config.split.strategy, config.split.test_size,
                                          config.split.n_splits, config.split.fold, config.seed)
    usable = train_rows[store.image_mask[train_rows] & store.text_mask[train_rows]]
    tc, m = config.training, config.model
    image_head = ProjectionHead(store.image_raw.shape[1], m.projection_dim, m.projection_init, config.seed)
    text_head = ProjectionHead(store.text_raw.shape[1], m.projection_dim, m.projection_init, config.seed + 1)

    before = _evaluate(store, image_head.eval(), text_head.eval(), labels, test_rows, config)
    loss_fn = build_loss(tc.loss, tc.margin)
    params = list(image_head.parameters()) + list(text_head.parameters())
    optim = torch.optim.AdamW(params, lr=tc.learning_rate, weight_decay=tc.weight_decay)
    img_raw = torch.from_numpy(store.image_raw)
    txt_raw = torch.from_numpy(store.text_raw)
    y = torch.from_numpy(labels)
    wi, wt = config.fusion.normalized_weights
    rng = np.random.default_rng(config.seed)
    history = []

    image_head.train()
    text_head.train()
    for epoch in range(1, tc.epochs + 1):
        losses = []
        for batch in pk_batches(labels, usable, tc.groups_per_batch, tc.samples_per_group, rng):
            idx = torch.from_numpy(batch)
            zi, zt = image_head(img_raw[idx]), text_head(txt_raw[idx])
            fused = torch.nn.functional.normalize(torch.cat([wi ** 0.5 * zi, wt ** 0.5 * zt], 1), dim=1)
            loss = loss_fn(zi, y[idx]) + loss_fn(zt, y[idx]) + loss_fn(fused, y[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(mean_loss)
        if epoch == 1 or epoch % 10 == 0 or epoch == tc.epochs:
            logger.info("epoch %3d/%d  loss %.4f", epoch, tc.epochs, mean_loss)

    image_head.eval()
    text_head.eval()
    after = _evaluate(store, image_head, text_head, labels, test_rows, config)
    out_path = config.resolve(tc.output_weights)
    save_projection_heads(out_path, image_head, text_head,
                          meta={"image_model": config.model.image_model,
                                "text_model": config.model.text_model,
                                "loss": tc.loss, "epochs": tc.epochs})
    report = {"loss": tc.loss, "margin": tc.margin, "epochs": tc.epochs,
              "n_train_products": int(len(usable)), "n_test_queries": int(len(test_rows)),
              "loss_history": history, "before": before, "after": after,
              "weights_path": str(Path(tc.output_weights))}
    results_dir = config.resolve(config.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metric_learning.json").write_text(json.dumps(report, indent=2))
    logger.info("saved fine-tuned heads to %s", out_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune projection heads (optional).")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    report = train(load_config(args.config))
    ks = [k for k in report["before"]["multimodal"] if k.startswith("recall@") or k.startswith("map@")]
    print(f"\nHeld-out test queries: {report['n_test_queries']}")
    print(f"{'modality':<12}{'metric':<11}{'before':>8}{'after':>8}")
    for modality in ("image", "text", "multimodal"):
        for k in ks:
            print(f"{modality:<12}{k:<11}{report['before'][modality][k]:>8.3f}"
                  f"{report['after'][modality][k]:>8.3f}")


if __name__ == "__main__":
    main()
