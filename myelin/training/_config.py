"""TrainConfig dataclass — all hyperparameters externalized to env per
Phase 4 refinement.

Setting via env lets a single training image serve a hyperparameter
sweep without rebuild. Constructor kwargs always win over env (the
``from_env()`` classmethod is what reads env; direct ``TrainConfig(...)``
uses the literal values you pass).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _read_float(env_name: str, default: float) -> float:
    raw = os.environ.get(env_name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "TrainConfig env %s=%r unparseable; default %.4f",
            env_name, raw, default,
        )
        return default


def _read_int(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "TrainConfig env %s=%r unparseable; default %d",
            env_name, raw, default,
        )
        return default


def _read_bool(env_name: str, default: bool) -> bool:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class TrainConfig:
    """All Memory-R1 contrastive training hyperparameters in one place.

    Defaults from SPARC §Phase 2 Algorithm C + §Phase 4 Refinement.
    """

    # Loss weights (sum to 1.0).
    loss_weight_utility: float = 0.5
    loss_weight_class: float = 0.3
    loss_weight_nce: float = 0.2

    # InfoNCE.
    infonce_temperature: float = 0.07

    # Optimizer.
    learning_rate_head: float = 5e-5
    learning_rate_full: float = 2e-5
    weight_decay: float = 0.01

    # Schedule.
    num_epochs: int = 3
    unfreeze_after_epoch: int = 1
    batch_size: int = 16
    max_steps: int = 0  # 0 → no cap (smoke uses MYELIN_MR1_MAX_STEPS=5)

    # Tokenizer.
    tokenizer_max_length: int = 512

    # Model selection.
    base_model_id: str = "microsoft/deberta-v3-small"
    use_pretrained: bool = True

    # Data splits.
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # Early-stopping.
    early_stop_factor: float = 1.1  # break when val_loss > best*1.1
    early_stop_min_step: int = 1000

    # Acceptance gate.
    target_test_class_accuracy: float = 0.75

    seed: int = 42

    @classmethod
    def from_env(cls) -> "TrainConfig":
        return cls(
            loss_weight_utility=_read_float("MYELIN_MR1_WEIGHT_UTILITY", 0.5),
            loss_weight_class=_read_float("MYELIN_MR1_WEIGHT_CLASS", 0.3),
            loss_weight_nce=_read_float("MYELIN_MR1_WEIGHT_NCE", 0.2),
            infonce_temperature=_read_float("MYELIN_MR1_INFONCE_TEMP", 0.07),
            learning_rate_head=_read_float("MYELIN_MR1_LR_HEAD", 5e-5),
            learning_rate_full=_read_float("MYELIN_MR1_LR_FULL", 2e-5),
            weight_decay=_read_float("MYELIN_MR1_WD", 0.01),
            num_epochs=_read_int("MYELIN_MR1_NUM_EPOCHS", 3),
            unfreeze_after_epoch=_read_int("MYELIN_MR1_UNFREEZE_AFTER", 1),
            batch_size=_read_int("MYELIN_MR1_BATCH_SIZE", 16),
            max_steps=_read_int("MYELIN_MR1_MAX_STEPS", 0),
            tokenizer_max_length=_read_int("MYELIN_MR1_MAX_LEN", 512),
            base_model_id=os.environ.get(
                "MYELIN_MR1_BASE_MODEL", "microsoft/deberta-v3-small"
            ),
            use_pretrained=_read_bool("MYELIN_MR1_USE_PRETRAINED", True),
            train_ratio=_read_float("MYELIN_MR1_TRAIN_RATIO", 0.6),
            val_ratio=_read_float("MYELIN_MR1_VAL_RATIO", 0.2),
            test_ratio=_read_float("MYELIN_MR1_TEST_RATIO", 0.2),
            early_stop_factor=_read_float("MYELIN_MR1_EARLY_STOP_FACTOR", 1.1),
            early_stop_min_step=_read_int("MYELIN_MR1_EARLY_STOP_MIN", 1000),
            target_test_class_accuracy=_read_float(
                "MYELIN_MR1_TARGET_ACC", 0.75
            ),
            seed=_read_int("MYELIN_MR1_SEED", 42),
        )


__all__ = ["TrainConfig"]
