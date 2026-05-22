"""Training-time machinery for the Memory-R1 contrastive utility
backend.

This sub-package is loaded only by the offline training entrypoint
(``python -m myelin.training.train_contrastive``) and by the runtime
encoder during weight loading (it imports ``model_factory.build_joint_model``).
Inference-only deployments do NOT need to install ``myelin[training]``;
the lighter ``myelin[contrastive]`` extra is sufficient.
"""
from __future__ import annotations

__all__: list[str] = []
