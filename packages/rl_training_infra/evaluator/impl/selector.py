from __future__ import annotations

from dataclasses import dataclass

from ..base import CheckpointSelectorBase


@dataclass(slots=True)
class CheckpointSelector(CheckpointSelectorBase):
    pass
