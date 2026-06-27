from .ordinal import corn_loss, coral_loss, ordinal_logits_to_label, label_to_levels
from .ldam import LDAMLoss, drw_weights
from .multitask import MultiTaskLoss

__all__ = [
    "corn_loss",
    "coral_loss",
    "ordinal_logits_to_label",
    "label_to_levels",
    "LDAMLoss",
    "drw_weights",
    "MultiTaskLoss",
]
