"""Silent-failure classifier (block D): split, pipelines, metrics, training."""
from .metrics import best_f2_threshold, brier, evaluate, f2_score, format_results, pr_auc
from .pipeline import build_models, build_preprocessor
from .split import Split, grouped_split, split_from_config
from .train import format_report, train_and_evaluate

__all__ = [
    "f2_score", "pr_auc", "brier", "evaluate", "best_f2_threshold", "format_results",
    "build_preprocessor", "build_models",
    "Split", "grouped_split", "split_from_config",
    "train_and_evaluate", "format_report",
]
