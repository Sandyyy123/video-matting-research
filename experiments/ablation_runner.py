"""
Ablation study runner for video matting experiments.

AblationRunner accepts a base configuration dict and a list of override
dicts (one per ablation condition).  It merges each override into a deep
copy of the base config, runs the training/evaluation loop, collects
metrics, logs to Weights & Biases, and writes a Markdown summary table.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# W&B is imported lazily so the module still works without it installed
try:
    import wandb as _wandb_module
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb_module = None
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a deep copy of `base`."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _format_metric(value: Any, decimals: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _markdown_table(rows: list[dict], column_order: Optional[list[str]] = None) -> str:
    """Render a list-of-dicts as a Markdown table.

    Args:
        rows:          Each dict is one data row.
        column_order:  Optional explicit column ordering.  Unknown keys are
                       appended at the end.
    """
    if not rows:
        return "_No results._\n"

    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    if column_order:
        ordered = [c for c in column_order if c in seen]
        remainder = [k for k in all_keys if k not in ordered]
        columns = ordered + remainder
    else:
        columns = all_keys

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines = []
    for row in rows:
        cells = [_format_metric(row.get(col, "")) for col in columns]
        body_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, sep] + body_lines) + "\n"


# ---------------------------------------------------------------------------
# AblationRunner
# ---------------------------------------------------------------------------

class AblationRunner:
    """Orchestrates ablation conditions and aggregates results.

    Args:
        base_config:      Base training/evaluation config dict.
        conditions:       List of (name, override_dict) tuples.  The override
                          is deep-merged with the base config.
        run_fn:           Callable ``run_fn(config: dict) -> dict`` that
                          executes one training+eval run and returns a
                          metrics dict (e.g. {"val/sad": 0.03, "val/mse": 0.01}).
        output_dir:       Where to write the Markdown summary.
        wandb_project:    W&B project name (set to None to disable W&B).
        wandb_entity:     W&B entity / team name.
        metric_columns:   Preferred column ordering in the summary table.
    """

    def __init__(
        self,
        base_config: dict,
        conditions: list[tuple[str, dict]],
        run_fn: Callable[[dict], dict],
        output_dir: str = "ablation_results",
        wandb_project: Optional[str] = "video-matting-ablation",
        wandb_entity: Optional[str] = None,
        metric_columns: Optional[list[str]] = None,
    ) -> None:
        self.base_config = base_config
        self.conditions = conditions
        self.run_fn = run_fn
        self.output_dir = Path(output_dir)
        self.wandb_project = wandb_project if _WANDB_AVAILABLE else None
        self.wandb_entity = wandb_entity
        self.metric_columns = metric_columns or [
            "condition",
            "val/sad",
            "val/mse",
            "val/grad",
            "val/conn",
            "runtime_s",
        ]

        self.results: list[dict] = []

    # ------------------------------------------------------------------
    # Run a single condition
    # ------------------------------------------------------------------

    def _run_condition(self, name: str, override: dict) -> dict:
        """Merge config, optionally init W&B run, call run_fn."""
        config = _deep_merge(self.base_config, override)
        config["_condition_name"] = name

        logger.info("Starting ablation condition: '%s'", name)
        t0 = time.time()

        wandb_run = None
        if self.wandb_project and _WANDB_AVAILABLE:
            wandb_run = _wandb_module.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=name,
                config=config,
                reinit=True,
            )

        try:
            metrics = self.run_fn(config)
        except Exception as exc:
            logger.error("Condition '%s' failed: %s", name, exc, exc_info=True)
            metrics = {"error": str(exc)}
        finally:
            runtime = time.time() - t0
            if wandb_run is not None:
                wandb_run.log({"runtime_s": runtime, **metrics})
                wandb_run.finish()

        row = {"condition": name, "runtime_s": round(runtime, 2), **metrics}
        return row

    # ------------------------------------------------------------------
    # Run all conditions
    # ------------------------------------------------------------------

    def run_all(self, resume_from: Optional[list[str]] = None) -> list[dict]:
        """Execute all ablation conditions sequentially.

        Args:
            resume_from: If provided, skip any condition whose name appears
                         in this list (useful for resuming interrupted runs).

        Returns:
            List of result dicts (one per condition), also stored in self.results.
        """
        skip_names: set[str] = set(resume_from or [])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for name, override in self.conditions:
            if name in skip_names:
                logger.info("Skipping already-run condition: '%s'", name)
                continue

            row = self._run_condition(name, override)
            self.results.append(row)
            logger.info(
                "Condition '%s' done. Metrics: %s",
                name,
                {k: v for k, v in row.items() if k not in ("condition",)},
            )

        self._write_summary()
        return self.results

    # ------------------------------------------------------------------
    # Summary output
    # ------------------------------------------------------------------

    def _write_summary(self) -> Path:
        """Write a Markdown summary table to output_dir/ablation_summary.md."""
        summary_path = self.output_dir / "ablation_summary.md"

        # Sort rows by val/sad ascending if available
        rows = sorted(
            self.results,
            key=lambda r: float(r.get("val/sad", float("inf"))),
        )

        lines = [
            "# Ablation Study Summary",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            f"Base config hash: `{_dict_hash(self.base_config)}`",
            f"Conditions run: {len(rows)}",
            "",
            "## Results",
            "",
            _markdown_table(rows, self.metric_columns),
            "",
            "## Condition Overrides",
            "",
        ]

        for name, override in self.conditions:
            lines.append(f"### `{name}`")
            lines.append("")
            lines.append("```yaml")
            lines.append(_dict_to_yaml_str(override))
            lines.append("```")
            lines.append("")

        content = "\n".join(lines)
        summary_path.write_text(content, encoding="utf-8")
        logger.info("Ablation summary written to: %s", summary_path)
        return summary_path

    def best_condition(self, metric: str = "val/sad", lower_is_better: bool = True) -> Optional[dict]:
        """Return the result row for the best-performing condition."""
        valid = [r for r in self.results if metric in r and not isinstance(r[metric], str)]
        if not valid:
            return None
        return min(valid, key=lambda r: r[metric]) if lower_is_better else max(valid, key=lambda r: r[metric])


# ---------------------------------------------------------------------------
# Tiny utilities
# ---------------------------------------------------------------------------

def _dict_hash(d: dict) -> str:
    import hashlib, json
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()[:8]


def _dict_to_yaml_str(d: dict, indent: int = 0) -> str:
    """Very small YAML-like serializer (avoids importing yaml for this util)."""
    lines = []
    prefix = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(_dict_to_yaml_str(v, indent + 1))
        else:
            lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Example usage (illustrative, not executed on import)
# ---------------------------------------------------------------------------

def _example_run_fn(config: dict) -> dict:
    """Placeholder training function used in unit tests / examples."""
    import random
    return {
        "val/sad": random.uniform(0.02, 0.08),
        "val/mse": random.uniform(0.001, 0.01),
        "val/grad": random.uniform(0.03, 0.1),
        "val/conn": random.uniform(0.02, 0.09),
    }


def build_example_runner(output_dir: str = "/tmp/ablation_out") -> AblationRunner:
    """Build an AblationRunner with typical loss-weight ablation conditions."""
    base = {
        "losses": {
            "temporal_consistency": {"weight": 0.3},
            "boundary_refinement": {"weight": 0.4},
            "trimap_guided": {"weight": 1.0},
        },
        "training": {"epochs": 5},
    }
    conditions = [
        ("baseline", {}),
        ("no_temporal", {"losses": {"temporal_consistency": {"weight": 0.0}}}),
        ("no_boundary", {"losses": {"boundary_refinement": {"weight": 0.0}}}),
        ("high_temporal", {"losses": {"temporal_consistency": {"weight": 1.0}}}),
        ("high_boundary", {"losses": {"boundary_refinement": {"weight": 2.0}}}),
    ]
    return AblationRunner(
        base_config=base,
        conditions=conditions,
        run_fn=_example_run_fn,
        output_dir=output_dir,
        wandb_project=None,  # disable W&B in example
    )
