import argparse
import csv
import json
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import torch
import yaml
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass(frozen=True)
class RunSummary:
    run_dir: str
    tags: list[str]
    model_target: str | None
    model_name: str | None
    ckpt_path: str | None
    params: int | None
    best_val_acc: float | None
    best_test_acc: float | None


def _read_hydra_config(run_dir: Path) -> dict[str, Any] | None:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        return None
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _find_metrics_csv(run_dir: Path) -> Path | None:
    # CSVLogger layout: <output_dir>/<name>/version_0/metrics.csv
    candidates = list(run_dir.glob("csv/**/metrics.csv"))
    if not candidates:
        return None
    # Choose the newest by mtime
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_best_metrics(metrics_csv: Path) -> tuple[float | None, float | None]:
    best_val: float | None = None
    best_test: float | None = None

    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "val/acc" in row and row["val/acc"] not in (None, ""):
                try:
                    v = float(row["val/acc"])
                    best_val = v if best_val is None else max(best_val, v)
                except ValueError:
                    pass
            if "test/acc" in row and row["test/acc"] not in (None, ""):
                try:
                    t = float(row["test/acc"])
                    best_test = t if best_test is None else max(best_test, t)
                except ValueError:
                    pass

    return best_val, best_test


def _find_ckpt(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None

    # Prefer "last.ckpt" as it's always present when save_last=True
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.is_file():
        return last_ckpt

    # Fallback: any ckpt
    candidates = sorted(ckpt_dir.glob("*.ckpt"))
    return candidates[-1] if candidates else None


def _count_params_from_ckpt(ckpt_path: Path, model_target: str | None) -> int | None:
    # Avoid unpickling arbitrary objects: we rely on weights_only=True.
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch versions may not support weights_only
        ckpt = torch.load(ckpt_path, map_location="cpu")

    hparams = ckpt.get("hyper_parameters") or {}

    # Currently we only reliably support CIFARModule (hyperparams are plain dict after our integration)
    if model_target and model_target.endswith("src.models.cifar_module.CIFARModule"):
        from src.models.cifar_module import CIFARModule

        model = CIFARModule(
            model_name=hparams.get("model_name"),
            model_hparams=hparams.get("model_hparams"),
            optimizer_name=hparams.get("optimizer_name"),
            optimizer_hparams=hparams.get("optimizer_hparams"),
        )
        return sum(p.numel() for p in model.parameters())

    # Fallback: estimate from state_dict (includes buffers; better than nothing)
    state_dict = ckpt.get("state_dict")
    if isinstance(state_dict, dict) and state_dict:
        total = 0
        for v in state_dict.values():
            if hasattr(v, "numel"):
                total += int(v.numel())
        return total if total > 0 else None

    return None


def summarize_run(run_dir: Path) -> RunSummary:
    cfg = _read_hydra_config(run_dir) or {}

    tags = cfg.get("tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]

    model_cfg = cfg.get("model") or {}
    model_target = model_cfg.get("_target_") if isinstance(model_cfg, dict) else None
    model_name = model_cfg.get("model_name") if isinstance(model_cfg, dict) else None

    metrics_csv = _find_metrics_csv(run_dir)
    best_val_acc, best_test_acc = (None, None)
    if metrics_csv is not None:
        best_val_acc, best_test_acc = _extract_best_metrics(metrics_csv)

    ckpt = _find_ckpt(run_dir)
    params = None
    if ckpt is not None:
        params = _count_params_from_ckpt(ckpt, model_target)

    summary = RunSummary(
        run_dir=str(run_dir),
        tags=[str(t) for t in tags],
        model_target=str(model_target) if model_target else None,
        model_name=str(model_name) if model_name else None,
        ckpt_path=str(ckpt) if ckpt else None,
        params=params,
        best_val_acc=best_val_acc,
        best_test_acc=best_test_acc,
    )

    # Write per-run summary.json (non-destructive)
    out_path = run_dir / "summary.json"
    out_path.write_text(json.dumps(summary.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def _format_float(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x * 100:.2f}%"


def _format_int(x: int | None) -> str:
    if x is None:
        return "-"
    return f"{x:,}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare & summarize Hydra/Lightning runs under outputs/.")
    parser.add_argument(
        "--glob",
        dest="globs",
        action="append",
        default=["outputs/*/*"],
        help="Run directory glob (repeatable). Default: outputs/*/*",
    )
    parser.add_argument(
        "--write",
        default=None,
        help="Optional path to write aggregated JSON (e.g. reports/summary.json)",
    )
    parser.add_argument(
        "--write-csv",
        default=None,
        help="Optional path to write aggregated CSV (e.g. reports/summary.csv)",
    )
    parser.add_argument(
        "--sort",
        default="val",
        choices=["val", "test", "params", "run"],
        help="Sort key for table.",
    )
    args = parser.parse_args()

    run_dirs: list[Path] = []
    for pat in args.globs:
        for p in glob(pat):
            run_dir = Path(p)
            if (run_dir / ".hydra" / "config.yaml").is_file():
                run_dirs.append(run_dir)

    run_dirs = sorted(set(run_dirs))
    if not run_dirs:
        console.print("No runs found. Try: --glob 'outputs/*/*' or check outputs directory.")
        raise SystemExit(1)

    summaries = [summarize_run(rd) for rd in run_dirs]

    if args.sort == "val":
        summaries.sort(key=lambda s: (s.best_val_acc is None, -(s.best_val_acc or 0.0)))
    elif args.sort == "test":
        summaries.sort(key=lambda s: (s.best_test_acc is None, -(s.best_test_acc or 0.0)))
    elif args.sort == "params":
        summaries.sort(key=lambda s: (s.params is None, s.params or 0))
    else:
        summaries.sort(key=lambda s: s.run_dir)

    table = Table(title="Experiment Comparison")
    table.add_column("Run", overflow="fold")
    table.add_column("Model")
    table.add_column("Tags", overflow="fold")
    table.add_column("Best val/acc", justify="right")
    table.add_column("Best test/acc", justify="right")
    table.add_column("Params", justify="right")

    for s in summaries:
        table.add_row(
            os.path.relpath(s.run_dir),
            s.model_name or "-",
            ",".join(s.tags) if s.tags else "-",
            _format_float(s.best_val_acc),
            _format_float(s.best_test_acc),
            _format_int(s.params),
        )

    console.print(table)

    if args.write:
        out_path = Path(args.write)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps([s.__dict__ for s in summaries], ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"Wrote JSON: {out_path}")

    if args.write_csv:
        out_path = Path(args.write_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run_dir",
                    "model_name",
                    "tags",
                    "best_val_acc",
                    "best_test_acc",
                    "params",
                    "ckpt_path",
                ],
            )
            writer.writeheader()
            for s in summaries:
                writer.writerow(
                    {
                        "run_dir": s.run_dir,
                        "model_name": s.model_name,
                        "tags": ",".join(s.tags),
                        "best_val_acc": s.best_val_acc,
                        "best_test_acc": s.best_test_acc,
                        "params": s.params,
                        "ckpt_path": s.ckpt_path,
                    }
                )
        console.print(f"Wrote CSV: {out_path}")


if __name__ == "__main__":
    main()
