from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration

try:
    from qwen_frozen_policy_adapter import (
        ACTION_DIM,
        DEFAULT_PROPRIO_DIM,
        PARAMETER_LIMIT,
        PROMPT_VERSION,
        ActionHead,
        auxiliary_vector,
        build_prompt,
        last_token_pool,
        load_qwen_processor,
        prepare_image,
        qwen_last_hidden_state,
        resolve_device,
        resolve_dtype,
    )
except ImportError:
    from scripts.qwen_frozen_policy_adapter import (
        ACTION_DIM,
        DEFAULT_PROPRIO_DIM,
        PARAMETER_LIMIT,
        PROMPT_VERSION,
        ActionHead,
        auxiliary_vector,
        build_prompt,
        last_token_pool,
        load_qwen_processor,
        prepare_image,
        qwen_last_hidden_state,
        resolve_device,
        resolve_dtype,
    )


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
DEFAULT_DATASET = "random-sequence/flock-robotics-vla-training-v2"
DEFAULT_DATASET_REVISION = "a90a4a6062e8a43a229408608a68344f04d24e9f"
CACHE_SCHEMA = "frozen_qwen_embedding_cache_v1"
POLICY_SCHEMA = "frozen_qwen_robotics_policy_v1"
BACKBONE_FILE_PATTERNS = (
    "*.json",
    "*.jinja",
    "*.model",
    "*.safetensors",
    "*.tiktoken",
    "*.txt",
    "LICENSE*",
    "NOTICE*",
    "README.md",
)


@dataclass(frozen=True)
class CacheData:
    embeddings: torch.Tensor
    auxiliary: torch.Tensor
    actions: torch.Tensor
    episode_hashes: torch.Tensor
    manifest: dict[str, Any]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_hidden_size(model: nn.Module) -> int:
    config = model.config
    text_config = getattr(config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Could not determine the Qwen text hidden size.")
    return int(hidden_size)


def stable_episode_hash(episode_id: str) -> int:
    digest = hashlib.blake2b(
        episode_id.encode("utf-8"), digest_size=8, person=b"flock-vla"
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def cache_spec(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA,
        "model": args.model,
        "model_revision": args.model_revision,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "dataset_split": args.dataset_split,
        "image_size": args.image_size,
        "proprio_dim": args.proprio_dim,
        "step_stride": args.step_stride,
        "max_samples": args.max_samples,
        "prompt_version": PROMPT_VERSION,
        "pooling": "last_non_padding_token",
    }


def validate_cache_manifest(manifest: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: cached={old!r}, requested={new!r}"
            for key, (old, new) in mismatches.items()
        )
        raise ValueError(
            "Embedding cache does not match this run. "
            f"{details}. Use --force-recompute with "
            "--precompute-embeddings to replace it."
        )


def clear_embedding_cache(cache_dir: Path) -> None:
    if not cache_dir.exists():
        return
    allowed_names = {"manifest.json", ".manifest.json.tmp"}
    unexpected = []
    for path in cache_dir.iterdir():
        is_shard = (
            path.is_file()
            and path.name.startswith((".", "shard-"))
            and "safetensors" in path.name
        )
        if path.is_file() and (path.name in allowed_names or is_shard):
            continue
        unexpected.append(path.name)
    if unexpected:
        raise ValueError(
            f"Refusing to clear {cache_dir}; it contains unrelated entries: "
            f"{', '.join(sorted(unexpected))}."
        )
    for path in cache_dir.iterdir():
        path.unlink()
    cache_dir.rmdir()


def iter_dataset_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The frozen Qwen trainer needs the 'datasets' package. "
            "Install requirements.txt first."
        ) from exc

    dataset = load_dataset(
        args.dataset,
        split=args.dataset_split,
        revision=args.dataset_revision,
        streaming=args.streaming,
    )
    selected = 0
    for row_index, row in enumerate(dataset):
        step = int(row.get("step", row_index))
        if step % args.step_stride:
            continue
        yield row
        selected += 1
        if args.max_samples > 0 and selected >= args.max_samples:
            return


def batched(
    rows: Iterable[dict[str, Any]], batch_size: int
) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_backbone(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, Any, str | None]:
    print(
        f"Loading frozen backbone {args.model}@{args.model_revision} "
        f"on {device} as {dtype}.",
        flush=True,
    )
    processor = load_qwen_processor(
        args.model,
        revision=args.model_revision,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    resolved_revision = getattr(model.config, "_commit_hash", None)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > args.parameter_limit:
        raise ValueError(
            f"Backbone has {parameter_count:,} parameters; configured limit is "
            f"{args.parameter_limit:,}."
        )
    print(f"Frozen backbone parameters: {parameter_count:,}", flush=True)
    return model, processor, resolved_revision


def exported_backbone_is_complete(qwen_dir: Path) -> bool:
    return (
        (qwen_dir / "config.json").is_file()
        and any(qwen_dir.glob("*.safetensors"))
        and (
            (qwen_dir / "tokenizer.json").is_file()
            or (qwen_dir / "tokenizer_config.json").is_file()
        )
        and (qwen_dir / "preprocessor_config.json").is_file()
    )


def validate_exported_backbone(
    qwen_dir: Path,
    args: argparse.Namespace,
    cache_manifest: dict[str, Any] | None = None,
) -> None:
    existing_config = json.loads((qwen_dir / "config.json").read_text())
    if existing_config.get("model_type") != "qwen2_5_vl":
        raise ValueError(f"{qwen_dir} is not a Qwen2.5-VL model directory.")
    marker_path = qwen_dir / "flock_export.json"
    if not marker_path.is_file():
        raise ValueError(
            f"{qwen_dir} has no flock_export.json provenance marker. "
            "Choose a fresh --out directory."
        )
    marker = json.loads(marker_path.read_text())
    if marker.get("source_model") != args.model:
        raise ValueError(
            f"{qwen_dir} was exported from {marker.get('source_model')!r}, "
            f"not {args.model!r}."
        )
    if marker.get("requested_revision") != args.model_revision:
        raise ValueError(
            f"{qwen_dir} was exported from revision "
            f"{marker.get('requested_revision')!r}, not "
            f"{args.model_revision!r}."
        )
    if cache_manifest is not None:
        cache_revision = cache_manifest.get("resolved_model_revision")
        export_revision = marker.get("resolved_revision")
        if cache_revision and export_revision and cache_revision != export_revision:
            raise ValueError(
                "The embedding cache and exported Qwen backbone use different "
                "resolved revisions."
            )
        text_config = existing_config.get("text_config", existing_config)
        exported_hidden_size = int(text_config["hidden_size"])
        cached_hidden_size = int(cache_manifest["hidden_size"])
        if exported_hidden_size != cached_hidden_size:
            raise ValueError(
                f"Exported hidden size {exported_hidden_size} does not match "
                f"cached hidden size {cached_hidden_size}."
            )


def _is_backbone_file(path: Path) -> bool:
    return any(
        fnmatch.fnmatch(path.name, pattern) for pattern in BACKBONE_FILE_PATTERNS
    )


def export_backbone(
    out_dir: Path,
    args: argparse.Namespace,
    resolved_revision: str | None,
) -> None:
    qwen_dir = out_dir / "qwen"
    if exported_backbone_is_complete(qwen_dir):
        validate_exported_backbone(qwen_dir, args)
        print(f"Reusing existing local backbone at {qwen_dir}.", flush=True)
        return
    if qwen_dir.exists() and any(qwen_dir.iterdir()):
        raise ValueError(
            f"Incomplete backbone export exists at {qwen_dir}. Remove that "
            "directory or choose a different --out path."
        )

    source_candidate = Path(args.model).expanduser()
    if source_candidate.is_dir():
        source_dir = source_candidate.resolve()
    else:
        source_dir = Path(
            snapshot_download(
                repo_id=args.model,
                revision=resolved_revision or args.model_revision,
                allow_patterns=list(BACKBONE_FILE_PATTERNS),
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qwen-export-", dir=out_dir) as temp:
        staged = Path(temp) / "qwen"
        staged.mkdir()
        copied = 0
        for source_path in source_dir.iterdir():
            if not source_path.is_file() or not _is_backbone_file(source_path):
                continue
            destination = staged / source_path.name
            shutil.copy2(source_path, destination)
            copied += 1
        if copied == 0 or not exported_backbone_is_complete(staged):
            raise ValueError(
                f"{source_dir} does not contain a complete Qwen2.5-VL snapshot."
            )
        marker = {
            "source_model": args.model,
            "requested_revision": args.model_revision,
            "resolved_revision": resolved_revision,
        }
        (staged / "flock_export.json").write_text(json.dumps(marker, indent=2) + "\n")
        if qwen_dir.exists():
            qwen_dir.rmdir()
        staged.replace(qwen_dir)
    print(f"Exported frozen backbone to {qwen_dir}.", flush=True)


def encode_batch(
    model: nn.Module,
    processor: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    images = [prepare_image(row["image"], image_size=args.image_size) for row in rows]
    prompts = [
        build_prompt(
            processor,
            instruction=str(row.get("instruction", "")),
            task=str(row.get("task", "")),
            difficulty=str(row.get("difficulty", "") or ""),
        )
        for row in rows
    ]
    processor_inputs = processor(
        text=prompts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    processor_inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in processor_inputs.items()
    }
    embeddings = last_token_pool(
        qwen_last_hidden_state(model, processor_inputs),
        processor_inputs["attention_mask"],
    )

    auxiliary = []
    actions = []
    episode_hashes = []
    for row in rows:
        auxiliary.append(
            auxiliary_vector(
                row["proprio"],
                row.get("step", 0),
                row.get("horizon", 1),
                proprio_dim=args.proprio_dim,
            )
        )
        action = np.asarray(row["action"], dtype=np.float32).reshape(-1)
        if action.size != ACTION_DIM:
            raise ValueError(
                f"Expected {ACTION_DIM} action values, got shape {action.shape}."
            )
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        actions.append(np.clip(action, -1.0, 1.0))
        episode_hashes.append(
            stable_episode_hash(str(row.get("episode_id", "unknown")))
        )

    return {
        "embeddings": embeddings.detach().float().cpu().to(torch.float16).contiguous(),
        "auxiliary": torch.from_numpy(np.stack(auxiliary)).contiguous(),
        "actions": torch.from_numpy(np.stack(actions)).contiguous(),
        "episode_hashes": torch.tensor(episode_hashes, dtype=torch.int64).contiguous(),
    }


def write_cache_shard(
    cache_dir: Path,
    shard_index: int,
    chunks: list[dict[str, torch.Tensor]],
) -> tuple[str, int]:
    combined = {
        key: torch.cat([chunk[key] for chunk in chunks], dim=0).contiguous()
        for key in chunks[0]
    }
    filename = f"shard-{shard_index:05d}.safetensors"
    temp_path = cache_dir / f".{filename}.tmp"
    final_path = cache_dir / filename
    save_file(combined, str(temp_path))
    temp_path.replace(final_path)
    return filename, int(combined["actions"].shape[0])


@torch.inference_mode()
def precompute_embeddings(
    model: nn.Module,
    processor: Any,
    args: argparse.Namespace,
    cache_dir: Path,
    device: torch.device,
    resolved_revision: str | None,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending: list[dict[str, torch.Tensor]] = []
    pending_count = 0
    shard_index = 0
    sample_count = 0
    shard_entries: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}

    progress = tqdm(desc="Precomputing Qwen embeddings", unit="sample")
    for rows in batched(iter_dataset_rows(args), args.embedding_batch_size):
        encoded = encode_batch(model, processor, rows, args, device)
        pending.append(encoded)
        batch_count = len(rows)
        pending_count += batch_count
        sample_count += batch_count
        progress.update(batch_count)
        for row in rows:
            task = str(row.get("task", "unknown"))
            task_counts[task] = task_counts.get(task, 0) + 1

        if pending_count >= args.cache_shard_samples:
            filename, count = write_cache_shard(cache_dir, shard_index, pending)
            shard_entries.append({"file": filename, "samples": count})
            pending = []
            pending_count = 0
            shard_index += 1

    if pending:
        filename, count = write_cache_shard(cache_dir, shard_index, pending)
        shard_entries.append({"file": filename, "samples": count})
    progress.close()

    if sample_count == 0:
        raise ValueError("The dataset selection produced no training samples.")

    manifest = {
        **cache_spec(args),
        "resolved_model_revision": resolved_revision,
        "hidden_size": model_hidden_size(model),
        "embedding_dtype": "float16",
        "sample_count": sample_count,
        "task_counts": task_counts,
        "shards": shard_entries,
    }
    temp_manifest = cache_dir / ".manifest.json.tmp"
    temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temp_manifest.replace(cache_dir / "manifest.json")
    print(
        f"Wrote {sample_count:,} embeddings to {len(shard_entries)} cache "
        f"shards in {cache_dir}.",
        flush=True,
    )
    return manifest


def prepare_cache_and_backbone(
    args: argparse.Namespace,
    cache_dir: Path,
    out_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    expected = cache_spec(args)

    if args.force_recompute:
        if not args.precompute_embeddings:
            raise ValueError("--force-recompute requires --precompute-embeddings.")
        clear_embedding_cache(cache_dir)

    cached_manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        cached_manifest = json.loads(manifest_path.read_text())
        validate_cache_manifest(cached_manifest, expected)

    need_cache = cached_manifest is None
    need_backbone = not exported_backbone_is_complete(out_dir / "qwen")
    if (
        need_cache
        and cache_dir.exists()
        and any(cache_dir.iterdir())
        and not args.force_recompute
    ):
        raise ValueError(
            f"{cache_dir} contains a partial embedding cache. Re-run with "
            "--precompute-embeddings --force-recompute."
        )
    if not need_backbone:
        validate_exported_backbone(
            out_dir / "qwen",
            args,
            cached_manifest,
        )
    if need_cache and not args.precompute_embeddings:
        raise FileNotFoundError(
            f"No compatible embedding cache at {cache_dir}. Add "
            "--precompute-embeddings to create it."
        )

    if not need_cache and not need_backbone:
        print(f"Reusing embedding cache at {cache_dir}.", flush=True)
        return cached_manifest

    if not need_cache:
        print(f"Reusing embedding cache at {cache_dir}.", flush=True)
        export_backbone(
            out_dir,
            args,
            cached_manifest.get("resolved_model_revision"),
        )
        validate_exported_backbone(
            out_dir / "qwen",
            args,
            cached_manifest,
        )
        return cached_manifest

    model, processor, resolved_revision = load_backbone(args, device, dtype)
    try:
        cached_manifest = precompute_embeddings(
            model,
            processor,
            args,
            cache_dir,
            device,
            resolved_revision,
        )
    finally:
        del model
        del processor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if need_backbone:
        export_backbone(out_dir, args, resolved_revision)
    validate_exported_backbone(
        out_dir / "qwen",
        args,
        cached_manifest,
    )
    if cached_manifest is None:
        raise RuntimeError("Embedding cache preparation did not produce a manifest.")
    return cached_manifest


def load_cache(cache_dir: Path, manifest: dict[str, Any]) -> CacheData:
    chunks: dict[str, list[torch.Tensor]] = {
        "embeddings": [],
        "auxiliary": [],
        "actions": [],
        "episode_hashes": [],
    }
    for shard in tqdm(manifest["shards"], desc="Loading embedding cache"):
        path = cache_dir / shard["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing embedding cache shard: {path}")
        tensors = load_file(str(path), device="cpu")
        for key in chunks:
            chunks[key].append(tensors[key])
    combined = {
        key: torch.cat(parts, dim=0).contiguous() for key, parts in chunks.items()
    }
    actual_count = int(combined["actions"].shape[0])
    if actual_count != int(manifest["sample_count"]):
        raise ValueError(
            f"Cache has {actual_count} samples but manifest declares "
            f"{manifest['sample_count']}."
        )
    return CacheData(manifest=manifest, **combined)


def episode_split_masks(
    episode_hashes: torch.Tensor,
    val_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    unique_episodes = sorted(set(episode_hashes.tolist()))
    if len(unique_episodes) < 2 or val_fraction <= 0:
        train_mask = torch.ones_like(episode_hashes, dtype=torch.bool)
        return (
            train_mask,
            ~train_mask,
            len(unique_episodes),
            0,
        )

    ranked = sorted(
        unique_episodes,
        key=lambda value: hashlib.blake2b(
            f"{seed}:{value}".encode(), digest_size=8
        ).digest(),
    )
    val_count = min(
        len(ranked) - 1,
        max(1, round(len(ranked) * val_fraction)),
    )
    val_episodes = torch.tensor(ranked[:val_count], dtype=torch.int64)
    val_mask = torch.isin(episode_hashes, val_episodes)
    return ~val_mask, val_mask, len(ranked) - val_count, val_count


def evaluate_head(
    head: ActionHead,
    loader: DataLoader,
    device: torch.device,
) -> float:
    head.eval()
    squared_error = 0.0
    element_count = 0
    with torch.inference_mode():
        for embeddings, auxiliary, actions in loader:
            embeddings = embeddings.to(device=device, dtype=torch.float32)
            auxiliary = auxiliary.to(device=device, dtype=torch.float32)
            actions = actions.to(device=device, dtype=torch.float32)
            prediction = head(embeddings, auxiliary)
            squared_error += float(
                torch.square(prediction - actions).sum().cpu().item()
            )
            element_count += actions.numel()
    head.train()
    return squared_error / max(element_count, 1)


def train_action_head(
    cache: CacheData,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ActionHead, dict[str, Any], torch.Tensor, torch.Tensor]:
    train_mask, val_mask, train_episodes, val_episodes = episode_split_masks(
        cache.episode_hashes,
        args.val_fraction,
        args.seed,
    )
    if not bool(train_mask.any()):
        raise ValueError("Episode split produced no training samples.")

    train_auxiliary = cache.auxiliary[train_mask].float()
    auxiliary_mean = train_auxiliary.mean(dim=0)
    auxiliary_std = train_auxiliary.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_auxiliary = (cache.auxiliary.float() - auxiliary_mean) / auxiliary_std

    train_dataset = TensorDataset(
        cache.embeddings[train_mask],
        normalized_auxiliary[train_mask],
        cache.actions[train_mask],
    )
    val_dataset = TensorDataset(
        cache.embeddings[val_mask],
        normalized_auxiliary[val_mask],
        cache.actions[val_mask],
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if len(val_dataset)
        else None
    )

    hidden_size = int(cache.embeddings.shape[1])
    head = ActionHead(
        hidden_size=hidden_size,
        proprio_dim=args.proprio_dim,
        action_dim=ACTION_DIM,
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_metric = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    global_steps = 0

    for epoch in range(1, args.epochs + 1):
        head.train()
        loss_sum = 0.0
        sample_count = 0
        progress = tqdm(train_loader, desc=f"Action head {epoch}/{args.epochs}")
        for embeddings, auxiliary, actions in progress:
            embeddings = embeddings.to(device=device, dtype=torch.float32)
            auxiliary = auxiliary.to(device=device, dtype=torch.float32)
            actions = actions.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            prediction = head(embeddings, auxiliary)
            loss = loss_fn(prediction, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()

            batch_count = actions.shape[0]
            loss_sum += float(loss.detach().cpu().item()) * batch_count
            sample_count += batch_count
            global_steps += 1
            progress.set_postfix(mse=loss_sum / max(sample_count, 1))

        train_mse = loss_sum / max(sample_count, 1)
        val_mse = (
            evaluate_head(head, val_loader, device)
            if val_loader is not None
            else train_mse
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_mse,
            }
        )
        print(
            f"epoch={epoch} train_mse={train_mse:.8f} val_mse={val_mse:.8f}",
            flush=True,
        )
        if math.isfinite(val_mse) and val_mse < best_metric:
            best_metric = val_mse
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Action-head training never produced a finite metric.")
    head.load_state_dict(best_state)
    report = {
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "train_episodes": train_episodes,
        "validation_episodes": val_episodes,
        "best_validation_mse": best_metric,
        "global_steps": global_steps,
        "history": history,
    }
    return head, report, auxiliary_mean, auxiliary_std


def count_safetensor_parameters(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*.safetensors"):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shape = handle.get_slice(key).get_shape()
                total += math.prod(shape)
    return total


def export_policy(
    head: ActionHead,
    training_report: dict[str, Any],
    cache: CacheData,
    auxiliary_mean: torch.Tensor,
    auxiliary_std: torch.Tensor,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    head_state = {
        key: value.detach().float().cpu().contiguous()
        for key, value in head.state_dict().items()
    }
    temp_head = out_dir / ".action_head.safetensors.tmp"
    save_file(head_state, str(temp_head))
    temp_head.replace(out_dir / "action_head.safetensors")

    source_adapter = Path(__file__).with_name("qwen_frozen_policy_adapter.py")
    shutil.copyfile(source_adapter, out_dir / "flock_robotics_adapter.py")

    qwen_parameter_count = count_safetensor_parameters(out_dir / "qwen")
    head_parameter_count = sum(tensor.numel() for tensor in head_state.values())
    parameter_count = qwen_parameter_count + head_parameter_count
    if parameter_count > args.parameter_limit:
        raise ValueError(
            f"Export contains {parameter_count:,} parameters; limit is "
            f"{args.parameter_limit:,}."
        )

    policy_config = {
        "schema_version": POLICY_SCHEMA,
        "model_type": "frozen_qwen2_5_vl_action_head",
        "source_model": args.model,
        "requested_model_revision": args.model_revision,
        "resolved_model_revision": cache.manifest.get("resolved_model_revision"),
        "hidden_size": int(cache.embeddings.shape[1]),
        "proprio_dim": args.proprio_dim,
        "action_dim": ACTION_DIM,
        "image_size": args.image_size,
        "prompt_version": PROMPT_VERSION,
        "pooling": "last_non_padding_token",
        "attn_implementation": args.attn_implementation,
        "auxiliary_features": [
            *[f"proprio_{index}" for index in range(args.proprio_dim)],
            "step_over_horizon",
        ],
        "auxiliary_mean": auxiliary_mean.tolist(),
        "auxiliary_std": auxiliary_std.tolist(),
        "parameter_count": parameter_count,
        "qwen_parameter_count": qwen_parameter_count,
        "action_head_parameter_count": head_parameter_count,
        "parameter_limit": args.parameter_limit,
        "adapter_inputs": [
            "image",
            "instruction",
            "task",
            "difficulty",
            "proprio",
            "step",
            "horizon",
        ],
    }
    (out_dir / "policy_config.json").write_text(
        json.dumps(policy_config, indent=2) + "\n"
    )

    report = {
        "cache": cache.manifest,
        "training": training_report,
        "optimizer": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.grad_clip,
        },
        "policy": {
            "parameter_count": parameter_count,
            "qwen_parameter_count": qwen_parameter_count,
            "action_head_parameter_count": head_parameter_count,
        },
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "requirements.txt").write_text(
        "numpy>=1.26\n"
        "Pillow>=9.0\n"
        "safetensors>=0.4\n"
        "torch>=2.3\n"
        "transformers>=4.49,<6\n"
    )
    (out_dir / "README.md").write_text(
        "# Frozen Qwen2.5-VL Robotics Policy\n\n"
        "This submission contains a local frozen Qwen2.5-VL-3B-Instruct "
        "backbone and a trained 7D action head. The adapter consumes only "
        "the documented image, instruction, task, difficulty, proprio, step, "
        "and horizon observation fields.\n\n"
        f"Total parameter count: {parameter_count:,}.\n"
    )
    return report


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0 or args.image_size % 28:
        raise ValueError("--image-size must be a positive multiple of 28.")
    if args.proprio_dim <= 0:
        raise ValueError("--proprio-dim must be positive.")
    if args.embedding_batch_size <= 0 or args.batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.cache_shard_samples <= 0:
        raise ValueError("--cache-shard-samples must be positive.")
    if args.step_stride <= 0:
        raise ValueError("--step-stride must be positive.")
    if not 0 <= args.val_fraction < 1:
        raise ValueError("--val-fraction must be in [0, 1).")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if not (args.device in {"auto", "cpu", "mps"} or args.device.startswith("cuda")):
        raise ValueError("--device must be auto, cpu, mps, cuda, or cuda:<index>.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze Qwen2.5-VL, precompute multimodal embeddings, train a "
            "small robotics action head, and export a self-contained policy."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--out",
        default="outputs/qwen3b_frozen_policy",
        help="Submission directory. Embedding cache is kept outside it.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Embedding cache directory. Defaults to a sibling of --out so it "
            "cannot be uploaded as model parameters."
        ),
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum selected timesteps; 0 uses the full split.",
    )
    parser.add_argument("--proprio-dim", type=int, default=DEFAULT_PROPRIO_DIM)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cache-shard-samples", type=int, default=2048)
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--precompute-embeddings",
        action="store_true",
        help="Create the embedding cache when it does not already exist.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Replace an existing embedding cache.",
    )
    parser.add_argument(
        "--parameter-limit",
        type=int,
        default=PARAMETER_LIMIT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    validate_args(args)
    set_seed(args.seed)

    out_dir = Path(args.out).expanduser().resolve()
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else out_dir.with_name(f"{out_dir.name}_embedding_cache")
    )
    if cache_dir == out_dir or out_dir in cache_dir.parents:
        raise ValueError(
            "--cache-dir must be outside --out; otherwise validators may count "
            "cached embeddings as submitted model parameters."
        )

    requested_device = args.device
    if requested_device == "auto":
        if torch.cuda.is_available():
            requested_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested_device = "mps"
        else:
            requested_device = "cpu"
    device = resolve_device(requested_device)
    dtype = resolve_dtype(args.dtype, device)
    if device.type == "cpu" and dtype == torch.float16:
        print(
            "float16 on CPU is poorly supported; using float32 instead.",
            file=sys.stderr,
        )
        dtype = torch.float32

    manifest = prepare_cache_and_backbone(
        args,
        cache_dir,
        out_dir,
        device,
        dtype,
    )
    cache = load_cache(cache_dir, manifest)
    head, training_report, auxiliary_mean, auxiliary_std = train_action_head(
        cache,
        args,
        device,
    )
    report = export_policy(
        head,
        training_report,
        cache,
        auxiliary_mean,
        auxiliary_std,
        args,
        out_dir,
    )
    summary = {
        "status": "ok",
        "output_directory": str(out_dir),
        "embedding_cache": str(cache_dir),
        "samples": int(cache.actions.shape[0]),
        "best_validation_mse": training_report["best_validation_mse"],
        "parameter_count": report["policy"]["parameter_count"],
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
