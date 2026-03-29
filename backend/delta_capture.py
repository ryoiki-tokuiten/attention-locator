from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DELTA_CACHE_ROOT = Path(tempfile.gettempdir()) / "attention-markdown-delta-cache"


def build_delta_storage_key(model_name: str, text: str) -> str:
    payload = f"{model_name}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def estimate_delta_storage_bytes(
    *,
    layer_count: int,
    token_count: int,
    head_count: int,
    hidden_size: int,
    head_dim: int,
    dtype: str,
) -> int:
    tensor_specs = (
        ((token_count,), np.dtype(np.int32).itemsize),
        ((layer_count, token_count, hidden_size), np.dtype(dtype).itemsize),
        ((layer_count, head_count, token_count, head_dim), np.dtype(dtype).itemsize),
        ((layer_count, head_count, token_count, hidden_size), np.dtype(dtype).itemsize),
        ((layer_count, token_count, hidden_size), np.dtype(dtype).itemsize),
        ((layer_count, token_count, hidden_size), np.dtype(dtype).itemsize),
        ((layer_count, token_count, hidden_size), np.dtype(dtype).itemsize),
    )
    return int(sum(np.prod(shape) * itemsize for shape, itemsize in tensor_specs))


def repeat_key_value_heads(value_states: Any, repeats: int) -> Any:
    if repeats == 1:
        return value_states

    batch_size, key_value_heads, sequence_length, head_dim = value_states.shape
    expanded = value_states[:, :, None, :, :].expand(
        batch_size,
        key_value_heads,
        repeats,
        sequence_length,
        head_dim,
    )
    return expanded.reshape(batch_size, key_value_heads * repeats, sequence_length, head_dim)


@dataclass
class DeltaCaptureMetadata:
    cache_key: str
    analysis_text: str
    storage_key: str
    artifact_dir: str
    token_count: int
    layer_count: int
    head_count: int
    hidden_size: int
    head_dim: int
    runtime_dtype: str
    storage_dtype: str
    estimated_storage_bytes: int
    tensor_shapes: dict[str, list[int]]
    tensor_paths: dict[str, str]
    warnings: list[str]

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            "cacheKey": payload["cache_key"],
            "analysisText": payload["analysis_text"],
            "storageKey": payload["storage_key"],
            "artifactDir": payload["artifact_dir"],
            "tokenCount": payload["token_count"],
            "layerCount": payload["layer_count"],
            "headCount": payload["head_count"],
            "hiddenSize": payload["hidden_size"],
            "headDim": payload["head_dim"],
            "runtimeDtype": payload["runtime_dtype"],
            "storageDtype": payload["storage_dtype"],
            "estimatedStorageBytes": payload["estimated_storage_bytes"],
            "tensorShapes": payload["tensor_shapes"],
            "tensorPaths": payload["tensor_paths"],
            "warnings": payload["warnings"],
        }


class DeltaTensorStore:
    def __init__(
        self,
        *,
        storage_key: str,
        layer_count: int,
        token_count: int,
        head_count: int,
        hidden_size: int,
        head_dim: int,
        storage_dtype: str = "float16",
    ) -> None:
        self.storage_key = storage_key
        self.storage_dtype = np.dtype(storage_dtype)
        self.artifact_dir = DELTA_CACHE_ROOT / storage_key
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.tensor_shapes = {
            "input_ids": [token_count],
            "residual_inputs": [layer_count, token_count, hidden_size],
            "head_contexts": [layer_count, head_count, token_count, head_dim],
            "head_projected_deltas": [layer_count, head_count, token_count, hidden_size],
            "attention_deltas_pre_norm": [layer_count, token_count, hidden_size],
            "attention_deltas_post_norm": [layer_count, token_count, hidden_size],
            "residual_after_attention": [layer_count, token_count, hidden_size],
        }
        self.tensor_paths = {
            name: str(self.artifact_dir / f"{name}.npy")
            for name in self.tensor_shapes
        }
        self.metadata_path = self.artifact_dir / "metadata.json"

        self._arrays = {
            "input_ids": np.lib.format.open_memmap(
                self.tensor_paths["input_ids"],
                mode="w+",
                dtype=np.int32,
                shape=tuple(self.tensor_shapes["input_ids"]),
            ),
            "residual_inputs": np.lib.format.open_memmap(
                self.tensor_paths["residual_inputs"],
                mode="w+",
                dtype=self.storage_dtype,
                shape=tuple(self.tensor_shapes["residual_inputs"]),
            ),
            "head_contexts": np.lib.format.open_memmap(
                self.tensor_paths["head_contexts"],
                mode="w+",
                dtype=self.storage_dtype,
                shape=tuple(self.tensor_shapes["head_contexts"]),
            ),
            "head_projected_deltas": np.lib.format.open_memmap(
                self.tensor_paths["head_projected_deltas"],
                mode="w+",
                dtype=self.storage_dtype,
                shape=tuple(self.tensor_shapes["head_projected_deltas"]),
            ),
            "attention_deltas_pre_norm": np.lib.format.open_memmap(
                self.tensor_paths["attention_deltas_pre_norm"],
                mode="w+",
                dtype=self.storage_dtype,
                shape=tuple(self.tensor_shapes["attention_deltas_pre_norm"]),
            ),
            "attention_deltas_post_norm": np.lib.format.open_memmap(
                self.tensor_paths["attention_deltas_post_norm"],
                mode="w+",
                dtype=self.storage_dtype,
                shape=tuple(self.tensor_shapes["attention_deltas_post_norm"]),
            ),
            "residual_after_attention": np.lib.format.open_memmap(
                self.tensor_paths["residual_after_attention"],
                mode="w+",
                dtype=self.storage_dtype,
                shape=tuple(self.tensor_shapes["residual_after_attention"]),
            ),
        }

    def _to_numpy(self, tensor: Any) -> np.ndarray:
        detached = tensor.detach()
        if str(detached.dtype) == "torch.bfloat16":
            detached = detached.to(dtype=torch.float32)
        return detached.to(device="cpu").numpy().astype(self.storage_dtype, copy=False)

    def write_input_ids(self, input_ids: Any) -> None:
        self._arrays["input_ids"][:] = np.asarray(input_ids, dtype=np.int32)

    def write_layer(
        self,
        layer_index: int,
        *,
        residual_input: Any,
        head_context: Any,
        head_projected_delta: Any,
        attention_delta_pre_norm: Any,
        attention_delta_post_norm: Any,
        residual_after_attention: Any,
    ) -> None:
        self._arrays["residual_inputs"][layer_index] = self._to_numpy(residual_input)
        self._arrays["head_contexts"][layer_index] = self._to_numpy(head_context)
        self._arrays["head_projected_deltas"][layer_index] = self._to_numpy(head_projected_delta)
        self._arrays["attention_deltas_pre_norm"][layer_index] = self._to_numpy(attention_delta_pre_norm)
        self._arrays["attention_deltas_post_norm"][layer_index] = self._to_numpy(attention_delta_post_norm)
        self._arrays["residual_after_attention"][layer_index] = self._to_numpy(residual_after_attention)

    def flush(self) -> None:
        for array in self._arrays.values():
            flush = getattr(array, "flush", None)
            if callable(flush):
                flush()

    def write_metadata(self, metadata: DeltaCaptureMetadata) -> None:
        self.flush()
        self.metadata_path.write_text(json.dumps(metadata.to_payload(), indent=2), encoding="utf-8")

    def cleanup(self) -> None:
        self.flush()
        shutil.rmtree(self.artifact_dir, ignore_errors=True)
