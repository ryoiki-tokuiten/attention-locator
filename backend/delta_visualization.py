from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.delta_capture import DeltaCaptureMetadata


DEFAULT_DELTA_VISUALIZATION_MODE = "local-global"
DEFAULT_DELTA_TOP_SOURCES = 4
DEFAULT_DELTA_GLOBAL_EDGES = 28
DEFAULT_DELTA_FLOW_BLOCK_SIZE = 6
LOCAL_FLOW_DISTANCE = 8
MEDIUM_FLOW_DISTANCE = 32
DELTA_VISUALIZATION_FILENAME = "delta_visualization.json"


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32, copy=False)

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.zeros_like(values, dtype=np.float32)

    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    if maximum <= minimum:
        return np.full(values.shape, 0.5, dtype=np.float32)

    normalized = (values.astype(np.float32) - minimum) / (maximum - minimum)
    return np.clip(normalized, 0.0, 1.0)


def _load_memmap(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def _distance_bucket(distance: int) -> str:
    if distance <= LOCAL_FLOW_DISTANCE:
        return "local"
    if distance <= MEDIUM_FLOW_DISTANCE:
        return "medium"
    return "global"


def _token_label(token: str) -> str:
    return token.replace("Ġ", " ").replace("▁", " ").replace("\n", "\\n")


def _edge_color(score: float) -> str:
    clamped = max(0.0, min(1.0, float(score)))
    if clamped <= 0.5:
        progress = clamped / 0.5 if clamped else 0.0
        start = np.array([64, 196, 110], dtype=np.float32)
        end = np.array([80, 154, 255], dtype=np.float32)
    else:
        progress = (clamped - 0.5) / 0.5
        start = np.array([80, 154, 255], dtype=np.float32)
        end = np.array([255, 92, 92], dtype=np.float32)
    color = np.round(start + ((end - start) * progress)).astype(np.int32)
    return f"rgb({int(color[0])}, {int(color[1])}, {int(color[2])})"


def visualization_cache_path(metadata: DeltaCaptureMetadata) -> Path:
    return Path(metadata.artifact_dir) / DELTA_VISUALIZATION_FILENAME


def load_cached_visualization(metadata: DeltaCaptureMetadata) -> dict[str, Any] | None:
    cache_path = visualization_cache_path(metadata)
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))


def write_cached_visualization(metadata: DeltaCaptureMetadata, payload: dict[str, Any]) -> None:
    visualization_cache_path(metadata).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_delta_visualization_payload(
    metadata: DeltaCaptureMetadata,
    analysis_text: str,
    torch_module: Any,
    device: str,
    *,
    top_sources: int = DEFAULT_DELTA_TOP_SOURCES,
    global_edges: int = DEFAULT_DELTA_GLOBAL_EDGES,
    flow_block_size: int = DEFAULT_DELTA_FLOW_BLOCK_SIZE,
) -> dict[str, Any]:
    token_offsets = _load_memmap(metadata.tensor_paths["token_offsets"]).astype(np.int32, copy=False)
    attention_weights = _load_memmap(metadata.tensor_paths["attention_weights"])
    projected_value_states = _load_memmap(metadata.tensor_paths["projected_value_states"])
    attention_deltas_post_norm = _load_memmap(metadata.tensor_paths["attention_deltas_post_norm"])

    token_count = int(metadata.token_count)
    visible_mask = np.zeros(token_count, dtype=bool)
    source_starts = np.full(token_count, -1, dtype=np.int32)
    source_ends = np.full(token_count, -1, dtype=np.int32)
    source_substrings: list[str] = []

    for token_index, (start, end) in enumerate(token_offsets.tolist()):
        start = int(start)
        end = int(end)
        source_starts[token_index] = start
        source_ends[token_index] = end
        substring = analysis_text[start:end] if end > start else ""
        source_substrings.append(substring)
        visible_mask[token_index] = end > start and bool(substring.strip())

    drift_by_layer = np.linalg.norm(attention_deltas_post_norm.astype(np.float32), axis=-1).transpose(1, 0)
    drift_scores = drift_by_layer.mean(axis=1) if drift_by_layer.size else np.zeros(token_count, dtype=np.float32)
    drift_norm = _normalize(drift_scores)
    drift_global_max = float(drift_by_layer.max()) if drift_by_layer.size else 0.0
    drift_layers_norm = (
        np.clip(drift_by_layer / drift_global_max, 0.0, 1.0).astype(np.float32)
        if drift_global_max > 0
        else np.zeros_like(drift_by_layer, dtype=np.float32)
    )
    peak_layers = drift_by_layer.argmax(axis=1) if drift_by_layer.size else np.zeros(token_count, dtype=np.int32)

    incoming_scores = np.zeros(token_count, dtype=np.float32)
    local_scores = np.zeros(token_count, dtype=np.float32)
    medium_scores = np.zeros(token_count, dtype=np.float32)
    global_scores = np.zeros(token_count, dtype=np.float32)
    top_source_scores = np.full((token_count, top_sources), -1.0, dtype=np.float32)
    top_source_indices = np.full((token_count, top_sources), -1, dtype=np.int32)

    flat_attention = attention_weights.reshape(metadata.layer_count * metadata.head_count, token_count, token_count)
    flat_projected_values = projected_value_states.reshape(
        metadata.layer_count * metadata.head_count,
        token_count,
        metadata.hidden_size,
    )

    use_cuda = device == "cuda" and bool(getattr(torch_module.cuda, "is_available", lambda: False)())
    flow_device = torch_module.device("cuda" if use_cuda else "cpu")

    for block_start in range(0, token_count, max(1, flow_block_size)):
        block_end = min(token_count, block_start + max(1, flow_block_size))
        source_slice = slice(block_start, block_end)

        attention_block = np.asarray(flat_attention[:, :, source_slice], dtype=np.float32)
        projected_value_block = np.asarray(flat_projected_values[:, source_slice, :], dtype=np.float32)

        attention_tensor = torch_module.from_numpy(attention_block).to(flow_device)
        projected_value_tensor = torch_module.from_numpy(projected_value_block).to(flow_device)

        coefficients = attention_tensor.permute(2, 1, 0).contiguous()
        source_vectors = projected_value_tensor.permute(1, 0, 2).contiguous()
        combined_vectors = torch_module.matmul(coefficients, source_vectors)
        score_block = torch_module.linalg.vector_norm(combined_vectors, dim=-1).to("cpu").numpy()

        del attention_tensor, projected_value_tensor, coefficients, source_vectors, combined_vectors
        if use_cuda:
            torch_module.cuda.empty_cache()

        for block_offset, source_index in enumerate(range(block_start, block_end)):
            target_start = source_index + 1
            if target_start >= token_count:
                continue

            scores = score_block[block_offset, target_start:].astype(np.float32, copy=False)
            if not scores.size:
                continue

            target_indices = np.arange(target_start, token_count, dtype=np.int32)
            distances = target_indices - source_index

            incoming_scores[target_indices] += scores
            local_mask = distances <= LOCAL_FLOW_DISTANCE
            medium_mask = (distances > LOCAL_FLOW_DISTANCE) & (distances <= MEDIUM_FLOW_DISTANCE)
            global_mask = distances > MEDIUM_FLOW_DISTANCE
            local_scores[target_indices[local_mask]] += scores[local_mask]
            medium_scores[target_indices[medium_mask]] += scores[medium_mask]
            global_scores[target_indices[global_mask]] += scores[global_mask]

            for local_target_offset, target_index in enumerate(target_indices.tolist()):
                score = float(scores[local_target_offset])
                if score <= 0:
                    continue
                min_slot = int(top_source_scores[target_index].argmin())
                if score > float(top_source_scores[target_index, min_slot]):
                    top_source_scores[target_index, min_slot] = score
                    top_source_indices[target_index, min_slot] = source_index

    incoming_norm = _normalize(incoming_scores)
    valid_top_scores = top_source_scores[top_source_scores > 0]
    max_top_score = float(valid_top_scores.max()) if valid_top_scores.size else 1.0

    tokens_payload: list[dict[str, Any]] = []
    for token_index, token in enumerate(metadata.tokens):
        bucket_total = float(local_scores[token_index] + medium_scores[token_index] + global_scores[token_index])
        local_share = float(local_scores[token_index] / bucket_total) if bucket_total > 0 else 0.0
        medium_share = float(medium_scores[token_index] / bucket_total) if bucket_total > 0 else 0.0
        global_share = float(global_scores[token_index] / bucket_total) if bucket_total > 0 else 0.0

        ranked_sources = sorted(
            [
                (int(source_index), float(score))
                for source_index, score in zip(top_source_indices[token_index].tolist(), top_source_scores[token_index].tolist())
                if source_index >= 0 and score > 0
            ],
            key=lambda item: (-item[1], item[0]),
        )
        top_total = sum(score for _, score in ranked_sources) or 1.0

        source_payloads: list[dict[str, Any]] = []
        for source_index, score in ranked_sources:
            distance = token_index - source_index
            source_payloads.append(
                {
                    "sourceIndex": source_index,
                    "score": round(score, 6),
                    "scoreNorm": round(max(0.0, min(1.0, score / max_top_score)), 6),
                    "share": round(score / top_total, 6),
                    "distance": distance,
                    "distanceBucket": _distance_bucket(distance),
                    "token": metadata.tokens[source_index],
                    "displayToken": _token_label(metadata.tokens[source_index]),
                    "sourceStart": None if source_starts[source_index] < 0 else int(source_starts[source_index]),
                    "sourceEnd": None if source_ends[source_index] < 0 else int(source_ends[source_index]),
                    "visible": bool(visible_mask[source_index]),
                }
            )

        tokens_payload.append(
            {
                "index": token_index,
                "token": token,
                "displayToken": _token_label(token),
                "sourceText": source_substrings[token_index],
                "sourceStart": None if source_starts[token_index] < 0 else int(source_starts[token_index]),
                "sourceEnd": None if source_ends[token_index] < 0 else int(source_ends[token_index]),
                "visible": bool(visible_mask[token_index]),
                "incomingFlow": round(float(incoming_scores[token_index]), 6),
                "incomingFlowNorm": round(float(incoming_norm[token_index]), 6),
                "localShare": round(local_share, 6),
                "mediumShare": round(medium_share, 6),
                "globalShare": round(global_share, 6),
                "drift": round(float(drift_scores[token_index]), 6),
                "driftNorm": round(float(drift_norm[token_index]), 6),
                "peakLayer": int(peak_layers[token_index]) if drift_by_layer.size else 0,
                "peakLayerRatio": round(
                    float(peak_layers[token_index] / max(1, metadata.layer_count - 1))
                    if metadata.layer_count > 1 and drift_by_layer.size
                    else 0.0,
                    6,
                ),
                "topSources": source_payloads,
            }
        )

    candidate_edges: list[dict[str, Any]] = []
    for target_index, token_payload in enumerate(tokens_payload):
        for source in token_payload["topSources"]:
            if not source["visible"] or not token_payload["visible"]:
                continue
            candidate_edges.append(
                {
                    "id": f"{source['sourceIndex']}:{target_index}",
                    "sourceIndex": source["sourceIndex"],
                    "targetIndex": target_index,
                    "score": source["score"],
                    "scoreNorm": source["scoreNorm"],
                    "distance": source["distance"],
                    "distanceBucket": source["distanceBucket"],
                    "color": _edge_color(source["scoreNorm"]),
                }
            )

    global_top_edges = sorted(
        candidate_edges,
        key=lambda edge: (-edge["score"], edge["sourceIndex"], edge["targetIndex"]),
    )[:max(1, global_edges)]

    return {
        "analysisText": analysis_text,
        "tokenCount": metadata.token_count,
        "layerCount": metadata.layer_count,
        "headCount": metadata.head_count,
        "hiddenSize": metadata.hidden_size,
        "defaultMode": DEFAULT_DELTA_VISUALIZATION_MODE,
        "tokens": tokens_payload,
        "driftByLayer": np.round(drift_layers_norm, 6).tolist(),
        "globalTopEdges": global_top_edges,
        "warnings": list(metadata.warnings),
    }
