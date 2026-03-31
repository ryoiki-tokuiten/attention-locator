from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from backend.delta_capture import DeltaCaptureMetadata


DEFAULT_DELTA_VISUALIZATION_MODE = "vector-flow"
DEFAULT_DELTA_TOP_SOURCES = 4
DEFAULT_DELTA_GLOBAL_EDGES = 28
DEFAULT_DELTA_FLOW_BLOCK_SIZE = 6
LOCAL_FLOW_DISTANCE = 1
MEDIUM_FLOW_DISTANCE = 4
ANGLE_VISUALIZATION_MULTIPLIER = 36.0
SENTENCE_VECTOR_MIN_LENGTH = 1.2
SENTENCE_VECTOR_MAX_LENGTH = 3.4
DELTA_VECTOR_MIN_LENGTH = 0.55
DELTA_VECTOR_MAX_LENGTH = 1.8
UNIT_CIRCLE_RADIUS = 1.0
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
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict) or
        not isinstance(payload.get("sentences"), list) or
        not isinstance(payload.get("vectorFlow"), dict)
    ):
        return None
    return payload


def write_cached_visualization(metadata: DeltaCaptureMetadata, payload: dict[str, Any]) -> None:
    visualization_cache_path(metadata).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _compute_opportunity_corrected_scores(
    sentence_flow: np.ndarray,
    sentence_boundaries_ids: Sequence[int],
) -> np.ndarray:
    sentence_count = int(sentence_flow.shape[0]) if sentence_flow.ndim == 2 else 0
    if sentence_count <= 1:
        return np.zeros(sentence_count, dtype=np.float32)

    incoming_scores = np.nan_to_num(
        sentence_flow.sum(axis=0) - np.diag(sentence_flow),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)

    if len(sentence_boundaries_ids) < sentence_count + 1:
        return incoming_scores

    total_usable_tokens = max(int(sentence_boundaries_ids[sentence_count]), 1)
    later_token_opportunities = np.asarray([
        max(total_usable_tokens - int(sentence_boundaries_ids[index + 1]), 1)
        for index in range(sentence_count)
    ], dtype=np.float32)

    corrected_scores = incoming_scores / later_token_opportunities
    finite_mask = np.isfinite(corrected_scores)
    if not finite_mask.all():
        corrected_scores = np.where(finite_mask, corrected_scores, 0.0)

    return corrected_scores.astype(np.float32, copy=False)


def _map_tokens_to_sentences(
    token_offsets: np.ndarray,
    sentence_offsets: Sequence[tuple[int | None, int | None]],
) -> np.ndarray:
    token_count = int(token_offsets.shape[0])
    sentence_count = len(sentence_offsets)
    token_to_sentence = np.full(token_count, -1, dtype=np.int32)

    normalized_sentence_offsets = [
        (
            None if start is None else int(start),
            None if end is None else int(end),
        )
        for start, end in sentence_offsets
    ]

    for token_index, (start, end) in enumerate(token_offsets.tolist()):
        start = int(start)
        end = int(end)
        if end <= start:
            continue

        best_sentence_index = -1
        best_overlap = 0
        for sentence_index in range(sentence_count):
            sentence_start, sentence_end = normalized_sentence_offsets[sentence_index]
            if sentence_start is None or sentence_end is None or sentence_end <= sentence_start:
                continue
            overlap = min(end, sentence_end) - max(start, sentence_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence_index = sentence_index

        if best_overlap > 0:
            token_to_sentence[token_index] = best_sentence_index

    return token_to_sentence


def _sentence_label(sentence: str, limit: int = 84) -> str:
    compact = " ".join(sentence.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit - 1].rstrip()}…"


def _project_to_2d(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    centered = vectors.astype(np.float32, copy=False) - vectors.mean(axis=0, keepdims=True)
    if centered.ndim != 2 or centered.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T if vh.size else np.zeros((vectors.shape[1], 0), dtype=np.float32)
    projected = centered @ basis if basis.size else np.zeros((vectors.shape[0], 0), dtype=np.float32)
    if projected.shape[1] < 2:
        projected = np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])))
    return projected[:, :2].astype(np.float32, copy=False)


def _normalize_vector_2d(vector: np.ndarray, radius: float, fallback_direction: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        fallback_norm = float(np.linalg.norm(fallback_direction))
        if fallback_norm <= 1e-6:
            return np.asarray([radius, 0.0], dtype=np.float32)
        return (fallback_direction.astype(np.float32, copy=False) / fallback_norm) * radius
    return (vector.astype(np.float32, copy=False) / norm) * radius


def _bounded_visual_lengths(
    magnitudes: np.ndarray,
    *,
    min_length: float,
    max_length: float,
) -> np.ndarray:
    if magnitudes.size == 0:
        return np.zeros(0, dtype=np.float32)

    clean = np.nan_to_num(magnitudes.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    positive = clean[clean > 1e-6]
    if positive.size == 0:
        return np.full(clean.shape, min_length, dtype=np.float32)

    lower = float(np.percentile(positive, 15))
    upper = float(np.percentile(positive, 90))
    spread = max(upper - lower, max(upper, 1.0) * 1e-3)
    normalized = np.clip((clean - lower) / spread, 0.0, None)
    eased = np.tanh(normalized * 1.35)
    lengths = min_length + (eased * (max_length - min_length))
    return np.clip(lengths, min_length, max_length).astype(np.float32, copy=False)


def _signed_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-6 or right_norm <= 1e-6:
        return 0.0
    cross = float((left[0] * right[1]) - (left[1] * right[0]))
    dot = float(np.dot(left, right))
    return math.degrees(math.atan2(cross, dot))


def _vector_angle_degrees(vector: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def _wrap_degrees(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def _vector_from_angle_degrees(angle: float, magnitude: float) -> np.ndarray:
    radians = math.radians(angle)
    return np.asarray([
        math.cos(radians) * magnitude,
        math.sin(radians) * magnitude,
    ], dtype=np.float32)


def _build_vector_flow_payload(
    *,
    sentence_payloads: list[dict[str, Any]],
    base_sentence_vectors: np.ndarray,
    delta_sentence_vectors: np.ndarray,
) -> dict[str, Any]:
    sentence_count = len(sentence_payloads)
    if sentence_count == 0:
        return {
            "projection": "pca",
            "angleAmplification": ANGLE_VISUALIZATION_MULTIPLIER,
            "baseWeight": SENTENCE_VECTOR_MIN_LENGTH,
            "deltaWeight": DELTA_VECTOR_MIN_LENGTH,
            "unitCircleRadius": UNIT_CIRCLE_RADIUS,
            "bounds": {"minX": -1.5, "maxX": 1.5, "minY": -1.5, "maxY": 1.5},
        }

    projection_source = np.concatenate(
        [
            base_sentence_vectors,
            delta_sentence_vectors,
            base_sentence_vectors + delta_sentence_vectors,
        ],
        axis=0,
    )
    projected = _project_to_2d(projection_source)
    projected_base = projected[:sentence_count]
    projected_delta = projected[sentence_count:sentence_count * 2]
    base_lengths = _bounded_visual_lengths(
        np.linalg.norm(base_sentence_vectors.astype(np.float32, copy=False), axis=1),
        min_length=SENTENCE_VECTOR_MIN_LENGTH,
        max_length=SENTENCE_VECTOR_MAX_LENGTH,
    )
    delta_lengths = _bounded_visual_lengths(
        np.linalg.norm(delta_sentence_vectors.astype(np.float32, copy=False), axis=1),
        min_length=DELTA_VECTOR_MIN_LENGTH,
        max_length=DELTA_VECTOR_MAX_LENGTH,
    )

    updated_end = np.zeros(2, dtype=np.float32)
    previous_composed_direction = np.asarray([1.0, 0.0], dtype=np.float32)
    all_points: list[np.ndarray] = []

    for sentence_index in range(sentence_count):
        anchor = updated_end.copy()
        raw_sentence_vector = _normalize_vector_2d(
            projected_base[sentence_index],
            float(base_lengths[sentence_index]),
            previous_composed_direction,
        )
        raw_delta_vector = _normalize_vector_2d(
            projected_delta[sentence_index],
            float(delta_lengths[sentence_index]),
            raw_sentence_vector,
        )
        raw_sentence_angle = _vector_angle_degrees(raw_sentence_vector)
        raw_delta_relative_angle = _signed_angle_degrees(raw_sentence_vector, raw_delta_vector)
        sentence_angle = _wrap_degrees(raw_sentence_angle * ANGLE_VISUALIZATION_MULTIPLIER)
        delta_relative_angle = _wrap_degrees(raw_delta_relative_angle * ANGLE_VISUALIZATION_MULTIPLIER)
        sentence_vector = _vector_from_angle_degrees(sentence_angle, float(base_lengths[sentence_index]))
        delta_angle = _wrap_degrees(sentence_angle + delta_relative_angle)
        delta_vector = _vector_from_angle_degrees(delta_angle, float(delta_lengths[sentence_index]))
        sentence_end = anchor + sentence_vector
        updated_end = sentence_end + delta_vector
        composed_vector = updated_end - anchor
        if np.linalg.norm(composed_vector) > 1e-6:
            previous_composed_direction = composed_vector

        sentence_payloads[sentence_index]["vectorFlow"] = {
            "anchorX": round(float(anchor[0]), 6),
            "anchorY": round(float(anchor[1]), 6),
            "sentenceEndX": round(float(sentence_end[0]), 6),
            "sentenceEndY": round(float(sentence_end[1]), 6),
            "updatedEndX": round(float(updated_end[0]), 6),
            "updatedEndY": round(float(updated_end[1]), 6),
            "sentenceAngle": round(sentence_angle, 6),
            "deltaAngle": round(delta_angle, 6),
            "deltaRelativeAngle": round(delta_relative_angle, 6),
            "composedAngle": round(_vector_angle_degrees(composed_vector), 6),
            "sentenceMagnitude": round(float(np.linalg.norm(sentence_vector)), 6),
            "deltaMagnitude": round(float(np.linalg.norm(delta_vector)), 6),
            "composedMagnitude": round(float(np.linalg.norm(composed_vector)), 6),
        }
        all_points.extend([anchor, sentence_end, updated_end])

    coordinates = np.vstack(all_points) if all_points else np.zeros((1, 2), dtype=np.float32)
    padding = UNIT_CIRCLE_RADIUS * 1.6
    return {
        "projection": "pca",
        "angleAmplification": ANGLE_VISUALIZATION_MULTIPLIER,
        "baseWeight": SENTENCE_VECTOR_MIN_LENGTH,
        "deltaWeight": DELTA_VECTOR_MIN_LENGTH,
        "unitCircleRadius": UNIT_CIRCLE_RADIUS,
        "bounds": {
            "minX": round(float(coordinates[:, 0].min() - padding), 6),
            "maxX": round(float(coordinates[:, 0].max() + padding), 6),
            "minY": round(float(coordinates[:, 1].min() - padding), 6),
            "maxY": round(float(coordinates[:, 1].max() + padding), 6),
        },
    }


def build_delta_visualization_payload(
    metadata: DeltaCaptureMetadata,
    analysis_text: str,
    sentence_payloads: Sequence[dict[str, Any]],
    sentence_boundaries_ids: Sequence[int],
    torch_module: Any,
    device: str,
    *,
    top_sources: int = DEFAULT_DELTA_TOP_SOURCES,
    global_edges: int = DEFAULT_DELTA_GLOBAL_EDGES,
    flow_block_size: int = DEFAULT_DELTA_FLOW_BLOCK_SIZE,
) -> dict[str, Any]:
    token_offsets = _load_memmap(metadata.tensor_paths["token_offsets"]).astype(np.int32, copy=False)
    residual_inputs = _load_memmap(metadata.tensor_paths["residual_inputs"])
    attention_weights = _load_memmap(metadata.tensor_paths["attention_weights"])
    projected_value_states = _load_memmap(metadata.tensor_paths["projected_value_states"])
    attention_deltas_post_norm = _load_memmap(metadata.tensor_paths["attention_deltas_post_norm"])

    token_count = int(metadata.token_count)
    sentences = [str(item.get("sentence", "")) for item in sentence_payloads]
    sentence_offsets = [
        (
            item.get("sourceStart"),
            item.get("sourceEnd"),
        )
        for item in sentence_payloads
    ]
    sentence_count = len(sentences)
    token_to_sentence = _map_tokens_to_sentences(token_offsets, sentence_offsets)
    sentence_token_counts = np.bincount(
        token_to_sentence[token_to_sentence >= 0],
        minlength=sentence_count,
    ).astype(np.int32, copy=False)
    visible_sentences = np.asarray([
        (
            start is not None and
            end is not None and
            int(end) > int(start) and
            bool(analysis_text[int(start):int(end)].strip())
        )
        for start, end in sentence_offsets
    ], dtype=bool)

    base_token_states = np.asarray(residual_inputs[0], dtype=np.float32)
    cumulative_attention_deltas = np.asarray(attention_deltas_post_norm, dtype=np.float32).sum(axis=0)
    base_sentence_vectors = np.zeros((sentence_count, metadata.hidden_size), dtype=np.float32)
    delta_sentence_vectors = np.zeros((sentence_count, metadata.hidden_size), dtype=np.float32)
    for sentence_index in range(sentence_count):
        token_indices = np.flatnonzero(token_to_sentence == sentence_index)
        if token_indices.size:
            base_sentence_vectors[sentence_index] = base_token_states[token_indices].mean(axis=0)
            delta_sentence_vectors[sentence_index] = cumulative_attention_deltas[token_indices].mean(axis=0)
    base_sentence_vectors = np.nan_to_num(base_sentence_vectors, nan=0.0, posinf=0.0, neginf=0.0)
    delta_sentence_vectors = np.nan_to_num(delta_sentence_vectors, nan=0.0, posinf=0.0, neginf=0.0)

    drift_by_layer = np.linalg.norm(attention_deltas_post_norm.astype(np.float32), axis=-1).transpose(1, 0)
    sentence_drift_by_layer = np.zeros((sentence_count, metadata.layer_count), dtype=np.float32)
    for sentence_index in range(sentence_count):
        token_indices = np.flatnonzero(token_to_sentence == sentence_index)
        if token_indices.size:
            sentence_drift_by_layer[sentence_index] = drift_by_layer[token_indices].mean(axis=0)

    drift_scores = sentence_drift_by_layer.mean(axis=1) if sentence_drift_by_layer.size else np.zeros(sentence_count, dtype=np.float32)
    drift_norm = _normalize(drift_scores)
    drift_global_max = float(sentence_drift_by_layer.max()) if sentence_drift_by_layer.size else 0.0
    drift_layers_norm = (
        np.clip(sentence_drift_by_layer / drift_global_max, 0.0, 1.0).astype(np.float32)
        if drift_global_max > 0
        else np.zeros_like(sentence_drift_by_layer, dtype=np.float32)
    )
    peak_layers = sentence_drift_by_layer.argmax(axis=1) if sentence_drift_by_layer.size else np.zeros(sentence_count, dtype=np.int32)

    sentence_pair_sums = np.zeros((sentence_count, sentence_count), dtype=np.float32)

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
            source_sentence_index = int(token_to_sentence[source_index])
            if source_sentence_index < 0:
                continue
            target_start = source_index + 1
            if target_start >= token_count:
                continue

            scores = score_block[block_offset, target_start:].astype(np.float32, copy=False)
            if not scores.size:
                continue

            target_indices = np.arange(target_start, token_count, dtype=np.int32)
            target_sentence_indices = token_to_sentence[target_indices]
            valid_mask = target_sentence_indices >= 0
            if not np.any(valid_mask):
                continue

            valid_target_sentence_indices = target_sentence_indices[valid_mask]
            valid_scores = scores[valid_mask]
            np.add.at(
                sentence_pair_sums[source_sentence_index],
                valid_target_sentence_indices,
                valid_scores,
            )

    sentence_pair_flow = np.zeros_like(sentence_pair_sums)
    for source_index in range(sentence_count):
        source_token_count = max(int(sentence_token_counts[source_index]), 1)
        for target_index in range(sentence_count):
            target_token_count = max(int(sentence_token_counts[target_index]), 1)
            normalization = math.sqrt(source_token_count * target_token_count)
            sentence_pair_flow[source_index, target_index] = (
                sentence_pair_sums[source_index, target_index] / normalization
            )

    incoming_scores = _compute_opportunity_corrected_scores(sentence_pair_flow, sentence_boundaries_ids)
    incoming_norm = _normalize(incoming_scores)
    valid_pair_scores = sentence_pair_flow[sentence_pair_flow > 0]
    max_pair_score = float(valid_pair_scores.max()) if valid_pair_scores.size else 1.0

    sentences_payload: list[dict[str, Any]] = []
    for sentence_index, sentence in enumerate(sentences):
        source_start, source_end = sentence_offsets[sentence_index]
        ranked_sources = sorted(
            [
                (candidate_source_index, float(sentence_pair_flow[candidate_source_index, sentence_index]))
                for candidate_source_index in range(sentence_count)
                if candidate_source_index != sentence_index and sentence_pair_flow[candidate_source_index, sentence_index] > 0
            ],
            key=lambda item: (-item[1], item[0]),
        )[:max(1, top_sources)]

        local_total = 0.0
        medium_total = 0.0
        global_total = 0.0
        for source_index in range(sentence_count):
            if source_index == sentence_index:
                continue
            score = float(sentence_pair_flow[source_index, sentence_index])
            if score <= 0:
                continue
            distance = sentence_index - source_index
            bucket = _distance_bucket(distance)
            if bucket == "local":
                local_total += score
            elif bucket == "medium":
                medium_total += score
            else:
                global_total += score

        bucket_total = local_total + medium_total + global_total
        top_total = sum(score for _, score in ranked_sources) or 1.0

        source_payloads: list[dict[str, Any]] = []
        for source_index, score in ranked_sources:
            distance = sentence_index - source_index
            source_start_offset, source_end_offset = sentence_offsets[source_index]
            source_payloads.append(
                {
                    "sourceIndex": source_index,
                    "score": round(score, 6),
                    "scoreNorm": round(max(0.0, min(1.0, score / max_pair_score)), 6),
                    "share": round(score / top_total, 6),
                    "distance": distance,
                    "distanceBucket": _distance_bucket(distance),
                    "sentence": sentences[source_index],
                    "displaySentence": _sentence_label(sentences[source_index]),
                    "sourceStart": source_start_offset,
                    "sourceEnd": source_end_offset,
                    "visible": bool(visible_sentences[source_index]),
                }
            )

        sentences_payload.append(
            {
                "index": sentence_index,
                "sentence": sentence,
                "displaySentence": _sentence_label(sentence),
                "sourceStart": source_start,
                "sourceEnd": source_end,
                "visible": bool(visible_sentences[sentence_index]),
                "incomingFlow": round(float(incoming_scores[sentence_index]), 6),
                "incomingFlowNorm": round(float(incoming_norm[sentence_index]), 6),
                "localShare": round(float(local_total / bucket_total) if bucket_total > 0 else 0.0, 6),
                "mediumShare": round(float(medium_total / bucket_total) if bucket_total > 0 else 0.0, 6),
                "globalShare": round(float(global_total / bucket_total) if bucket_total > 0 else 0.0, 6),
                "drift": round(float(drift_scores[sentence_index]), 6),
                "driftNorm": round(float(drift_norm[sentence_index]), 6),
                "peakLayer": int(peak_layers[sentence_index]) if sentence_drift_by_layer.size else 0,
                "peakLayerRatio": round(
                    float(peak_layers[sentence_index] / max(1, metadata.layer_count - 1))
                    if metadata.layer_count > 1 and sentence_drift_by_layer.size
                    else 0.0,
                    6,
                ),
                "topSources": source_payloads,
            }
        )

    vector_flow_payload = _build_vector_flow_payload(
        sentence_payloads=sentences_payload,
        base_sentence_vectors=base_sentence_vectors,
        delta_sentence_vectors=delta_sentence_vectors,
    )

    candidate_edges: list[dict[str, Any]] = []
    for target_index, sentence_payload in enumerate(sentences_payload):
        for source in sentence_payload["topSources"]:
            if not source["visible"] or not sentence_payload["visible"]:
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
        "sentenceCount": sentence_count,
        "layerCount": metadata.layer_count,
        "headCount": metadata.head_count,
        "hiddenSize": metadata.hidden_size,
        "defaultMode": DEFAULT_DELTA_VISUALIZATION_MODE,
        "sentences": sentences_payload,
        "driftByLayer": np.round(drift_layers_norm, 6).tolist(),
        "globalTopEdges": global_top_edges,
        "vectorFlow": vector_flow_payload,
        "warnings": list(metadata.warnings),
    }
