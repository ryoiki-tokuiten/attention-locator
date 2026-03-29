from __future__ import annotations

import gc
import re
import shutil
import unicodedata
from dataclasses import dataclass
from math import ceil, floor, isfinite
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from backend.delta_capture import (
    DeltaCaptureMetadata,
    DeltaTensorStore,
    build_delta_storage_key,
    estimate_delta_storage_bytes,
    repeat_key_value_heads,
)
from backend.delta_visualization import (
    build_delta_visualization_payload,
    load_cached_visualization,
    write_cached_visualization,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT_DIR / "model"
DEFAULT_MAX_INPUT_TOKENS = 10_000
DEFAULT_MAX_DELTA_VISUALIZATION_TOKENS = 384
HIGH_MEMORY_WARNING_TOKENS = 4_096
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
WHITESPACE_RE = re.compile(r"\s+")


def discover_default_model_name() -> str:
    gguf_files = sorted(DEFAULT_MODEL_DIR.glob("*.gguf"))
    if gguf_files:
        preferred_files = sorted(
            gguf_files,
            key=lambda path: (
                0 if "gemma" in path.name.lower() else 1,
                path.name.lower(),
            ),
        )
        return str(preferred_files[0])
    return "Qwen/Qwen2.5-1.5B-Instruct"


DEFAULT_MODEL_NAME = discover_default_model_name()


@dataclass
class SAVISDocumentState:
    cache_key: str
    analysis_text: str
    prepared_text: str
    token_count: int
    sentences: list[str]
    raw_scores: list[float]
    sentence_payloads: list[dict[str, Any]]
    sentence_attention: list[list[float]]
    tokens: list[str]
    token_attention: Any
    sentence_boundaries_ids: list[int]
    sentence_source_offsets: list[tuple[int | None, int | None]]
    warnings: list[str]


def normalize_sentence(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = ZERO_WIDTH_RE.sub("", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized.lower()


def sanitize_numeric_value(
    value: Any,
    *,
    default: float = 0.0,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default

    if not isfinite(numeric):
        numeric = default

    if clamp_min is not None:
        numeric = max(clamp_min, numeric)
    if clamp_max is not None:
        numeric = min(clamp_max, numeric)

    return numeric


def sanitize_numeric_sequence(
    values: Sequence[Any],
    *,
    default: float = 0.0,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> list[float]:
    return [
        sanitize_numeric_value(
            value,
            default=default,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
        for value in values
    ]


def sanitize_numeric_matrix(
    rows: Sequence[Sequence[Any]],
    *,
    default: float = 0.0,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> list[list[float]]:
    return [
        sanitize_numeric_sequence(
            row,
            default=default,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
        for row in rows
    ]


def normalize_scores(raw_scores: Sequence[float]) -> list[float]:
    sanitized_scores = sanitize_numeric_sequence(raw_scores)
    if not sanitized_scores:
        return []

    minimum = min(sanitized_scores)
    maximum = max(sanitized_scores)

    if minimum == maximum:
        return [0.5 for _ in sanitized_scores]

    scale = maximum - minimum
    return [round((score - minimum) / scale, 6) for score in sanitized_scores]


def percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower_index = floor(position)
    upper_index = ceil(position)

    if lower_index == upper_index:
        return ordered[lower_index]

    progress = position - lower_index
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + ((upper_value - lower_value) * progress)


def assign_score_bands(weights: Sequence[float]) -> list[str]:
    if not weights:
        return []

    spread = max(weights) - min(weights)
    if spread < 0.08:
        return ["medium" for _ in weights]

    high_cutoff = percentile(weights, 0.78)
    medium_cutoff = percentile(weights, 0.38)

    # Keep the attention view expressive even when the model distribution is flat.
    high_cutoff = min(max(high_cutoff, 0.48), 0.82)
    medium_cutoff = min(max(medium_cutoff, 0.18), high_cutoff - 0.08)

    bands: list[str] = []
    for weight in weights:
        if weight >= high_cutoff:
            bands.append("high")
        elif weight >= medium_cutoff:
            bands.append("medium")
        else:
            bands.append("low")

    return bands


def compute_opportunity_corrected_sentence_scores(
    sentence_attention: Any,
    sentence_boundaries_ids: Sequence[int],
    torch_module: Any,
) -> list[float]:
    sentence_count = int(getattr(sentence_attention, "shape", [0])[0] or 0)
    if sentence_count <= 1:
        return [0.0 for _ in range(sentence_count)]

    incoming_scores = torch_module.nan_to_num(
        sentence_attention.sum(dim=0) - torch_module.diag(sentence_attention),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if len(sentence_boundaries_ids) < sentence_count + 1:
        return sanitize_numeric_sequence(incoming_scores.tolist())

    total_usable_tokens = max(int(sentence_boundaries_ids[sentence_count]), 1)
    later_token_opportunities = [
        max(total_usable_tokens - int(sentence_boundaries_ids[index + 1]), 1)
        for index in range(sentence_count)
    ]
    opportunity_tensor = torch_module.tensor(
        later_token_opportunities,
        dtype=incoming_scores.dtype,
        device=incoming_scores.device,
    )
    mean_opportunity = opportunity_tensor.float().mean().clamp(min=1.0).to(incoming_scores.dtype)

    return sanitize_numeric_sequence((incoming_scores * (mean_opportunity / opportunity_tensor)).tolist())


class BaseSemanticMarkdownAnalyzer:
    model_name = DEFAULT_MODEL_NAME
    device = "cpu"

    def analyze_markdown(self, markdown_text: str) -> list[dict[str, Any]]:
        from backend.savis.attention import sentence_tokenize_text

        sentences = sentence_tokenize_text(markdown_text)

        if not sentences:
            return []

        raw_scores = list(self.compute_raw_scores(sentences))
        return self._build_sentence_payloads(sentences, raw_scores)

    def analyze_document(self, markdown_text: str) -> dict[str, Any]:
        sentence_weights = self.analyze_markdown(markdown_text)
        return {
            "analysisText": markdown_text.replace("\r\n", "\n").replace("\r", "\n"),
            "tokenCount": 0,
            "sentenceCount": len(sentence_weights),
            "sentenceWeights": sentence_weights,
            "sentenceAttention": [],
            "warnings": [],
        }

    def analyze_sentence_pair(self, markdown_text: str, sentence_x_index: int, sentence_y_index: int) -> dict[str, Any]:
        raise NotImplementedError("This analyzer does not support token-level sentence pair attention.")

    def capture_delta_document(self, markdown_text: str) -> dict[str, Any]:
        raise NotImplementedError("This analyzer does not support delta vector capture.")

    def analyze_delta_visualization(self, markdown_text: str) -> dict[str, Any]:
        raise NotImplementedError("This analyzer does not support delta visualization.")

    def _build_sentence_payloads(
        self,
        sentences: Sequence[str],
        raw_scores: Sequence[float],
        *,
        sentence_source_offsets: Sequence[tuple[int | None, int | None]] | None = None,
    ) -> list[dict[str, Any]]:
        if len(raw_scores) != len(sentences):
            raise ValueError("Sentence score count did not match the number of extracted sentences.")

        normalized_scores = normalize_scores(raw_scores)
        bands = assign_score_bands(normalized_scores)

        if sentence_source_offsets is None:
            sentence_source_offsets = [(None, None) for _ in sentences]

        payloads: list[dict[str, Any]] = []
        for index, (sentence, weight) in enumerate(zip(sentences, normalized_scores)):
            source_start, source_end = sentence_source_offsets[index]
            payloads.append(
                {
                    "index": index,
                    "sentence": sentence,
                    "normalizedSentence": normalize_sentence(sentence),
                    "weight": weight,
                    "band": bands[index],
                    "sourceStart": source_start,
                    "sourceEnd": source_end,
                }
            )

        return payloads

    def compute_raw_scores(self, sentences: Sequence[str]) -> Sequence[float]:
        raise NotImplementedError


class FullContextSAVISAnalyzer(BaseSemanticMarkdownAnalyzer):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_delta_visualization_tokens: int = DEFAULT_MAX_DELTA_VISUALIZATION_TOKENS,
    ) -> None:
        from backend.savis import Attention, ISA

        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_delta_visualization_tokens = max_delta_visualization_tokens
        self.torch = self._import_torch()
        self.attention_backend = Attention(model_name, attn_implementation="eager")
        self.device = self.attention_backend.device
        self.isa_class = ISA
        self._cached_state: SAVISDocumentState | None = None
        self._cached_delta_capture: DeltaCaptureMetadata | None = None

    def _import_torch(self):
        import torch

        return torch

    def _release_inference_memory(self) -> None:
        gc.collect()
        if self.device == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _clear_cached_delta_capture(self) -> None:
        if self._cached_delta_capture is None:
            return

        artifact_dir = Path(self._cached_delta_capture.artifact_dir)
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)
        self._cached_delta_capture = None

    def _has_non_finite_tensor_values(self, tensor: Any) -> bool:
        return bool(hasattr(tensor, "numel") and tensor.numel() and not self.torch.isfinite(tensor).all().item())

    def compute_raw_scores(self, sentences: Sequence[str]) -> Sequence[float]:
        if not sentences:
            return []
        state = self._compute_state("\n".join(sentences))
        return state.raw_scores

    def analyze_markdown(self, markdown_text: str) -> list[dict[str, Any]]:
        if not markdown_text.strip():
            self._cached_state = None
            return []

        state = self._compute_state(markdown_text)
        return state.sentence_payloads

    def analyze_document(self, markdown_text: str) -> dict[str, Any]:
        if not markdown_text.strip():
            self._cached_state = None
            return {
                "analysisText": markdown_text.replace("\r\n", "\n").replace("\r", "\n"),
                "tokenCount": 0,
                "sentenceCount": 0,
                "sentenceWeights": [],
                "sentenceAttention": [],
                "warnings": [],
            }

        state = self._compute_state(markdown_text)
        return {
            "analysisText": state.analysis_text,
            "tokenCount": state.token_count,
            "sentenceCount": len(state.sentence_payloads),
            "sentenceWeights": state.sentence_payloads,
            "sentenceAttention": state.sentence_attention,
            "warnings": state.warnings,
        }

    def analyze_sentence_pair(self, markdown_text: str, sentence_x_index: int, sentence_y_index: int) -> dict[str, Any]:
        if not markdown_text.strip():
            raise ValueError("No analyzable sentences were found in the provided markdown.")

        state = self._compute_state(markdown_text)
        sentence_count = len(state.sentences)
        if not 0 <= sentence_x_index < sentence_count or not 0 <= sentence_y_index < sentence_count:
            raise ValueError("Sentence indices were outside the available sentence range.")

        sentence_boundaries = state.sentence_boundaries_ids
        x_start = sentence_boundaries[sentence_x_index]
        x_end = sentence_boundaries[sentence_x_index + 1]
        y_start = sentence_boundaries[sentence_y_index]
        y_end = sentence_boundaries[sentence_y_index + 1]

        subset_non_finite = self._has_non_finite_tensor_values(state.token_attention)
        attention_subset = self.torch.nan_to_num(
            state.token_attention[y_start:y_end, x_start:x_end],
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(min=0.0, max=1.0)

        warnings = list(state.warnings)
        if subset_non_finite and "Some non-finite token attention values were stabilized before rendering." not in warnings:
            warnings.append("Some non-finite token attention values were stabilized before rendering.")

        return {
            "sentenceXIndex": sentence_x_index,
            "sentenceYIndex": sentence_y_index,
            "sentenceX": state.sentences[sentence_x_index],
            "sentenceY": state.sentences[sentence_y_index],
            "columnTokens": state.tokens[x_start:x_end],
            "rowTokens": state.tokens[y_start:y_end],
            "attention": sanitize_numeric_matrix(attention_subset.tolist(), clamp_min=0.0, clamp_max=1.0),
            "warnings": warnings,
        }

    def capture_delta_document(self, markdown_text: str) -> dict[str, Any]:
        from backend.savis.attention import _model_input_device

        prepared_text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
        if not prepared_text.strip():
            raise ValueError("No analyzable text was provided for delta capture.")

        if (
            self._cached_delta_capture is not None
            and self._cached_delta_capture.cache_key == prepared_text
            and Path(self._cached_delta_capture.artifact_dir).exists()
        ):
            return self._cached_delta_capture.to_payload()

        if self._cached_delta_capture is not None:
            self._clear_cached_delta_capture()
            self._release_inference_memory()

        model = self.attention_backend.model
        tokenizer = self.attention_backend.tokenizer
        input_device = _model_input_device(model, self.device)

        token_ids = tokenizer(prepared_text, add_special_tokens=True).input_ids
        token_count = len(token_ids)
        delta_token_limit = min(self.max_input_tokens, self.max_delta_visualization_tokens)
        if delta_token_limit is not None and token_count > delta_token_limit:
            raise ValueError(
                f"Document is {token_count} tokens, which exceeds the delta visualization limit of {delta_token_limit}."
            )

        encoded = tokenizer(prepared_text, return_tensors="pt")
        try:
            offset_payload = tokenizer(prepared_text, add_special_tokens=True, return_offsets_mapping=True)
            token_offsets = np.asarray(offset_payload["offset_mapping"], dtype=np.int32)
        except Exception as exc:  # pragma: no cover - depends on tokenizer capabilities
            raise NotImplementedError("The active tokenizer does not expose source offset mappings for delta visualization.") from exc
        token_strings = tokenizer.convert_ids_to_tokens(offset_payload["input_ids"])
        model_inputs = {
            key: value.to(input_device)
            for key, value in encoded.items()
        }

        inner_model = getattr(model, "model", model)
        layers = list(getattr(inner_model, "layers", []))
        if not layers:
            raise NotImplementedError("The current model does not expose decoder layers for delta capture.")

        first_attention_module = getattr(layers[0], "self_attn", None)
        if first_attention_module is None:
            raise NotImplementedError("The current model layers do not expose self-attention modules for delta capture.")

        layer_count = len(layers)
        head_dim = int(getattr(first_attention_module, "head_dim", 0) or 0)
        hidden_size = int(getattr(getattr(first_attention_module, "o_proj", None), "out_features", 0) or 0)
        head_count = int(
            getattr(first_attention_module, "num_heads", 0)
            or (
                getattr(getattr(first_attention_module, "o_proj", None), "in_features", 0) // head_dim
                if head_dim
                else 0
            )
        )

        if not all([layer_count, head_count, head_dim, hidden_size]):
            raise NotImplementedError("Could not resolve layer/head dimensions for delta capture.")

        storage_key = build_delta_storage_key(self.model_name, prepared_text)
        store = DeltaTensorStore(
            storage_key=storage_key,
            layer_count=layer_count,
            token_count=token_count,
            head_count=head_count,
            hidden_size=hidden_size,
            head_dim=head_dim,
        )

        warnings: list[str] = []
        estimated_storage_bytes = estimate_delta_storage_bytes(
            layer_count=layer_count,
            token_count=token_count,
            head_count=head_count,
            hidden_size=hidden_size,
            head_dim=head_dim,
            dtype=str(store.storage_dtype),
        )
        if estimated_storage_bytes >= 1024 * 1024 * 1024:
            warnings.append(
                "Delta capture for this document is large enough that it will spill well past one gigabyte on disk."
            )
        if token_count > HIGH_MEMORY_WARNING_TOKENS:
            warnings.append(
                "This document is large enough that per-layer delta capture may be slow or memory intensive."
            )

        residual_inputs_by_layer: dict[int, Any] = {}
        completed_layers: set[int] = set()
        hooks: list[Any] = []

        def build_layer_pre_hook(layer_index: int):
            def layer_pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                hidden_states = kwargs.get("hidden_states")
                if hidden_states is None and args:
                    hidden_states = args[0]
                if hidden_states is None:
                    raise RuntimeError("Layer input hidden states were not available for delta capture.")
                residual_inputs_by_layer[layer_index] = hidden_states.detach()

            return layer_pre_hook

        def build_attention_hook(layer_index: int, layer_module: Any):
            def attention_hook(
                attention_module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                output: tuple[Any, Any],
            ) -> None:
                residual_input = residual_inputs_by_layer.pop(layer_index, None)
                if residual_input is None:
                    raise RuntimeError("Layer residual input was not captured before attention output.")

                attn_output, attn_weights = output
                if attn_weights is None:
                    raise RuntimeError(
                        "Delta capture requires attention weights from the active attention implementation."
                    )

                normalized_hidden_states = kwargs.get("hidden_states")
                if normalized_hidden_states is None and args:
                    normalized_hidden_states = args[0]
                if normalized_hidden_states is None:
                    raise RuntimeError("Attention input hidden states were not available for delta capture.")
                batch_size, sequence_length, _ = normalized_hidden_states.shape
                module_head_count = int(
                    getattr(attention_module, "num_heads", 0)
                    or (attention_module.o_proj.in_features // attention_module.head_dim)
                )
                key_value_groups = int(getattr(attention_module, "num_key_value_groups", 1) or 1)
                key_value_heads = int(
                    getattr(attention_module, "num_key_value_heads", 0)
                    or max(module_head_count // key_value_groups, 1)
                )
                value_states = attention_module.v_proj(normalized_hidden_states)
                value_states = value_states.view(batch_size, sequence_length, key_value_heads, attention_module.head_dim)
                value_states = value_states.transpose(1, 2)
                value_states = repeat_key_value_heads(
                    value_states,
                    key_value_groups,
                )

                head_context = self.torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
                projection_blocks = attention_module.o_proj.weight.reshape(
                    attention_module.o_proj.out_features,
                    module_head_count,
                    attention_module.head_dim,
                ).permute(1, 0, 2).contiguous()
                projected_value_states = self.torch.einsum(
                    "bhtd,hod->bhto",
                    value_states,
                    projection_blocks,
                ).contiguous()
                head_projected_delta = self.torch.einsum(
                    "bthd,hod->btho",
                    head_context,
                    projection_blocks,
                ).contiguous()
                attention_delta_post_norm = layer_module.post_attention_layernorm(attn_output)
                residual_after_attention = residual_input + attention_delta_post_norm

                store.write_layer(
                    layer_index,
                    residual_input=residual_input.squeeze(0),
                    attention_weights=attn_weights.squeeze(0),
                    value_states=value_states.squeeze(0),
                    projected_value_states=projected_value_states.squeeze(0),
                    head_context=head_context.squeeze(0).permute(1, 0, 2),
                    head_projected_delta=head_projected_delta.squeeze(0).permute(1, 0, 2),
                    attention_delta_pre_norm=attn_output.squeeze(0),
                    attention_delta_post_norm=attention_delta_post_norm.squeeze(0),
                    residual_after_attention=residual_after_attention.squeeze(0),
                )
                completed_layers.add(layer_index)

            return attention_hook

        for layer_index, layer in enumerate(layers):
            hooks.append(layer.register_forward_pre_hook(build_layer_pre_hook(layer_index), with_kwargs=True))
            hooks.append(layer.self_attn.register_forward_hook(build_attention_hook(layer_index, layer), with_kwargs=True))

        try:
            store.write_input_ids(model_inputs["input_ids"][0].detach().cpu().numpy())
            store.write_token_offsets(token_offsets)
            forward_kwargs: dict[str, Any] = {}
            if getattr(self.attention_backend, "model_family", None) == "causal":
                forward_kwargs["use_cache"] = False
            with self.torch.no_grad():
                model(**model_inputs, **forward_kwargs)
        except RuntimeError as exc:
            store.cleanup()
            if "out of memory" in str(exc).lower():
                raise ValueError(
                    "The model ran out of memory while capturing per-layer delta vectors. "
                    "Try a shorter document or lower the max input token limit."
                ) from exc
            raise
        finally:
            for hook in hooks:
                hook.remove()
            residual_inputs_by_layer.clear()
            self._release_inference_memory()

        if len(completed_layers) != layer_count:
            store.cleanup()
            raise RuntimeError("Delta capture did not finish for every decoder layer.")

        runtime_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
        metadata = DeltaCaptureMetadata(
            cache_key=prepared_text,
            analysis_text=prepared_text,
            storage_key=storage_key,
            artifact_dir=str(store.artifact_dir),
            token_count=token_count,
            layer_count=layer_count,
            head_count=head_count,
            hidden_size=hidden_size,
            head_dim=head_dim,
            runtime_dtype=runtime_dtype,
            storage_dtype=str(store.storage_dtype),
            estimated_storage_bytes=estimated_storage_bytes,
            tokens=token_strings,
            tensor_shapes=store.tensor_shapes,
            tensor_paths=store.tensor_paths,
            warnings=warnings,
        )
        store.write_metadata(metadata)
        self._cached_delta_capture = metadata
        return metadata.to_payload()

    def analyze_delta_visualization(self, markdown_text: str) -> dict[str, Any]:
        prepared_text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
        if not prepared_text.strip():
            raise ValueError("No analyzable text was provided for delta visualization.")

        self.capture_delta_document(prepared_text)
        metadata = self._cached_delta_capture
        if metadata is None:
            raise RuntimeError("Delta capture metadata was not available after capture.")

        cached_visualization = load_cached_visualization(metadata)
        if cached_visualization is not None:
            return cached_visualization

        visualization_payload = build_delta_visualization_payload(
            metadata,
            prepared_text,
            self.torch,
            self.device,
        )
        write_cached_visualization(metadata, visualization_payload)
        return visualization_payload

    def _compute_state(
        self,
        markdown_text: str,
    ) -> SAVISDocumentState:
        from backend.savis.attention import map_sentences_to_source_offsets, sentence_tokenize_text

        prepared_text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
        if not prepared_text.strip():
            raise ValueError("No analyzable sentences were found in the provided text.")

        if self._cached_state and self._cached_state.cache_key == prepared_text:
            return self._cached_state
        if self._cached_state is not None:
            self._cached_state = None
            self._release_inference_memory()

        warnings: list[str] = []

        try:
            (
                _prepared_text,
                token_count,
                attentions,
                tokenizer,
                input_ids,
                _outputs,
                sequences,
            ) = self.attention_backend.get_attention_for_document(
                prepared_text,
                max_input_tokens=self.max_input_tokens,
                segment_sentences=False,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise ValueError(
                    "The model ran out of memory while extracting full-document attention. "
                    "Try a shorter document or lower the max input token limit."
                ) from exc
            raise

        isa = self.isa_class(sequences, attentions, tokenizer, source_text=prepared_text)
        token_attention_tensor = isa.head_attentions[0].max(dim=0).values.detach().cpu()
        sentence_attention_tensor = isa.sentence_attention.detach().cpu()
        had_non_finite_token_attention = self._has_non_finite_tensor_values(token_attention_tensor)
        had_non_finite_sentence_attention = self._has_non_finite_tensor_values(sentence_attention_tensor)
        token_attention_tensor = self.torch.nan_to_num(
            token_attention_tensor,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(min=0.0, max=1.0)
        sentence_attention_tensor = self.torch.nan_to_num(
            sentence_attention_tensor,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(min=0.0, max=1.0)

        resolved_sentences = list(isa.sentences)
        sentence_boundaries_ids = list(isa.sentence_boundaries_ids)
        sentence_source_offsets = list(isa.sentence_source_offsets)
        usable_count = min(
            len(resolved_sentences),
            len(sentence_source_offsets),
            int(sentence_attention_tensor.shape[0]),
        )

        if usable_count == 0:
            resolved_sentences = sentence_tokenize_text(prepared_text)
            sentence_source_offsets = map_sentences_to_source_offsets(prepared_text, resolved_sentences)
            usable_count = min(
                len(resolved_sentences),
                len(sentence_source_offsets),
                int(sentence_attention_tensor.shape[0]),
            )

        resolved_sentences = resolved_sentences[:usable_count]
        sentence_source_offsets = sentence_source_offsets[:usable_count]
        trimmed_attention = sentence_attention_tensor[:usable_count, :usable_count]

        if token_count > HIGH_MEMORY_WARNING_TOKENS:
            warnings.append(
                "This document is large enough that full attention extraction may be slow or memory intensive."
            )

        if usable_count < len(isa.sentences):
            warnings.append(
                "Sentence alignment from the SAVIS tokenizer pass was truncated."
            )
        if any(source_start is None or source_end is None for source_start, source_end in sentence_source_offsets):
            warnings.append(
                "Some sentence source ranges could not be mapped back onto the original markdown exactly."
            )
        if had_non_finite_sentence_attention:
            warnings.append(
                "Some non-finite attention values were stabilized before rendering."
            )
        if had_non_finite_token_attention:
            warnings.append(
                "Some non-finite token attention values were stabilized before rendering."
            )

        raw_scores: list[float]
        if usable_count <= 1:
            raw_scores = [0.0 for _ in resolved_sentences]
        else:
            raw_scores = compute_opportunity_corrected_sentence_scores(
                trimmed_attention,
                sentence_boundaries_ids[:usable_count + 1],
                self.torch,
            )

        sentence_payloads = self._build_sentence_payloads(
            resolved_sentences,
            raw_scores,
            sentence_source_offsets=sentence_source_offsets,
        )
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu())

        del isa
        del attentions
        del input_ids
        del sequences
        self._release_inference_memory()

        state = SAVISDocumentState(
            cache_key=prepared_text,
            analysis_text=prepared_text,
            prepared_text=prepared_text,
            token_count=token_count,
            sentences=resolved_sentences,
            raw_scores=raw_scores,
            sentence_payloads=sentence_payloads,
            sentence_attention=sanitize_numeric_matrix(trimmed_attention.tolist(), clamp_min=0.0, clamp_max=1.0),
            tokens=tokens,
            token_attention=token_attention_tensor,
            sentence_boundaries_ids=sentence_boundaries_ids,
            sentence_source_offsets=sentence_source_offsets,
            warnings=warnings,
        )
        self._cached_state = state
        return state
