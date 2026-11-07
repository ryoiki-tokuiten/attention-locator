from __future__ import annotations

import re
from pathlib import Path

import nltk
import torch
from nltk.tokenize import PunktSentenceTokenizer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging


hf_logging.disable_progress_bar()
SMALL_GGUF_STABLE_DTYPE_BYTES = 768 * 1024 * 1024
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")


def _resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_fallback_sentence_tokenizer = PunktSentenceTokenizer()


def _safe_sent_tokenize(text):
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        return _fallback_sentence_tokenizer.tokenize(text)


def sentence_tokenize_text(text: str) -> list[str]:
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [sentence.strip() for sentence in _safe_sent_tokenize(normalized_text) if sentence.strip()]


def _build_normalized_offset_map(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    offset_map: list[int] = []

    for index, char in enumerate(text):
        if ZERO_WIDTH_RE.match(char):
            continue
        if char.isspace():
            if not normalized_chars or normalized_chars[-1] == " ":
                continue
            normalized_chars.append(" ")
            offset_map.append(index)
            continue

        normalized_chars.append(char.lower())
        offset_map.append(index)

    while normalized_chars and normalized_chars[0] == " ":
        normalized_chars.pop(0)
        offset_map.pop(0)

    while normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        offset_map.pop()

    return "".join(normalized_chars), offset_map


def map_sentences_to_source_offsets(text: str, sentences: list[str]) -> list[tuple[int | None, int | None]]:
    normalized_source, offset_map = _build_normalized_offset_map(text)
    search_start = 0
    offsets: list[tuple[int | None, int | None]] = []

    for sentence in sentences:
        normalized_sentence, _ = _build_normalized_offset_map(sentence)
        if not normalized_sentence:
            offsets.append((None, None))
            continue

        match_index = normalized_source.find(normalized_sentence, search_start)
        if match_index < 0:
            match_index = normalized_source.find(normalized_sentence)
        if match_index < 0:
            offsets.append((None, None))
            continue

        source_start = offset_map[match_index]
        source_end = offset_map[match_index + len(normalized_sentence) - 1] + 1
        offsets.append((source_start, source_end))
        search_start = match_index + len(normalized_sentence)

    return offsets


def _resolve_model_source(model_name):
    model_path = Path(str(model_name)).expanduser()

    if model_path.is_file() and model_path.suffix == ".gguf":
        return str(model_path.parent), {"gguf_file": model_path.name}

    if model_path.is_dir():
        gguf_files = sorted(
            model_path.glob("*.gguf"),
            key=lambda path: (
                0 if "gemma" in path.name.lower() else 1,
                path.name.lower(),
            ),
        )
        if gguf_files:
            return str(model_path), {"gguf_file": gguf_files[0].name}

    return str(model_name), {}


def _filter_tokenizer_kwargs(kwargs):
    unsupported_keys = {"attn_implementation", "dtype", "torch_dtype", "device_map", "low_cpu_mem_usage"}
    return {key: value for key, value in kwargs.items() if key not in unsupported_keys}


def _resolve_gguf_path(model_source, resolved_kwargs):
    gguf_file = resolved_kwargs.get("gguf_file")
    if not gguf_file:
        return None
    return Path(model_source) / gguf_file


def _default_model_dtype(device, model_source=None, resolved_kwargs=None):
    resolved_kwargs = resolved_kwargs or {}

    if device == "cuda":
        gguf_path = _resolve_gguf_path(model_source, resolved_kwargs)
        if gguf_path and gguf_path.exists() and gguf_path.stat().st_size <= SMALL_GGUF_STABLE_DTYPE_BYTES:
            # Small quantized GGUF models were producing NaNs in attention extraction under float16.
            return torch.bfloat16
        return torch.float16

    if device == "mps":
        return torch.float16

    return torch.float32


def _default_model_runtime_kwargs(device):
    if device == "cuda":
        return {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
    return {}


def _model_input_device(model, fallback_device):
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for mapped_device in hf_device_map.values():
            if mapped_device in {"cpu", "disk"}:
                continue
            return torch.device(mapped_device)
        return torch.device("cpu")

    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device

    return torch.device(fallback_device)


def prepare_document_text(text, segment_sentences=True):
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")

    if segment_sentences:
        segments = sentence_tokenize_text(normalized_text)
        if segments:
            return "\n".join(segments)

    return normalized_text


class Attention:
    def __init__(self, model_name, **kwargs):
        self.device = _resolve_device()
        self.model_source, resolved_kwargs = _resolve_model_source(model_name)

        tokenizer_kwargs = _filter_tokenizer_kwargs(kwargs)
        tokenizer_kwargs.update(resolved_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_source, **tokenizer_kwargs)

        model_kwargs = dict(kwargs)
        model_kwargs.update(resolved_kwargs)
        model_kwargs.setdefault("attn_implementation", "eager")
        for key, value in _default_model_runtime_kwargs(self.device).items():
            model_kwargs.setdefault(key, value)
        if "dtype" not in model_kwargs and "torch_dtype" not in model_kwargs:
            model_kwargs["dtype"] = _default_model_dtype(self.device, self.model_source, resolved_kwargs)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_source, **model_kwargs)
            self.model_family = "causal"
        except Exception:
            encoder_kwargs = dict(model_kwargs)
            encoder_kwargs.pop("attn_implementation", None)
            self.model = AutoModel.from_pretrained(self.model_source, **encoder_kwargs)
            self.model_family = "encoder"

        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if "device_map" not in model_kwargs:
            self.model.to(self.device)
        self.model.eval()

    def get_attention_for_document(self, text: str, max_input_tokens: int | None = None, segment_sentences: bool = True):
        prepared_text = prepare_document_text(text, segment_sentences=segment_sentences)
        if not prepared_text:
            raise ValueError("No analyzable text was provided.")

        token_ids = self.tokenizer(prepared_text, add_special_tokens=True).input_ids
        token_count = len(token_ids)
        if max_input_tokens is not None and token_count > max_input_tokens:
            raise ValueError(
                f"Document is {token_count} tokens after sentence preparation, which exceeds the limit of {max_input_tokens}."
            )

        encoded = self.tokenizer(prepared_text, return_tensors="pt", return_special_tokens_mask=True)
        input_device = _model_input_device(self.model, self.device)
        model_inputs = {
            key: value.to(input_device)
            for key, value in encoded.items()
            if key != "special_tokens_mask"
        }

        forward_kwargs = {"output_attentions": True}
        if self.model_family == "causal":
            forward_kwargs["use_cache"] = False

        with torch.no_grad():
            outputs = self.model(**model_inputs, **forward_kwargs)

        input_ids = model_inputs["input_ids"].detach().cpu()
        sequences = input_ids[0]

        return token_count, outputs.attentions, self.tokenizer, input_ids, sequences


class ISA:
    def __init__(self, generated_sequences, attentions, tokenizer, source_text: str | None = None):
        self.tokenizer = tokenizer

        sentence_boundaries_ids, sentences = self._find_sentence_boundaries(generated_sequences)

        head_attentions = self._integrate_layers(attentions)
        sentence_attention_heads = self._calculate_sentence_attention(head_attentions, sentence_boundaries_ids)
        sentence_attention = self._aggregate_attention(sentence_attention_heads)

        self.sentence_boundaries_ids = sentence_boundaries_ids
        self.sentences = sentences
        self.sentence_source_offsets = (
            map_sentences_to_source_offsets(source_text, sentences)
            if source_text is not None
            else [(None, None) for _ in sentences]
        )
        self.head_attentions = head_attentions
        self.sentence_attention = sentence_attention

    def _integrate_layers(self, attentions):
        stacked_attentions = torch.stack(attentions, dim=0)
        return torch.max(stacked_attentions, dim=0).values

    def _aggregate_attention(self, sentence_attentions):
        max_attention_heads, _ = torch.max(sentence_attentions, dim=1)
        return max_attention_heads.squeeze(0)

    def _find_sentence_boundaries(self, sequences):
        sentences = []
        sentence_boundaries_text = [0]
        sentence_boundaries_ids = [0]
        tokens = self.tokenizer.convert_ids_to_tokens(sequences)

        for idx_i, token_i in enumerate(tokens):
            if token_i == '\n' or '\n' in token_i or idx_i == len(tokens) - 1:
                text = self.tokenizer.decode(sequences[:idx_i + 1])

                text_chunk = text[sentence_boundaries_text[-1]:]
                nltk_sentences = _safe_sent_tokenize(text_chunk)

                idx_nltk = 0
                idx_j = sentence_boundaries_ids[-1]
                while idx_j < idx_i + 1:
                    if idx_nltk >= len(nltk_sentences):
                        break
                    nltk_sentence = nltk_sentences[idx_nltk]
                    slice_start = sentence_boundaries_ids[-1]
                    idx_j += 1

                    text_slice = self.tokenizer.decode(sequences[slice_start:idx_j + 1])
                    if nltk_sentence in text_slice:
                        sentences.append(text_slice)
                        sentence_boundaries_text.append(len(self.tokenizer.decode(sequences[:idx_j + 1])))
                        sentence_boundaries_ids.append(idx_j + 1)
                        idx_nltk += 1
                        continue

        return sentence_boundaries_ids, sentences

    def _calculate_sentence_attention(self, attentions, sentence_boundaries_ids):
        _, num_heads, _, _ = attentions.shape
        num_sentences = len(sentence_boundaries_ids) - 1

        sentence_attentions = torch.zeros((1, num_heads, num_sentences, num_sentences))

        for head in range(num_heads):
            for i in range(num_sentences):
                start_i = sentence_boundaries_ids[i]
                end_i = sentence_boundaries_ids[i + 1]
                for j in range(num_sentences):
                    start_j = sentence_boundaries_ids[j]
                    end_j = sentence_boundaries_ids[j + 1]
                    attention_slice = attentions[0, head, start_i:end_i, start_j:end_j]

                    if attention_slice.numel() == 0:
                        max_attention = 0
                    else:
                        max_attention = torch.max(attention_slice).item()

                    sentence_attentions[0, head, i, j] = max_attention

        return sentence_attentions
