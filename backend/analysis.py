from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Any, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_MODEL_NAME = "prajjwal1/bert-tiny"
DEFAULT_MAX_WINDOW_TOKENS = 384
DEFAULT_WINDOW_OVERLAP_SENTENCES = 2
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
WHITESPACE_RE = re.compile(r"\s+")
HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")
FENCED_CODE_RE = re.compile(r"(^|\n)(`{3,}|~{3,})[^\n]*\n[\s\S]*?\n\2[^\n]*(?=\n|$)", re.MULTILINE)
DISPLAY_MATH_RE = re.compile(r"\$\$[\s\S]*?\$\$")
PAREN_MATH_RE = re.compile(r"\\\([\s\S]*?\\\)")
BRACKET_MATH_RE = re.compile(r"\\\[[\s\S]*?\\\]")
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
LINK_RE = re.compile(r"\[([^\]]+)]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"(?<!`)`([^`]+)`(?!`)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
TABLE_SEPARATOR_RE = re.compile(r"^\|?[\s:\-]+(?:\|[\s:\-]+)+\|?$")
HORIZONTAL_RULE_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
SIMPLE_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class SentenceWindow:
    start_index: int
    end_index: int
    sentences: list[str]


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int


def normalize_sentence(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = ZERO_WIDTH_RE.sub("", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized.lower()


def sentence_tokenize(text: str) -> list[str]:
    clean_text = WHITESPACE_RE.sub(" ", text).strip()
    if not clean_text:
        return []

    try:
        import nltk

        try:
            sentences = nltk.sent_tokenize(clean_text)
        except LookupError:
            from nltk.tokenize import PunktSentenceTokenizer

            sentences = PunktSentenceTokenizer().tokenize(clean_text)
    except ImportError:
        sentences = SIMPLE_SENTENCE_SPLIT_RE.split(clean_text)

    return [WHITESPACE_RE.sub(" ", sentence).strip() for sentence in sentences if sentence.strip()]


def markdown_to_analysis_text(markdown_text: str) -> str:
    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    text = HTML_COMMENT_RE.sub("\n", text)
    text = FENCED_CODE_RE.sub("\n", text)
    text = DISPLAY_MATH_RE.sub(" ", text)
    text = BRACKET_MATH_RE.sub(" ", text)
    text = PAREN_MATH_RE.sub(" ", text)
    text = INLINE_MATH_RE.sub(" ", text)
    text = IMAGE_RE.sub(" ", text)
    text = LINK_RE.sub(r"\1", text)
    text = INLINE_CODE_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)

    retained_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or HORIZONTAL_RULE_RE.match(line):
            continue
        if TABLE_SEPARATOR_RE.match(line):
            continue

        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^>\s*", "", line)
        line = re.sub(r"^\s*[-*+]\s+", "", line)
        line = re.sub(r"^\s*\d+\.\s+", "", line)

        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
            line = ". ".join(cells)

        line = WHITESPACE_RE.sub(" ", line).strip()
        if line:
            retained_lines.append(line)

    return "\n".join(retained_lines)


def normalize_scores(raw_scores: Sequence[float]) -> list[float]:
    if not raw_scores:
        return []

    minimum = min(raw_scores)
    maximum = max(raw_scores)

    if minimum == maximum:
        return [0.5 for _ in raw_scores]

    scale = maximum - minimum
    return [round((score - minimum) / scale, 6) for score in raw_scores]


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


def pack_sentence_windows(
    sentences: Sequence[str],
    tokenizer: Any,
    max_window_tokens: int = DEFAULT_MAX_WINDOW_TOKENS,
    overlap_sentences: int = DEFAULT_WINDOW_OVERLAP_SENTENCES,
) -> list[SentenceWindow]:
    if not sentences:
        return []

    token_lengths = [
        max(1, len(tokenizer(sentence, add_special_tokens=False).input_ids))
        for sentence in sentences
    ]

    windows: list[SentenceWindow] = []
    start_index = 0

    while start_index < len(sentences):
        token_total = 2
        end_index = start_index

        while end_index < len(sentences):
            proposed_length = token_lengths[end_index] + (1 if end_index > start_index else 0)

            if end_index > start_index and token_total + proposed_length > max_window_tokens:
                break

            token_total += proposed_length
            end_index += 1

            if token_total >= max_window_tokens:
                break

        if end_index == start_index:
            end_index = start_index + 1

        windows.append(
            SentenceWindow(
                start_index=start_index,
                end_index=end_index,
                sentences=list(sentences[start_index:end_index]),
            )
        )

        if end_index >= len(sentences):
            break

        next_start = max(end_index - overlap_sentences, start_index + 1)
        start_index = next_start

    return windows


def build_sentence_token_spans(
    sentence_token_lengths: Sequence[int],
    special_tokens_mask: Sequence[int],
) -> list[TokenSpan]:
    regular_positions = [index for index, is_special in enumerate(special_tokens_mask) if not is_special]
    required_tokens = sum(sentence_token_lengths)

    if len(regular_positions) < required_tokens:
        raise ValueError("Encoded token sequence was shorter than the expected sentence token lengths.")

    spans: list[TokenSpan] = []
    cursor = 0

    for token_length in sentence_token_lengths:
        if token_length <= 0:
            spans.append(TokenSpan(start=0, end=0))
            continue

        start = regular_positions[cursor]
        end = regular_positions[cursor + token_length - 1] + 1
        spans.append(TokenSpan(start=start, end=end))
        cursor += token_length

    return spans


def aggregate_sentence_attention(attentions: Sequence[Any], sentence_spans: Sequence[TokenSpan], torch_module: Any):
    if not attentions or not sentence_spans:
        return torch_module.zeros((0, 0))

    stacked_attentions = torch_module.stack(tuple(attentions), dim=0)
    integrated_layers = stacked_attentions.max(dim=0).values.squeeze(0)
    num_heads = int(integrated_layers.shape[0])
    num_sentences = len(sentence_spans)
    sentence_attentions = torch_module.zeros(
        (num_heads, num_sentences, num_sentences),
        device=integrated_layers.device,
        dtype=integrated_layers.dtype,
    )

    for head_index in range(num_heads):
        for source_index, source_span in enumerate(sentence_spans):
            if source_span.end <= source_span.start:
                continue

            for target_index, target_span in enumerate(sentence_spans):
                if target_span.end <= target_span.start:
                    continue

                attention_slice = integrated_layers[
                    head_index,
                    source_span.start:source_span.end,
                    target_span.start:target_span.end,
                ]
                if attention_slice.numel():
                    sentence_attentions[head_index, source_index, target_index] = attention_slice.max()

    return sentence_attentions.max(dim=0).values


class BaseSemanticMarkdownAnalyzer:
    model_name = DEFAULT_MODEL_NAME
    device = "cpu"

    def analyze_markdown(self, markdown_text: str) -> list[dict[str, Any]]:
        analysis_text = markdown_to_analysis_text(markdown_text)
        sentences = sentence_tokenize(analysis_text)

        if not sentences:
            return []

        raw_scores = list(self.compute_raw_scores(sentences))
        if len(raw_scores) != len(sentences):
            raise ValueError("Sentence score count did not match the number of extracted sentences.")

        normalized_scores = normalize_scores(raw_scores)
        bands = assign_score_bands(normalized_scores)

        return [
            {
                "index": index,
                "sentence": sentence,
                "normalizedSentence": normalize_sentence(sentence),
                "weight": weight,
                "band": bands[index],
            }
            for index, (sentence, weight) in enumerate(zip(sentences, normalized_scores))
        ]

    def compute_raw_scores(self, sentences: Sequence[str]) -> Sequence[float]:
        raise NotImplementedError


class SlidingWindowISAAnalyzer(BaseSemanticMarkdownAnalyzer):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_window_tokens: int = DEFAULT_MAX_WINDOW_TOKENS,
        overlap_sentences: int = DEFAULT_WINDOW_OVERLAP_SENTENCES,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

        self.model_name = model_name
        self.max_window_tokens = max_window_tokens
        self.overlap_sentences = overlap_sentences
        self.torch = self._import_torch()
        self.device = self._resolve_device()
        if "bert" in model_name.lower():
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name, output_attentions=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model.to(self.device)
        self.model.eval()

    def _import_torch(self):
        import torch

        return torch

    def _resolve_device(self) -> str:
        if self.torch.cuda.is_available():
            return "cuda"
        if hasattr(self.torch.backends, "mps") and self.torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def compute_raw_scores(self, sentences: Sequence[str]) -> Sequence[float]:
        if not sentences:
            return []
        if len(sentences) == 1:
            return [0.0]

        raw_scores = [0.0 for _ in sentences]
        windows = pack_sentence_windows(
            sentences,
            self.tokenizer,
            max_window_tokens=self.max_window_tokens,
            overlap_sentences=self.overlap_sentences,
        )

        for window in windows:
            window_scores = self._score_window(window.sentences)
            for offset, score in enumerate(window_scores):
                global_index = window.start_index + offset
                if global_index >= len(raw_scores):
                    break
                raw_scores[global_index] = max(raw_scores[global_index], float(score))

        return raw_scores

    def _score_window(self, window_sentences: Sequence[str]) -> list[float]:
        if not window_sentences:
            return []
        if len(window_sentences) == 1:
            return [0.0]

        window_text = "\n".join(window_sentences)
        encoded = self.tokenizer(
            window_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_window_tokens,
            return_special_tokens_mask=True,
        )
        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key != "special_tokens_mask"
        }
        sentence_token_lengths = [
            max(1, len(self.tokenizer(sentence, add_special_tokens=False).input_ids))
            for sentence in window_sentences
        ]

        with self.torch.no_grad():
            outputs = self.model(**model_inputs, output_attentions=True)

        sentence_spans = build_sentence_token_spans(
            sentence_token_lengths,
            encoded["special_tokens_mask"][0].tolist(),
        )
        sentence_attention = aggregate_sentence_attention(
            outputs.attentions,
            sentence_spans,
            self.torch,
        ).detach().cpu()

        usable_count = min(len(window_sentences), int(sentence_attention.shape[0]))
        if usable_count <= 1:
            return [0.0 for _ in window_sentences]

        trimmed_attention = sentence_attention[:usable_count, :usable_count]
        incoming_scores = trimmed_attention.sum(dim=0) - self.torch.diag(trimmed_attention)
        scores = [float(score) for score in incoming_scores.tolist()]

        if usable_count < len(window_sentences):
            scores.extend([0.0] * (len(window_sentences) - usable_count))

        return scores
