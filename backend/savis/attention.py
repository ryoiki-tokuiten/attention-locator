from __future__ import annotations

import re
from pathlib import Path

import nltk
import torch
from nltk.tokenize import PunktSentenceTokenizer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.utils import logging as hf_logging


hf_logging.disable_progress_bar()
SMALL_GGUF_STABLE_DTYPE_BYTES = 768 * 1024 * 1024


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


def _resolve_model_source(model_name):
    model_path = Path(str(model_name)).expanduser()

    if model_path.is_file() and model_path.suffix == ".gguf":
        return str(model_path.parent), {"gguf_file": model_path.name}, model_path.name

    if model_path.is_dir():
        gguf_files = sorted(
            model_path.glob("*.gguf"),
            key=lambda path: (
                0 if "gemma" in path.name.lower() else 1,
                path.name.lower(),
            ),
        )
        if gguf_files:
            return str(model_path), {"gguf_file": gguf_files[0].name}, gguf_files[0].name

    return str(model_name), {}, str(model_name)


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
    if segment_sentences:
        segments = [sentence.strip() for sentence in _safe_sent_tokenize(text) if sentence.strip()]
    else:
        segments = [line.strip() for line in text.splitlines() if line.strip()]

    if not segments and text.strip():
        segments = [text.strip()]

    return "\n".join(segments)


class Attention:
    def __init__(self, model_name, **kwargs):
        self.device = _resolve_device()
        self.model_source, resolved_kwargs, self.model_name = _resolve_model_source(model_name)

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

    def get_attention_for_texts(self, text_x: str, text_y: str = None):
        if self.model_family == "causal":
            input_text = prepare_document_text(text_x, segment_sentences=True)
            input_device = _model_input_device(self.model, self.device)
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(input_device)
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1,
                output_attentions=True,
                return_dict_in_generate=True,
                stopping_criteria=None,
            )
            generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            attentions = outputs.attentions

            return generated_text, attentions, self.tokenizer, input_ids, outputs

        text_combined = "\n".join(part for part in [text_x, text_y] if part)
        input_ids = self.tokenizer(text_combined, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model(input_ids, output_attentions=True)
        attentions_combined = outputs.attentions

        tokens_x = self.tokenizer(text_x, return_tensors="pt").input_ids.to(self.device)
        x_seq_len = tokens_x.size(1)

        return x_seq_len, attentions_combined, self.tokenizer, input_ids, outputs

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

        return prepared_text, token_count, outputs.attentions, self.tokenizer, input_ids, outputs, sequences


class ISA:
    def __init__(self, generated_sequences, attentions, tokenizer):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.attention_type = 'enc' if isinstance(attentions[0], torch.Tensor) else 'dec'

        # 문장별로 분리
        # sentences = [sentence.strip() for sentence in generated_text.split('\n') if sentence.strip()]
        sentence_boundaries_text, sentence_boundaries_ids, sentences = self._find_sentence_boundaries(generated_sequences)

        # 문장 간 attention 계산
        layer_attentions = self._integrate_steps(attentions)
        head_attentions = self._integrate_layers(layer_attentions)
        sentence_attention_heads = self._calculate_sentence_attention(head_attentions, sentence_boundaries_ids)
        sentence_attention = self._aggregate_attention(sentence_attention_heads)

        self.sequences = generated_sequences
        self.sentence_boundaries_text = sentence_boundaries_text
        self.sentence_boundaries_ids = sentence_boundaries_ids
        self.sentences = sentences
        self.attentions = attentions
        self.layer_attentions = layer_attentions
        self.head_attentions = head_attentions
        self.sentence_attention_heads = sentence_attention_heads
        self.sentence_attention = sentence_attention
    
    def _integrate_steps(self, attentions):
        if self.attention_type == 'enc':
            # 문장 관계
            # shape: (num_layers, batch_size, num_heads, max_seq_length, max_seq_length)
            return attentions
        else:
            # 생성 모델
            self.attention_type = 'dec'
            num_layers = len(attentions[0])  # 레이어 수
            batch_size, num_heads = attentions[0][0].shape[:2]  # 배치 크기, 헤드 수
            final_seq_len = attentions[-1][-1].shape[-1]  # 최종 시퀀스 길이

            # 최종 합쳐진 어텐션을 저장할 텐서
            integrated = torch.zeros(
                num_layers, batch_size, num_heads, final_seq_len, final_seq_len
            ).to(attentions[0][0].device)

            for step_idx, step_attention in enumerate(attentions):
                for layer_idx, layer_attention in enumerate(step_attention):
                    current_seq_len = layer_attention.shape[-1]  # 현재 스텝의 시퀀스 길이

                    if step_idx == 0:
                        # 첫 스텝은 입력 토큰의 어텐션으로 채움
                        integrated[layer_idx, :, :, :current_seq_len, :current_seq_len] = layer_attention
                    else:
                        # 새로 생성된 토큰의 어텐션 값만 추가
                        integrated[layer_idx, :, :, current_seq_len-1:current_seq_len, :current_seq_len] = layer_attention
        
            # shape: (num_layers, batch_size, num_heads, max_seq_length, max_seq_length)
            return integrated

    def _integrate_layers(self, attentions):
        if self.attention_type == 'enc':
            # 문장 관계
            # attentions shape: (num_layers, batch_size, num_heads, max_seq_length, max_seq_length)
            _, num_heads, max_seq_length, _ = attentions[0].shape

            # 레이어별 어텐션을 합치기 위해 (num_layers, batch_size, num_heads, max_seq_length, max_seq_length)로 스택
            stacked_attentions = torch.stack(attentions, dim=0)  # shape: (num_layers, batch_size, num_heads, max_seq_length, max_seq_length)

            # 어텐션 결합 방법: Max pooling
            integrated = torch.max(stacked_attentions, dim=0)[0]  # (batch_size, num_heads, max_seq_length, max_seq_length)
        else:
            # num_layers, batch_size, num_heads, max_seq_length, _ = attentions.shape[0]

            # Max pooling
            integrated = torch.max(attentions, dim=0)[0]  # shape: (batch_size, num_heads, max_seq_length, max_seq_length)

        # shape: (batch_size, num_heads, max_seq_length, max_seq_length)
        return integrated
    
    def _aggregate_attention(self, sentence_attentions):
        # input shape: (1, num_heads, num_sentences, num_sentences)
        max_attention_heads, _ = torch.max(sentence_attentions, dim=1)

        return max_attention_heads.squeeze(0) # shape: (num_sentences, num_sentences)

    def _remove_tags(self, text):
        # 태그 제거
        return re.sub(r'<[^>]+>', '', text)
    
    def _find_sentence_boundaries(self, sequences):
        sentences = []
        sentence_boundaries_text = [0]
        sentence_boundaries_ids = [0]
        tokens = self.tokenizer.convert_ids_to_tokens(sequences)

        for idx_i, token_i in enumerate(tokens):
            if token_i == '\n' or '\n' in token_i or idx_i == len(tokens)-1:
                text = self.tokenizer.decode(sequences[:idx_i+1])

                text_chunk = text[sentence_boundaries_text[-1]:]
                nltk_sentences = _safe_sent_tokenize(text_chunk)

                idx_nltk = 0
                idx_j = sentence_boundaries_ids[-1]
                while idx_j < idx_i+1:
                    if idx_nltk >= len(nltk_sentences):
                        break
                    nltk_sentence = nltk_sentences[idx_nltk]
                    slice_start = sentence_boundaries_ids[-1]
                    idx_j += 1

                    text_slice = self.tokenizer.decode(sequences[slice_start:idx_j+1])
                    if nltk_sentence in text_slice:
                        sentences.append(text_slice)
                        sentence_boundaries_text.append(len(self.tokenizer.decode(sequences[:idx_j+1])))
                        sentence_boundaries_ids.append(idx_j+1)
                        idx_nltk += 1
                        continue
        
        return sentence_boundaries_text, sentence_boundaries_ids, sentences

    def _calculate_sentence_attention(self, attentions, sentence_boundaries_ids):
        # 문장 간 attention 계산
        _, num_heads, _, seq_length = attentions.shape # shape: (1, num_heads, seq_length, seq_length)
        num_sentences = len(sentence_boundaries_ids) - 1 # 시작점과 끝점을 포함하므로 1 뺌

        # 0으로 초기화
        sentence_attentions = torch.zeros((1, num_heads, num_sentences, num_sentences))

        # for layer in range(num_layers):
        # layer = 0
        for head in range(num_heads):
            for i in range(num_sentences):
                start_i = sentence_boundaries_ids[i]
                end_i = sentence_boundaries_ids[i + 1]
                for j in range(num_sentences):
                    start_j = sentence_boundaries_ids[j]
                    end_j = sentence_boundaries_ids[j + 1]
                    # 문장 범위 내 텐서 추출
                    attention_slice = attentions[0, head, start_i:end_i, start_j:end_j]
                    
                    if attention_slice.numel() == 0: # 빈 텐서 검사
                        max_attention = 0
                    else:
                        max_attention = torch.max(attention_slice).item() # 문장 내 토큰들의 attention의 최대값
                    
                    sentence_attentions[0, head, i, j] = max_attention # 각 헤드 내에서 i번째 문장관 j번째 문장의 attention

        return sentence_attentions # shape: (1, num_heads, num_sentences, num_sentences)
    
    def get_sentence_token_attention(self, sentence_x_idx=None, sentence_y_idx=None):
        """
        BertViz 시각화에 필요한 데이터를 준비하는 함수
        
        Parameters:
            sentence_x_idx: 첫 번째 문장의 인덱스
            sentence_y_idx: 두 번째 문장의 인덱스
        
        Returns:
            attention_data: bertviz 포맷의 어텐션 데이터
            tokens: 토큰 리스트
            sentence_b_start: 두 번째 문장의 시작 인덱스 (해당하는 경우)
        """
        # 토큰 준비
        tokens = self.tokenizer.convert_ids_to_tokens(self.sequences)

        # 문장 경계 처리
        x_start = self.sentence_boundaries_ids[sentence_x_idx]
        x_end = self.sentence_boundaries_ids[sentence_x_idx + 1]
        y_start = self.sentence_boundaries_ids[sentence_y_idx]
        y_end = self.sentence_boundaries_ids[sentence_y_idx + 1]

        # 두 문장의 토큰 결합
        combined_tokens = tokens[x_start:x_end] + tokens[y_start:y_end]

        # 어텐션 데이터 준비
        if self.attention_type == 'dec':  # 생성 모델의 경우
            num_layers = len(self.attentions[0])
            attention_data = []

            for layer in range(num_layers):
                sent_x_self_attention = self.layer_attentions[layer, :, :, x_start:x_end, x_start:x_end]  # x->x
                sent_y_self_attention = self.layer_attentions[layer, :, :, y_start:y_end, y_start:y_end]  # y->y
                sent_x_to_y_attention = self.layer_attentions[layer, :, :, x_start:x_end, y_start:y_end]  # x->y
                sent_y_to_x_attention = self.layer_attentions[layer, :, :, y_start:y_end, x_start:x_end]  # y->x

                # Attention 결합
                sent_x_combined = torch.cat((sent_x_self_attention, sent_x_to_y_attention), dim=3)
                sent_y_combined = torch.cat((sent_y_to_x_attention, sent_y_self_attention), dim=3)
                combined_attention = torch.cat((sent_x_combined, sent_y_combined), dim=2)

                layer_attention = combined_attention[0].unsqueeze(0)  # batch 차원 추가
                attention_data.append(layer_attention)
        else:  # BERT 계열 모델의 경우
            attention_data = [att.unsqueeze(0) for att in self.attentions]  # batch 차원 추가

        return attention_data, combined_tokens, x_end-x_start
