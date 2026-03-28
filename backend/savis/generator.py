from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.utils import logging as hf_logging

from .attention import (
    _default_model_dtype,
    _filter_tokenizer_kwargs,
    _default_model_runtime_kwargs,
    _model_input_device,
    _resolve_device,
    _resolve_model_source,
    prepare_document_text,
)


hf_logging.disable_progress_bar()


class TextGenerator:
    def __init__(self, model_name, **kwargs):
        self.device = _resolve_device()
        self.model_source, resolved_kwargs, _ = _resolve_model_source(model_name)

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

        self.model = AutoModelForCausalLM.from_pretrained(self.model_source, **model_kwargs)
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if "device_map" not in model_kwargs:
            self.model.to(self.device)
        self.model.eval()

    def generate_text(self, input_text, max_new_tokens=100, stop_newline=True):
        prepared_input = self._apply_sep_token(input_text)

        input_device = _model_input_device(self.model, self.device)
        input_ids = self.tokenizer(prepared_input, return_tensors="pt").input_ids.to(input_device)
        if stop_newline:
            stopping_criteria = StoppingCriteriaList([StopOnNewLineCriteria(self.tokenizer)])
        else:
            stopping_criteria = None
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            return_dict_in_generate=True,
            stopping_criteria=stopping_criteria,
            # repetition_penalty=1.2,
            # no_repeat_ngram_size=2,
        )
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        attentions = outputs.attentions
        return generated_text, attentions, self.tokenizer, input_ids, outputs

    def _apply_sep_token(self, text):
        return prepare_document_text(text, segment_sentences=True)

class StopOnNewLineCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # 마지막으로 생성된 토큰이 줄바꿈인지 확인
        if input_ids[0, -1] == self.tokenizer.eos_token_id or \
            self.tokenizer.decode(input_ids[0, -1]) == "\n":
            return True
        return False
