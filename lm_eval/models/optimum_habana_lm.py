import torch
import logging
import torch.nn.functional as F
from importlib.util import find_spec
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
logger = logging.getLogger(__name__)


@register_model("oh", "optimum-habana")
class OptimumLM(HFLM):
    def __init__(self, device="hpu", **kwargs) -> None:
        self.hpu_device = device
        super().__init__(device=self.hpu_device, **kwargs)

    def _create_model(self, pretrained: str, ** kwargs) -> None:
        from oh_utils import initialize_model

        model_kwargs = kwargs if kwargs else {}
        model_kwargs["model_name_or_path"] = pretrained
        print(model_kwargs)

        model, _, tokenizer, generation_config = initialize_model(model_kwargs, logger)
        self.tokenizer = tokenizer
        self.model = model
        self.options = generation_config
        self._batch_size = model_kwargs.batch_size
        self.model_inputs = {"use_cache": self.options.use_cache}

        if self.model.config.model_type in [
            "llama",
            "mistral",
            "falcon",
            "phi",
            "mixtral",
            "qwen2",
            "gptj",
            "starcoder2",
            "baichuan"
        ]:
            self.model_inputs.update({"reuse_cache": self.options.reuse_cache})

        if self.model.config.model_type in ["llama", "mistral", "qwen2", "falcon", "starcoder2", "baichuan"]:
            if self.model.config.model_type != "falcon":
                self.model_inputs.update({"attn_softmax_bf16": self.options.attn_softmax_bf16})
            self.model_inputs.update(
                {
                    "use_flash_attention": self.options.use_flash_attention,
                    "flash_attention_recompute": self.options.flash_attention_recompute,
                    "flash_attention_causal_mask": self.options.flash_attention_causal_mask,
                }
            )

        if model_kwargs.warmup:
            self.warm_up()

    def warm_up(self):
        for bucket_size in reversed(self.buckets):
            inps = torch.ones((self._batch_size, bucket_size), dtype=torch.int64)
            self._model_call(inps)
            pass

    @property
    def eot_token_id(self):
        return self.model.config.eos_token_id

    @property
    def max_length(self):
        return self.buckets[-1]

    @property
    def max_gen_toks(self):
        raise NotImplementedError()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        # We need to do padding ourselves, otherwise we'll end up with recompilations
        # Returning 'cpu' to keep tensors on CPU in lm_eval code
        return "cpu"

    def tok_encode(self, string):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError()

    def find_bucket(self, length):
        return [b for b in self.buckets if b >= length][0]

    def _model_call(self, inps):
        bs, seq_length = inps.shape
        padding_length = 0
        if self.options.static_shapes:
            bucket_length = self.find_bucket(seq_length)
            if self.options.use_cache and self.options.reuse_cache:
                self.model.allocate_kv_cache(bs, bucket_length + 1, bucket_length)
            padding_length = bucket_length - seq_length
            inps = F.pad(inps, (0, padding_length), value=self.model.config.pad_token_id)
        logits = self.model(inps.to(self._device), **self.model_inputs)["logits"].cpu()

        if self.options.static_shapes and padding_length > 0:
            logits = logits[:, :-padding_length, :]
        logits = logits.to(torch.float32)
        return logits
