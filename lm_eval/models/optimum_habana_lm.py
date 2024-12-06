import os
import torch
import logging
import argparse
import torch.nn.functional as F
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
logger = logging.getLogger(__name__)


def setup_parser(parser):
    # Arguments management
    parser.add_argument("--device", "-d", type=str, choices=["hpu"], help="Device to run", default="hpu")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model (on the HF Hub or locally).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=0,
        help="If > 0 then pad and truncate the input sequences to this specified length of tokens. \
            if == 0, then truncate to 16 (original default) \
            if < 0, then do not truncate, use full input prompt",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of inference iterations for benchmarking.")
    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="Local process rank.")
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        help="Optional argument if you want to assess your model on a given dataset of the HF Hub.",
    )
    parser.add_argument(
        "--column_name",
        default=None,
        type=str,
        help="If `--dataset_name` was given, this will be the name of the column to use as prompts for generation.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling for generation.",
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="Number of beams used for beam search generation. 1 means greedy search will be performed.",
    )
    parser.add_argument(
        "--top_k",
        default=None,
        type=int,
        help="Size of candidate set used for re-ranking in contrastive search. top_k > 1 enables contrastive search.",
    )
    parser.add_argument(
        "--penalty_alpha",
        default=None,
        type=float,
        help="Degeneration penalty for contrastive search. penalty_alpha > 0 enables contrastive search.",
    )
    parser.add_argument(
        "--trim_logits",
        action="store_true",
        help="Calculate logits only for the last token to save memory in the first step.",
    )
    parser.add_argument(
        "--seed",
        default=27,
        type=int,
        help="Seed to use for random generation. Useful to reproduce your runs with `--do_sample`.",
    )
    parser.add_argument(
        "--profiling_warmup_steps",
        default=0,
        type=int,
        help="Number of steps to ignore for profiling.",
    )
    parser.add_argument(
        "--profiling_steps",
        default=0,
        type=int,
        help="Number of steps to capture for profiling.",
    )
    parser.add_argument(
        "--profiling_record_shapes",
        default=False,
        type=bool,
        help="Record shapes when enabling profiling.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        type=str,
        nargs="*",
        help='Optional argument to give a prompt of your choice as input. Can be a single string (eg: --prompt "Hello world"), or a list of space-separated strings (eg: --prompt "Hello world" "How are you?")',
    )
    parser.add_argument(
        "--bad_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that are not allowed to be generated.",
    )
    parser.add_argument(
        "--force_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that must be generated.",
    )
    parser.add_argument(
        "--assistant_model",
        default=None,
        type=str,
        help="Optional argument to give a path to a draft/assistant model for assisted decoding.",
    )
    parser.add_argument(
        "--peft_model",
        default=None,
        type=str,
        help="Optional argument to give a path to a PEFT model.",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--token",
        default=None,
        type=str,
        help="The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
        "generated when running `huggingface-cli login` (stored in `~/.huggingface`).",
    )
    parser.add_argument(
        "--model_revision",
        default="main",
        type=str,
        help="The specific model version to use (can be a branch name, tag name or commit id).",
    )
    parser.add_argument(
        "--attn_softmax_bf16",
        action="store_true",
        help="Whether to run attention softmax layer in lower precision provided that the model supports it and "
        "is also running in lower precision.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output directory to store results in.",
    )
    parser.add_argument(
        "--bucket_size",
        default=-1,
        type=int,
        help="Bucket size to maintain static shapes. If this number is negative (default is -1) \
            then we use `shape = prompt_length + max_new_tokens`. If a positive number is passed \
            we increase the bucket in steps of `bucket_size` instead of allocating to max (`prompt_length + max_new_tokens`).",
    )
    parser.add_argument(
        "--bucket_internal",
        action="store_true",
        help="Split kv sequence into buckets in decode phase. It improves throughput when max_new_tokens is large.",
    )
    parser.add_argument(
        "--dataset_max_samples",
        default=-1,
        type=int,
        help="If a negative number is passed (default = -1) perform inference on the whole dataset, else use only `dataset_max_samples` samples.",
    )
    parser.add_argument(
        "--limit_hpu_graphs",
        action="store_true",
        help="Skip HPU Graph usage for first token to save memory",
    )
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        help="Whether to reuse key/value cache for decoding. It should save memory.",
    )
    parser.add_argument("--verbose_workers", action="store_true", help="Enable output from non-master workers")
    parser.add_argument(
        "--simulate_dyn_prompt",
        default=None,
        type=int,
        nargs="*",
        help="If empty, static prompt is used. If a comma separated list of integers is passed, we warmup and use those shapes for prompt length.",
    )
    parser.add_argument(
        "--reduce_recompile",
        action="store_true",
        help="Preprocess on cpu, and some other optimizations. Useful to prevent recompilations when using dynamic prompts (simulate_dyn_prompt)",
    )

    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Whether to enable Habana Flash Attention, provided that the model supports it.",
    )
    parser.add_argument(
        "--flash_attention_recompute",
        action="store_true",
        help="Whether to enable Habana Flash Attention in recompute mode on first token generation. This gives an opportunity of splitting graph internally which helps reduce memory consumption.",
    )
    parser.add_argument(
        "--flash_attention_causal_mask",
        action="store_true",
        help="Whether to enable Habana Flash Attention in causal mode on first token generation.",
    )
    parser.add_argument(
        "--flash_attention_fast_softmax",
        action="store_true",
        help="Whether to enable Habana Flash Attention in fast softmax mode.",
    )
    parser.add_argument(
        "--book_source",
        action="store_true",
        help="Whether to use project Guttenberg books data as input. Usefull for testing large sequence lenghts.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Whether to use torch compiled model or not.",
    )
    parser.add_argument(
        "--ignore_eos",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to disable stopping with eos token when calling `generate`. --no-ignore_eos to disable it",
    )
    parser.add_argument("--temperature", default=1.0, type=float, help="Temperature value for text generation")
    parser.add_argument("--top_p", default=1.0, type=float, help="Top_p value for generating text via sampling")
    parser.add_argument(
        "--const_serialization_path",
        "--csp",
        type=str,
        help="Path to serialize const params. Const params will be held on disk memory instead of being allocated on host memory.",
    )
    parser.add_argument(
        "--disk_offload",
        action="store_true",
        help="Whether to enable device map auto. In case no space left on cpu, weights will be offloaded to disk.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust the execution of code from datasets/models defined on the Hub. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.",
    )
    parser.add_argument(
        "--load_quantized_model",
        action="store_true",
        help="Whether to load model from hugging face checkpoint.",
    )
    parser.add_argument(
        "--parallel_strategy",
        type=str,
        choices=["tp", "none"],  # Add other strategies as needed
        default="none",
        help="Run multi card with the specified parallel strategy. Choices are 'tp' for Tensor Parallel Strategy or 'none'.",
    )
    parser.add_argument(
        "--use_dynamic_moe",
        action="store_true",
        help="Whether to use dynamic MoE kernel.",
    )


def setup_lm_eval_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Evaluation script for HPU")

    parser.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        help="Input length buckets to use with static_shapes",
        default=[16, 32, 64, 128, 189, 284],
    )

    parser.add_argument(
        "--output_file", "-o", type=str, help="Output file with end results and runtime parameters", required=True
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Tasks to run",
        default=["hellaswag", "lambada_openai", "piqa", "winogrande"],
    )
    parser.add_argument("--limit_iters", type=int, help="limit examples to run that many iterations", default=None)
    setup_parser(parser)
    return parser


@register_model("oh", "optimum-habana")
class OptimumLM(HFLM):
    def __init__(self, device="hpu", **kwargs) -> None:
        self.hpu_device = device
        super().__init__(device=self.hpu_device, **kwargs)

    def _create_model(self, ** kwargs) -> None:
        from lm_eval.models.oh_utils import initialize_model

        parser = setup_lm_eval_parser()
        arg_str = ['--model_name_or_path=/hf/hf_models/llama-2-7b-chat-hf/']
        print(arg_str)
        model_kwargs = parser.parse_args(arg_str)
        print(model_kwargs)

        if model_kwargs.torch_compile:
            model_kwargs.use_hpu_graphs = False

        if not model_kwargs.use_hpu_graphs:
            model_kwargs.limit_hpu_graphs = False

        if model_kwargs.use_flash_attention and not model_kwargs.flash_attention_fast_softmax:
            model_kwargs.flash_attention_fast_softmax = True

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
