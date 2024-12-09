from typing import Optional, Union

import os
import copy
import shutil
import tempfile
import torch
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from optimum.habana.checkpoint_utils import (
    get_ds_injection_policy,
    model_is_optimized,
    model_on_meta,
    write_checkpoints_json,
)

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


def setup_distributed(args):
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "0"))
    args.global_rank = int(os.getenv("RANK", "0"))


def exclude_hpu_graph_configs(args):
    # Excluded configs for batch size 1 for hpu graph
    if args.batch_size == 1 and args.limit_hpu_graphs:
        if "falcon-180B" in args.model_name_or_path or "falcon-180b" in args.model_name_or_path:
            return False
        if args.world_size == 2 or args.world_size == 4 or args.world_size == 8:
            if args.max_input_tokens >= 4096 and args.max_new_tokens >= 128:
                return False
        return True
    else:
        return False


def setup_env(args):
    # TODO: SW-167588 - WA for memory issue in hqt prep_model
    os.environ.setdefault("EXPERIMENTAL_WEIGHT_SHARING", "FALSE")

    if args.global_rank == 0 and not args.torch_compile and args.show_graphs_count:
        os.environ.setdefault("GRAPH_VISUALIZATION", "true")
        shutil.rmtree(".graph_dumps", ignore_errors=True)

    if args.world_size > 0:
        os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
        os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")

    if args.use_hpu_graphs and args.limit_hpu_graphs and not args.reuse_cache and args.bucket_internal:
        # Based upon above conditions and below env variable,
        # we can call HPU graphs clear_inputs().
        os.environ.setdefault("PT_HPUGRAPH_DISABLE_TENSOR_CACHE", "1")

    # Tweak generation so that it runs faster on Gaudi
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    adapt_transformers_to_gaudi()


def setup_device(args):
    if args.device == "hpu":
        import habana_frameworks.torch.core as htcore
    return torch.device(args.device)


def setup_model(args, model_dtype, model_kwargs):
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype, **model_kwargs)
    model = model.eval().to(args.device)

    if args.use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        model = wrap_in_hpu_graph(model)

    return model


# patching LinearAllreduce to use ScopedLinearAllReduce
def patch_scoped_linear_all_reduce(model):
    from deepspeed.module_inject.layers import LinearAllreduce
    from optimum.habana.transformers.models.modeling_all_models import ScopedLinearAllReduce

    for name, module in model.named_children():
        if type(module) is LinearAllreduce:
            SL = ScopedLinearAllReduce(mod=module)
            setattr(model, name, SL)
        patch_scoped_linear_all_reduce(module)


def setup_distributed_model(args, model_dtype, model_kwargs):
    import deepspeed

    deepspeed.init_distributed(dist_backend="hccl")
    config = AutoConfig.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype, **model_kwargs)
    load_to_meta = model_on_meta(config)

    if load_to_meta:
        # Construct model with fake meta tensors, later will be replaced on devices during ds-inference ckpt load
        with deepspeed.OnDevice(dtype=model_dtype, device="meta"):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=model_dtype)

        # Model loaded to meta is managed differently
        checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="+w")

        write_checkpoints_json(
            args.model_name_or_path,
            args.local_rank,
            checkpoints_json,
            token=args.token,
        )
    else:
        # TODO: revisit placement on CPU when auto-injection is possible
        with deepspeed.OnDevice(dtype=model_dtype, device="cpu"):
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype, **model_kwargs)

    model.eval()

    # Initialize the model
    ds_inference_kwargs = {"dtype": model_dtype}
    ds_inference_kwargs["tensor_parallel"] = {"tp_size": args.world_size}
    ds_inference_kwargs["enable_cuda_graph"] = args.use_hpu_graphs
    ds_inference_kwargs["injection_policy"] = get_ds_injection_policy(config)
    if load_to_meta:
        ds_inference_kwargs["checkpoint"] = checkpoints_json.name

    model = deepspeed.init_inference(model, **ds_inference_kwargs)
    model = model.module

    if model.config.model_type in ["llama", "falcon", "qwen2", "starcoder2", "gemma"]:
        patch_scoped_linear_all_reduce(model)

    return model


def setup_tokenizer(args, model):
    tokenizer_kwargs = {
        "revision": args.model_revision,
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
    }

    if args.bad_words is not None or args.force_words is not None:
        tokenizer_kwargs["add_prefix_space"] = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    if not model.config.is_encoder_decoder:
        tokenizer.padding_side = "left"

    if model.config.model_type == "llama":
        if model.generation_config.pad_token_id is None:
            if isinstance(model.generation_config.eos_token_id, int):
                model.generation_config.pad_token_id = model.generation_config.eos_token_id
            elif isinstance(model.generation_config.eos_token_id, list):
                model.generation_config.pad_token_id = model.generation_config.eos_token_id[0]
        tokenizer.bos_token_id = model.generation_config.bos_token_id
        if isinstance(model.generation_config.eos_token_id, int):
            tokenizer.eos_token_id = model.generation_config.eos_token_id
        elif isinstance(model.generation_config.eos_token_id, list):
            tokenizer.eos_token_id = model.generation_config.eos_token_id[0]
        tokenizer.pad_token_id = model.generation_config.pad_token_id
        tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
    if model.config.model_type == "persimmon":
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        tokenizer.bos_token_id = model.generation_config.bos_token_id
        tokenizer.eos_token_id = model.generation_config.eos_token_id
        tokenizer.pad_token_id = model.generation_config.pad_token_id
        tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)

    # HACK: MiniCPM3 does not support list EOS token ID generation config.
    if model.config.model_type == "minicpm3" and isinstance(model.generation_config.eos_token_id, list):
        model.generation_config.eos_token_id = model.generation_config.eos_token_id[-1]

    # Some models like GPT2 do not have a PAD token so we have to set it if necessary
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return tokenizer, model


def setup_generation_config(args, model, tokenizer):
    bad_words_ids = None
    force_words_ids = None
    if args.bad_words is not None:
        bad_words_ids = [tokenizer.encode(bad_word, add_special_tokens=False) for bad_word in args.bad_words]
    if args.force_words is not None:
        force_words_ids = [tokenizer.encode(force_word, add_special_tokens=False) for force_word in args.force_words]

    is_optimized = model_is_optimized(model.config)

    # Generation configuration
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.use_cache = args.use_kv_cache
    generation_config.static_shapes = is_optimized
    generation_config.bucket_size = args.bucket_size if is_optimized else -1
    generation_config.bucket_internal = args.bucket_internal
    generation_config.do_sample = args.do_sample
    generation_config.num_beams = args.num_beams
    generation_config.top_k = args.top_k
    generation_config.penalty_alpha = args.penalty_alpha
    generation_config.bad_words_ids = bad_words_ids
    generation_config.force_words_ids = force_words_ids
    generation_config.num_return_sequences = args.num_return_sequences
    generation_config.trim_logits = args.trim_logits
    generation_config.attn_softmax_bf16 = args.attn_softmax_bf16
    generation_config.limit_hpu_graphs = args.limit_hpu_graphs
    generation_config.reuse_cache = args.reuse_cache
    generation_config.reduce_recompile = args.reduce_recompile
    if generation_config.reduce_recompile:
        assert generation_config.bucket_size > 0
    generation_config.use_flash_attention = args.use_flash_attention
    generation_config.flash_attention_recompute = args.flash_attention_recompute
    generation_config.flash_attention_causal_mask = args.flash_attention_causal_mask
    generation_config.flash_attention_fast_softmax = args.flash_attention_fast_softmax
    generation_config.trust_remote_code = args.trust_remote_code
    generation_config.valid_sequence_lengths = None

    return generation_config


def initialize_model(args):
    setup_distributed(args)
    if exclude_hpu_graph_configs(args):
        args.limit_hpu_graphs = False
    setup_env(args)
    setup_device(args)

    use_deepspeed = args.world_size > 0
    if use_deepspeed or args.bf16:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float
        args.attn_softmax_bf16 = False

    model_kwargs = {
        "revision": args.model_revision,
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
    }

    model = (setup_model(args, model_dtype, model_kwargs) if not use_deepspeed else setup_distributed_model(args, model_dtype, model_kwargs))

    tokenizer, model = setup_tokenizer(args, model)
    generation_config = setup_generation_config(args, model, tokenizer)

    return model, tokenizer, generation_config


class Args(object):
    device = "hpu"
    model_name_or_path = ""
    bf16 = True
    max_new_tokens = 100
    max_input_tokens = 0
    batch_size = 1
    warmup = 3
    n_iterations = 5
    reuse_cache = True
    use_kv_cache = True
    use_hpu_graphs = True
    attn_softmax_bf16 = True
    limit_hpu_graphs = True
    do_sample = True
    torch_compile = False
    show_graphs_count = False
    bucket_internal = True
    bucket_size = 128
    trust_remote_code = True
    num_beams = 1
    bad_words = None
    force_words = None
    token = None
    top_k = None
    num_return_sequences = 1
    trim_logits = True
    penalty_alpha = None
    reduce_recompile = False
    use_flash_attention = True
    flash_attention_recompute = False
    flash_attention_causal_mask = False
    flash_attention_fast_softmax = False
    model_revision = "main"
    buckets = [16, 32, 64, 128, 189, 284, 384]


@register_model("oh", "optimum-habana")
class OptimumHabanaLM(HFLM):
    def __init__(self, pretrained: str, batch_size: Optional[Union[int, str]] = 1, **kwargs) -> None:
        args = Args()

        for key in kwargs:
            setattr(args, key, kwargs[key])

        args.model_name_or_path = pretrained
        args.batch_size = batch_size

        model, tokenizer, generation_config = initialize_model(args)
        super().__init__(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

        self._model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self._device = args.device
        self.buckets = sorted(args.buckets)

        self.model_inputs = {"use_cache": self.generation_config.use_cache}
        if self._model.config.model_type in [
            "llama",
            "mistral",
            "falcon",
            "phi",
            "mixtral",
            "qwen2",
            "gptj",
            "starcoder2",
            "gemma",
            "baichuan",
        ]:
            self.model_inputs.update({"reuse_cache": self.generation_config.reuse_cache})

        if self._model.config.model_type in ["llama", "mistral", "qwen2", "falcon", "starcoder2", "gemma", "baichuan"]:
            if self._model.config.model_type != "falcon":
                self.model_inputs.update({"attn_softmax_bf16": self.generation_config.attn_softmax_bf16})
            self.model_inputs.update(
                {
                    "use_flash_attention": self.generation_config.use_flash_attention,
                    "flash_attention_recompute": self.generation_config.flash_attention_recompute,
                    "flash_attention_causal_mask": self.generation_config.flash_attention_causal_mask,
                }
            )

        if args.warmup:
            self.warm_up()

    def warm_up(self):
        for bucket_size in reversed(self.buckets):
            inps = torch.ones((self.batch_size, bucket_size), dtype=torch.int64)
            self._model_call(inps)
            pass

    @property
    def max_length(self):
        return self.buckets[-1]

    def find_bucket(self, length):
        return [b for b in self.buckets if b >= length][0]

    def _model_call(self, inps, attn_mask=None, labels=None):
        bs, seq_length = inps.shape
        padding_length = 0
        if self.generation_config.static_shapes:
            bucket_length = self.find_bucket(seq_length)
            if self.generation_config.use_cache and self.generation_config.reuse_cache:
                self._model.allocate_kv_cache(bs, bucket_length + 1, bucket_length)
            padding_length = bucket_length - seq_length
            inps = F.pad(inps, (0, padding_length), value=self._model.config.pad_token_id)
        logits = self._model(inps.to(self._device), **self.model_inputs)["logits"].cpu()

        if self.generation_config.static_shapes and padding_length > 0:
            logits = logits[:, :-padding_length, :]
        logits = logits.to(torch.float32)
        return logits
