from typing import Optional

import os
import copy
import shutil
import tempfile
import torch

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
        if args.quant_config:
            htcore.hpu_set_env()
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


@register_model("optimum-habana")
class HabanaModelAdapter(HFLM):
    def __init__(self, pretrained: str, batch_size: Optional[int] = 1, **kwargs) -> None:
        super().__init__(pretrained=pretrained, batch_size=batch_size)

        model, tokenizer, generation_config = initialize_model(kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

        self._device = torch.device("hpu")

    def _model_call(self, inps, attn_mask=None, labels=None):
        logits = self.model(inps.to(self._device), **self.generation_config)["logits"].cpu()
        logits = logits.to(torch.float32)
        return logits
