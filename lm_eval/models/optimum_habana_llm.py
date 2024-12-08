from importlib.util import find_spec
from pathlib import Path

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

from lm_eval.api.model import *
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance
from tqdm import tqdm
from huggingface_hub import HfApi

from lm_eval import utils
from lm_eval.models.utils import (
    Collator,
    configure_pad_token,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)

import copy
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer, PreTrainedTokenizerFast

try:
    HABANA_AVAILABLE = True
    import habana_frameworks.torch.hpu as torch_hpu
    import habana_frameworks.torch.core as htcore
    from optimum.habana.checkpoint_utils import (
        model_is_optimized,
    )
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
except ModuleNotFoundError:
    HABANA_AVAILABLE = False

eval_logger = utils.eval_logger


@register_model("oh")
class OptimumHabana_HF(TemplateLM):
    """
    Enable optimum-habana
    """
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str = "/host/mnt/disk3/hf_models/qwen1.5-moe-a2.7b-chat",
        device: Optional[str] = "hpu",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        logits_cache: bool = False,
        tokenizer: Optional[
            Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast]
        ] = None,
        backend: Literal["default", "causal", "seq2seq"] = "default",
        use_fast_tokenizer: Optional[bool] = True,
        truncation: Optional[bool] = False,
        batch_size: Optional[int] = 1,
        trust_remote_code: Optional[bool] = False,
        max_length: Optional[int] = None,
        max_batch_size: Optional[int] = 64,
        add_bos_token: Optional[bool] = True,
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        # the followings are the Habana only options:
        use_kv_cache: Optional[bool] = True,
        use_hpu_graphs: Optional[bool] = True,
        use_lazy_mode: Optional[bool] = True,
        attn_softmax_bf16: Optional[bool] = False,
        bucket_size: Optional[int] = 256,
        bucket_internal: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        limit_hpu_graphs: Optional[bool] = True,
        trim_logits: Optional[bool] = True,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        reduce_recompile: Optional[bool] = False,
        **kwargs,
    ) -> None:
        if not HABANA_AVAILABLE:
            raise ImportError(
                "Only run in Gaudi server.",
            )
        # Yun inherit from LM
        super().__init__()
        adapt_transformers_to_gaudi()

        print(f"Yun: bucket_size as parameter: {bucket_size}")
        self._device = torch.device(device)
        self.logits_cache = logits_cache
        self.tokenizer = None
        self.backend = backend
        self._batch_size = 1
        self.use_lazy_mode = use_lazy_mode
        self.use_hpu_graphs = use_hpu_graphs
        self._max_length = max_length

        revision = "main"
        # revision = revision + ("/" + subfolder if subfolder is not None else "")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_tokenizer,
        )

        # Yun only support causal (llama, qwen, glm, etc.)
        self.backend = "causal"
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=get_dtype(dtype),
            trust_remote_code=trust_remote_code,
            # Yun TODO: add GaudiConfig, and other parameters..
        )
        self.model = self.model.eval().to(self.device)

        # TODO: Generation_config was init in _model_generate(), hardcoded.
        # self.generation_config = copy.deepcopy(self.model.generation_config)
        # self.generation_config.max_new_tokens = max_length
        # self.generation_config.use_cache = True
        # # assert the model is optimized
        # #self.generation_config.static_shapes = True
        # self.generation_config.bucket_size = 128
        # self.generation_config.bucket_internal = True
        # self.generation_config.do_sample = False
        # #self.generation_config.temperature = 0
        # #self.generation_config.num_beams = args.num_beams
        # #self.generation_config.top_k = args.top_k
        # #self.generation_config.penalty_alpha = args.penalty_alpha
        # #self.generation_config.bad_words_ids = bad_words_ids
        # #self.generation_config.force_words_ids = force_words_ids
        # #self.generation_config.num_return_sequences = args.num_return_sequences
        # self.generation_config.trim_logits = True
        # self.generation_config.attn_softmax_bf16 = True
        # self.generation_config.limit_hpu_graphs = False
        # self.generation_config.reuse_cache = True
        # self.generation_config.reduce_recompile = False
        # if self.generation_config.reduce_recompile:
        #     assert self.generation_config.bucket_size > 0
        # self.generation_config.use_flash_attention = True
        # self.generation_config.flash_attention_recompute = False
        # self.generation_config.flash_attention_causal_mask = False
        # self.generation_config.flash_attention_fast_softmax = False
        # self.generation_config.trust_remote_code = True
        if self.use_hpu_graphs:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph
            self.model = wrap_in_hpu_graph(self.model, disable_tensor_cache=True)

        self.truncation = False
        # self.logits_cache = logits_cache
        # select (or create) a pad token to use
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)

        if self.model.config.model_type == "llama":
            if self.model.generation_config.pad_token_id is None:
                if isinstance(self.model.generation_config.eos_token_id, int):
                    self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
                elif isinstance(self.model.generation_config.eos_token_id, list):
                    self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id[0]
            self.tokenizer.bos_token_id = self.model.generation_config.bos_token_id
            if isinstance(self.model.generation_config.eos_token_id, int):
                self.tokenizer.eos_token_id = self.model.generation_config.eos_token_id
            elif isinstance(self.model.generation_config.eos_token_id, list):
                self.tokenizer.eos_token_id = self.model.generation_config.eos_token_id[0]
            self.tokenizer.pad_token_id = self.model.generation_config.pad_token_id
            self.tokenizer.pad_token = self.tokenizer.decode(self.tokenizer.pad_token_id)
            self.tokenizer.eos_token = self.tokenizer.decode(self.tokenizer.eos_token_id)
            self.tokenizer.bos_token = self.tokenizer.decode(self.tokenizer.bos_token_id)
        self.add_bos_token = add_bos_token

        self.batch_schedule = 1
        self._batch_size = int(batch_size)
        self.max_batch_size = max_batch_size
        print("@@@ Yun: Init optimum-habana. ")

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self.model.config

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        return self.tokenizer.bos_token_id or self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    # Yun, inherit from huggingface.py, TODO: why dont let lm_eval set the max_gen_tokens?

    @property
    def max_gen_toks(self) -> int:
        return 512

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    # TODO: single HPU only
    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _model_generate(self, context, attention_mask, stop, **generation_kwargs):
        # build stopping criteria
        # !!! TODO: Yun if add stopping_criteria, Llama3-8b-instruct doesn't work, output output 1 tokn.
        # stopping_criteria = stop_sequences_criteria(
        #    self.tokenizer, stop, context.shape[1], context.shape[0]
        # )

        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.max_new_tokens = 512
        generation_config.do_sample = False
        # generation_config.temperature = 0.7
        # generation_config.top_p = 0.8
        # generation_config.top_k = 20
        # generation_config.repetition_penalty = 1.05
        generation_config.use_flash_attention = False  # True
        generation_config.bucket_size = 256
        generation_config.bucket_internal = False
        generation_config.reuse_cache = False
        generation_config.trim_logits = True
        generation_config.ignore_eos = False
        generation_config.static_shapes = True
        generation_config.limit_hpu_graphs = True
        # print(f" GGGGGGG Yun generation_config: {generation_config}")
        # print(f" GGGGGGG Yun config: {self.config}")

        cont = self.model.generate(
            input_ids=context,
            attention_mask=attention_mask,
            generation_config=generation_config,
            ignore_eos=True,
            # Disabled by Yun, otherwise llama3.1 output only 1 token
            # stopping_criteria=stopping_criteria,
            lazy_mode=True,  # self.use_lazy_mode,
            hpu_graphs=True,  # self.use_hpu_graphs,
            pad_token_id=self.tokenizer.pad_token_id,
        ).cpu()
        # print(f" $$$$$$$$$$$$$$$$$$$ resps: {cont}")
        from habana_frameworks.torch.hpu.metrics import metric_global
        gc_metric = metric_global("graph_compilation")
        print(f"Yun graph_compilation statistics: ")
        print(gc_metric.stats())
        print(" %%%%%%%%%%%%%%%%%%%%%%%% ")
        return cont

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            if self.backend == "causal":
                special_tokens_kwargs = {
                    "add_special_tokens": False or self.add_bos_token
                }
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode_plus(string,
                                              padding="longest",
                                              pad_to_multiple_of=256,
                                              # max_length=self.max_length,
                                              **special_tokens_kwargs)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self.backend == "causal":
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        if isinstance(strings, tuple):
            print(f"Warning: strings is a tuple, not List[], wired.")
            strings = list(strings)
        encoding = self.tokenizer.batch_encode_plus(
            strings,
            padding="longest",
            pad_to_multiple_of=256,
            # padding="max_length",
            # max_length=2048,
            truncation=True,
            return_tensors="pt",
            **add_special_tokens,
        )
        print(f" TTT Before left truncate Yun batched encode shape={encoding['input_ids'].shape}")
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]
        self.tokenizer.padding_side = old_padding_side
        print(f" TTT Yun batched encode shape={encoding['input_ids'].shape}")

        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        def _collate(req: Tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        batch_size = self.batch_size
        batch_fn = None

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        print(f" Batch size: {batch_size}")
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            # add EOS token to stop sequences
            eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            if self.backend == "causal":
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
            elif self.backend == "seq2seq":
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length

            print(f"Yun max_ctx_len/left_truncate_len = {max_ctx_len}")
            # encode, pad, and truncate contexts for this batch
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            # perform batched generation
            cont = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                **kwargs,
            )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                if self.backend == "causal":
                    cont_toks = cont_toks[context_enc.shape[1]:]

                s = self.tok_decode(cont_toks)

                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        s = s.split(term)[0]

                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        loglikelihoods = []

        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        for (string,) in tqdm(
            [req.args for req in requests], disable=(disable_tqdm or (self.rank != 0))
        ):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            pad_amnt = 0
            if self.world_size > 1:
                # We pad out the external document-level iterator so the inner iterator doesn't hang
                mytensor = torch.tensor(len(rolling_token_windows), device=self.device)
                gathered = (
                    self.accelerator.gather(mytensor).cpu().detach().numpy().tolist()
                )

                pad_amnt = max(gathered) - gathered[self.rank]
                if pad_amnt > 0:
                    rolling_token_windows += pad_amnt * [rolling_token_windows[0]]

            string_nll = self._loglikelihood_tokens(
                requests=rolling_token_windows,
                disable_tqdm=True,
                override_bs=adaptive_batch_size,
            )

            if (self.world_size > 1) and (pad_amnt > 0):
                string_nll = [x[0] for x in string_nll[:-pad_amnt]]
            else:
                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

            # cache this loglikelihood_rolling request
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts"
            if self.backend == "causal" and self.logits_cache
            else None,
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                if self.backend == "causal":
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape
                elif self.backend == "seq2seq":
                    inp = torch.tensor(
                        (context_enc)[-self.max_length:],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape

                    # build encoder attn masks
                    encoder_attns.append(torch.ones_like(inp))

                    cont = torch.tensor(
                        (continuation_enc)[-self.max_length:],
                        # TODO: left-shift these?
                        # TODO: our code assumes we never end up truncating conts for either model type
                        dtype=torch.long,
                        device=self.device,
                    )
                    (contlen,) = cont.shape

                    conts.append(cont)

                    padding_len_cont = (
                        max(padding_len_cont, contlen)
                        if padding_len_cont is not None
                        else contlen
                    )

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            if self.backend == "causal":
                batched_inps = pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )  # [batch, padding_len_inp]
            elif self.backend == "seq2seq":
                # TODO: left-pad encoder inps and mask?
                batched_inps = pad_and_concat(
                    padding_len_inp, inps
                )  # [batch, padding_len_inp]
                batched_conts = pad_and_concat(
                    padding_len_cont, conts
                )  # [batch, padding_len_cont]
                batched_encoder_mask = pad_and_concat(
                    padding_len_inp, encoder_attns
                )  # [batch, padding_len_inp]
                call_kwargs = {
                    "attn_mask": batched_encoder_mask,
                    "labels": batched_conts,
                }

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs), dim=-1
            )  # [batch, padding_length (inp or cont), vocab]

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if self.backend == "causal"
                    else None
                )
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    if request_str is not None:
                        # special case: loglikelihood_rolling produces a number of loglikelihood requests
                        # all with cache key None. instead do add_partial on the per-example level
                        # in the loglikelihood_rolling() function for those.
                        self.cache_hook.add_partial(
                            "loglikelihood", request_str, answer
                        )
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )

    def get_model_info(self) -> dict:
        """
        Method to get Hugging Face model information for experiment reproducibility.
        """

        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            else:
                return ""

        def get_model_sha(pretrained: str, revision: str) -> str:
            try:
                model_info = HfApi().model_info(repo_id=pretrained, revision=revision)
                return model_info.sha
            except Exception as e:
                eval_logger.warn(
                    f"Failed to get model SHA for {pretrained} at revision {revision}. Error: {e}"
                )
                return ""

        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
            "model_revision": self.revision,
            "model_sha": get_model_sha(self.pretrained, self.revision),
        }
        if self.peft:
            model_info["peft_sha"] = get_model_sha(self.peft, self.revision)
        if self.delta:
            model_info["delta_sha"] = get_model_sha(self.delta, self.revision)
        return model_info
