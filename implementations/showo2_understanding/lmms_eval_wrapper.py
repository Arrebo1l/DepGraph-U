import torch
import logging
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from accelerate import Accelerator, DistributedType
from omegaconf import OmegaConf
from typing import List, Optional, Union, Tuple


from lmms_eval.models.showo2_utils import Showo2Qwen2_5, WanVAE
from lmms_eval.models.showo2_utils.misc import get_text_tokenizer
from lmms_eval.models.showo2_utils.utils import image_transform
from lmms_eval.models.showo2_utils.omni_attention import omni_attn_mask_naive, full_attn_mask_naive

import os
import json
import time
import hashlib
from collections import OrderedDict
def load_state_dict(model_path):
    if model_path.endswith(".bin"):
        state_dict = torch.load(model_path)
    else:
        checkpoint_files = sorted(
            [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.bin')]
        )

        state_dict = OrderedDict()
        for checkpoint_file in checkpoint_files:
            print(f"Loading checkpoint: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file)
            state_dict.update(checkpoint)

    return state_dict

import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")

SYSTEM_PROMPT = "system\nYou are a helpful assistant.<|im_end|>"

class UnifiedShowoModel(torch.nn.Module):
    def __init__(self,
                 config_file,
                 tokenizer,
                 ckpt_path,
                 showo_token_ids,
                 ):
        super().__init__()

        # Load config
        self.yaml_conf = OmegaConf.load(config_file)

        self.ids = showo_token_ids
        self.tokenizer = tokenizer
        self.yaml_conf.model.showo.llm_vocab_size = len(tokenizer)
        self.num_image_tokens = self.yaml_conf.dataset.preprocessing.num_image_tokens
        self.sys_prompt_ids = self.tokenizer(SYSTEM_PROMPT, add_special_tokens=False)['input_ids']
        self.role_a = self.tokenizer("\n<|im_start|>user\n", add_special_tokens=False)['input_ids']
        self.role_b = self.tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)['input_ids']

        self.vae_model = WanVAE(vae_pth=self.yaml_conf.model.vae_model.pretrained_model_path, dtype=torch.float32)

        self.showo = Showo2Qwen2_5(**self.yaml_conf.model.showo)
        if self.yaml_conf.model.showo.load_from_showo:
            self.showo = Showo2Qwen2_5.from_pretrained(self.yaml_conf.model.showo.pretrained_model_path, use_safetensors=False)
        else:
            self.showo = Showo2Qwen2_5(**self.yaml_conf.model.showo)
            state_dict = load_state_dict(self.yaml_conf.model_path)
            self.showo.load_state_dict(state_dict)
        self.showo.requires_grad_(False)
        self.showo.eval()

        self._flashu_und_enabled = os.environ.get("FLASHU_UNDERSTANDING", "0") == "1"
        self._flashu_wanda_calibrate = os.environ.get("FLASHU_UNDERSTANDING_CALIBRATE", "0") == "1"
        self._flashu_wanda_samples = 0
        self._flashu_wanda_limit = int(os.environ.get("FLASHU_UNDERSTANDING_NCALIB", "20"))
        self._flashu_timing_file = os.environ.get("FLASHU_UNDERSTANDING_TIMING_FILE")
        self._flashu_profile = os.environ.get("FLASHU_UNDERSTANDING_PROFILE", "0") == "1"

        if self._flashu_wanda_calibrate:
            from flashu.wanda_pruner import WandaPruner
            self._flashu_wanda_pruner = WandaPruner(self.showo.showo)
            print(
                f"[FlashU-UND] Wanda calibration enabled for "
                f"{self._flashu_wanda_limit} samples"
            )

        if self._flashu_und_enabled:
            from flashu.understanding import (
                UnderstandingConfig,
                apply_permanent_ffn_pruning,
                patch_understanding_forward,
            )
            score_path = os.environ["FLASHU_UNDERSTANDING_SCORES"]
            scores = torch.load(score_path, map_location="cpu", weights_only=True)
            config = UnderstandingConfig(
                ffn_prune_ratio=float(os.environ.get("FLASHU_UNDERSTANDING_RP", "0.10")),
                layer_skip_ratio=float(os.environ.get("FLASHU_UNDERSTANDING_RLS", "0.10")),
                layer_skip_interval=int(os.environ.get("FLASHU_UNDERSTANDING_TLS", "10")),
                token_prune_layer=int(os.environ.get("FLASHU_UNDERSTANDING_LPRUNE", "2")),
                visual_token_prune_ratio=float(os.environ.get("FLASHU_UNDERSTANDING_RHO", "0.50")),
                visual_token_keep_mode=os.environ.get(
                    "FLASHU_UNDERSTANDING_VNORM_KEEP", "low"
                ).strip().lower(),
            )
            self._flashu_pruning_report = apply_permanent_ffn_pruning(
                self.showo.showo,
                scores,
                ratio=config.ffn_prune_ratio,
                friendly_multiple=config.friendly_multiple,
            )
            # KV-cache-compatible FlashU subset: permanent FFN pruning changes
            # only channel widths, so the stock Qwen cache remains valid.  The
            # reconstructed token-pruning and DLS forward is needed only when
            # either of those two mechanisms is active.  Avoiding that forward
            # is essential because it deliberately rejects past_key_values and
            # because layer-wise token removal / intermittent layer skipping
            # produce heterogeneous or incomplete caches.
            needs_custom_forward = (
                config.layer_skip_ratio > 0.0
                or config.visual_token_prune_ratio > 0.0
            )
            if needs_custom_forward:
                patch_understanding_forward(self.showo, config)
            else:
                print(
                    "[FlashU-UND] KV-compatible mode: permanent FFN pruning "
                    "enabled; token pruning and DLS disabled; using stock "
                    "Qwen forward for past_key_values support"
                )
            print(f"[FlashU-UND] acceleration enabled: {config}")

    def forward(self, v_tsr, input_ids, max_new_tokens, top_k,device):

        if self._flashu_timing_file:
            torch.cuda.synchronize()
            flashu_start = time.perf_counter()
        profile_events = None
        if self._flashu_profile:
            profile_events = {
                name: torch.cuda.Event(enable_timing=True)
                for name in ("start", "after_vae", "qwen_start", "qwen_end")
            }
            profile_events["start"].record()

        # vae encoding
        image_latents = self.vae_model.sample(v_tsr.unsqueeze(2)).squeeze(2)
        if profile_events is not None:
            profile_events["after_vae"].record()

        image_embeds_und = self.showo.image_embedder_und(image_latents)
        image_embeds_gen = self.showo.image_embedder_gen(image_latents)
        image_embeds_und = image_embeds_und + self.showo.position_embedding(self.showo.image_position_ids)
        image_embeds_und = self.showo.und_trans(image_embeds_und)['last_hidden_state']
        image_embeds = self.showo.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

        text_tokens_a = torch.tensor([self.ids['bos_id']] + self.sys_prompt_ids + self.role_a).to(device)[None, :]
        text_tokens_b = torch.tensor([self.ids['boi_id'], self.ids['eoi_id']] + input_ids + self.role_b).to(device)[
                        None, :]
        text_embeds_a = self.showo.showo.model.embed_tokens(text_tokens_a)
        text_embeds_b = self.showo.showo.model.embed_tokens(text_tokens_b)

        if self.yaml_conf.model.showo.add_time_embeds:
            time_embeds = self.showo.time_embed(torch.Tensor([[1.0]]).to(device), text_embeds_a.dtype)
            if hasattr(self.showo, 'time_embed_proj'):
                time_embeds = self.showo.time_embed_proj(time_embeds)
            input_embeds = torch.cat([
                text_embeds_a,
                text_embeds_b[:, :1],
                time_embeds,
                image_embeds,
                text_embeds_b[:, 1:]
            ], dim=1)
            modality_positions = torch.tensor([text_tokens_a.shape[1] + 2, self.num_image_tokens])[None, None, :].to(device)
        else:
            input_embeds = torch.cat([
                text_embeds_a,
                text_embeds_b[:, :1],
                image_embeds,
                text_embeds_b[:, 1:]
            ], dim=1)
            modality_positions = torch.tensor([text_tokens_a.shape[1] + 1, self.num_image_tokens])[None, None, :].to(device)

        attention_mask = omni_attn_mask_naive(
            B=input_embeds.size(0),
            LEN=input_embeds.size(1),
            modalities=modality_positions,
            device=device, inverted=True
        ).to(input_embeds.dtype)

        if self._flashu_und_enabled:
            from flashu.understanding import configure_understanding_sample
            configure_understanding_sample(
                self.showo,
                visual_start=int(modality_positions[0, 0, 0].item()),
                visual_length=int(modality_positions[0, 0, 1].item()),
            )

        if profile_events is not None:
            profile_events["qwen_start"].record()
        cont_toks_list = self.showo.mmu_generate(input_embeds=input_embeds,
                                                 attention_mask=attention_mask,
                                                 top_k=top_k,
                                                 max_new_tokens=max_new_tokens,
                                                 eos_token=self.tokenizer.eos_token_id)
        if profile_events is not None:
            profile_events["qwen_end"].record()

        if self._flashu_wanda_calibrate:
            self._flashu_wanda_samples += 1
            if self._flashu_wanda_samples == self._flashu_wanda_limit:
                all_scores = self._flashu_wanda_pruner.get_structural_importance_scores()
                ffn_scores = {
                    name: score.detach().cpu()
                    for name, score in all_scores.items()
                    if name.endswith("mlp.down_proj")
                }
                score_path = os.environ["FLASHU_UNDERSTANDING_SCORES"]
                os.makedirs(os.path.dirname(score_path), exist_ok=True)
                torch.save(ffn_scores, score_path)
                self._flashu_wanda_pruner.remove_hooks()
                print(
                    f"[FlashU-UND] saved {len(ffn_scores)} FFN Wanda scores "
                    f"from {self._flashu_wanda_samples} samples to {score_path}"
                )

        if self._flashu_timing_file:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - flashu_start
            record = {
                "seconds": elapsed,
                "generated_tokens": len(cont_toks_list),
                "sample_identity": self._attribution_sample_identity,
                "logits_mode": os.environ["FLASHU_ATTRIBUTION_LOGITS_MODE"],
                "kv_enabled": os.environ.get("FLASHU_UNDERSTANDING_KV_CACHE", "0") == "1",
                "dls_ratio": float(os.environ["FLASHU_UNDERSTANDING_RLS"]),
                "accelerated": self._flashu_und_enabled,
                # The complete visual path is outside mmu_generate, so this
                # must remain exactly one call per evaluated sample.
                "vision_path_calls": 1,
            }
            if self._flashu_und_enabled:
                qwen = self.showo.showo
                record["token_lengths"] = qwen._flashu_und_token_lengths
                record["skip_history"] = qwen._flashu_und_skip_history
                record["cache_history"] = getattr(
                    qwen, "_flashu_und_cache_history", []
                )
                record["decode_step_profile"] = getattr(
                    qwen, "_flashu_und_last_step_profile", []
                )
            if profile_events is not None:
                record["profile_ms"] = {
                    "vae": profile_events["start"].elapsed_time(profile_events["after_vae"]),
                    "post_vae_vision_and_setup": profile_events["after_vae"].elapsed_time(profile_events["qwen_start"]),
                    "vision_and_setup_total": profile_events["start"].elapsed_time(profile_events["qwen_start"]),
                    "qwen_generate": profile_events["qwen_start"].elapsed_time(profile_events["qwen_end"]),
                    "gpu_total": profile_events["start"].elapsed_time(profile_events["qwen_end"]),
                }
                qwen_config = self.showo.showo.config
                record["attention"] = {
                    "config_implementation": getattr(qwen_config, "_attn_implementation", None),
                    "module_class": type(self.showo.showo.model.layers[0].self_attn).__name__,
                    "mask_rank": int(attention_mask.ndim),
                    "mask_shape": list(attention_mask.shape),
                }
            os.makedirs(os.path.dirname(self._flashu_timing_file), exist_ok=True)
            with open(self._flashu_timing_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

        return cont_toks_list


@register_model("showo2_qwen2_5")
class Showo2_Qwen2_5(lmms):
    """
    Evaluating Show-2, a unified understanding & generation model, on multimodal understanding benchmarks.
    Reference: https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/model_guide.md
    """
    def __init__(
        self,
        config_file: str = "",
        ckpt_path: str = "",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache=True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Setup accelerator and device; let's first not use accelerate for simplicity
        self._device = device
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        # Load config
        self.yaml_conf = OmegaConf.load(config_file)

        # Load tokenizer
        self._tokenizer, showo_token_ids = get_text_tokenizer(self.yaml_conf.model.showo.llm_model_path,
                                                             add_showo_tokens=True,
                                                             return_showo_token_ids=True,
                                                             llm_name="qwen2_5")

        # Load model
        self._model = UnifiedShowoModel(config_file, self._tokenizer, ckpt_path, showo_token_ids)

        print("Using config: ", config_file)
        print("Using checkpoint: ", ckpt_path)
        
        # Define other configs. The data processing config may also need to set up here.
        self.image_size = self.yaml_conf.dataset.preprocessing.resolution
        # self.image_size = 384
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._rank = 0
        self._word_size = 1

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

    # we directly call the exact model by their name
    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eod_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "We have not implemented this function for Show-2 yet"

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        # ====================== common data processing ======================
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        top_k = 1 

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            # ====================== define our customized data processing =======================
            # Get visual tokens
            v_tsr_list = []
            for vis in visuals:
                v_tsr = image_transform(vis, resolution=self.image_size).to(self.device, non_blocking=True)
                v_tsr_list.append(v_tsr)
            v_tsr = torch.stack(v_tsr_list, dim=0)

            v_tsr = v_tsr[:1]

            # Get text tokens (the text questions is in the 'contexts' variable from the request chunk)
            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")
                    contexts[i] = contexts[i].strip()
                    contexts[i] = "\n" + contexts[i]

            input_ids = [self.tokenizer(prompt, add_special_tokens=False).input_ids
                         for prompt in contexts]
            # input_ids = torch.nn.utils.rnn.pad_sequence(
            #     input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            # )
            # input_ids = input_ids.to(self.device)
            input_ids = input_ids[0]
            gen_kwargs = all_gen_kwargs[0]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            # preconfigure gen_kwargs with defaults
            if "image_sizes" not in gen_kwargs:
                try:
                    gen_kwargs["image_sizes"] = [visuals[0].size]
                except:
                    gen_kwargs["image_sizes"] = None
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # The adapter sorts requests by prompt length. Store identities
            # before the timed forward so timing rows can be paired by ID,
            # never incorrectly paired to evaluator output by line number.
            original_doc = self.task_dict[task][split][doc_id[0]]
            self.model._attribution_sample_identity = {
                "doc_id": int(doc_id[0]),
                "original_index": int(original_doc["index"]),
                "task": task,
                "prompt_sha256": hashlib.sha256(contexts[0].encode()).hexdigest(),
            }
            cont_toks_list = self.model(v_tsr,
                                        input_ids,
                                        gen_kwargs["max_new_tokens"],
                                        top_k,
                                        self._device
                                        )

            cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]
            result = self.tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)

            result = result[0].strip()

            # print(result)
            res.append(result)
            self.cache_hook.add_partial("generate_until", (contexts[0], gen_kwargs), result)
            pbar.update(1)

        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
