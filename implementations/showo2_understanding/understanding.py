"""FlashU multimodal-understanding acceleration reconstruction.

The public FlashU repository currently contains the generation path but not the
understanding implementation described in Algorithm 1 of the paper.  This
module reconstructs that path without changing the generation code:

* task-specific permanent FFN pruning (Wanda scores, ``r_p``),
* final-token dynamic layer skipping (``r_LS`` / ``T_LS``), and
* V-Norm visual-token pruning before the configured shallow layer (``rho``).

The implementation is intentionally isolated and opt-in so that the official
Show-o2 baseline remains an untouched control.
"""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnderstandingConfig:
    """Paper Table 8 defaults for the multimodal-understanding path."""

    ffn_prune_ratio: float = 0.10
    layer_skip_ratio: float = 0.10
    layer_skip_interval: int = 10
    token_prune_layer: int = 2  # one-based, matching Algorithm 1
    visual_token_prune_ratio: float = 0.50
    # ``low`` follows Sec. 3.5.1 (prune the highest V norms). ``high``
    # follows Algorithm 1 line 34 (TopK is assigned to I_keep).
    visual_token_keep_mode: str = "low"
    friendly_multiple: int = 64


def _score_for_layer(
    scores_by_layer: Dict[str, torch.Tensor], layer_idx: int
) -> torch.Tensor:
    suffix = f"model.layers.{layer_idx}.mlp.down_proj"
    matches = [value for name, value in scores_by_layer.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one Wanda score ending in {suffix!r}, found {len(matches)}"
        )
    return matches[0]


@torch.no_grad()
def apply_permanent_ffn_pruning(
    qwen_causal_lm: nn.Module,
    scores_by_layer: Dict[str, torch.Tensor],
    ratio: float = 0.10,
    friendly_multiple: int = 64,
) -> Dict[str, object]:
    """Physically shrink every FFN for the understanding-only model copy.

    The paper keeps full and pruned FFNs for task/time switching.  Our
    understanding evaluation runs in an independent process, so replacing the
    three projections directly is equivalent to always selecting ``FFN_p`` and
    avoids retaining a second set of weights.
    """

    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"FFN prune ratio must be in [0, 1), got {ratio}")

    per_layer: List[Dict[str, int]] = []
    total_pruned = 0

    for layer_idx, layer in enumerate(qwen_causal_lm.model.layers):
        mlp = layer.mlp
        old_size = int(mlp.intermediate_size)
        ideal_keep = old_size - int(old_size * ratio)
        keep_size = (ideal_keep // friendly_multiple) * friendly_multiple
        keep_size = max(min(keep_size, old_size), min(friendly_multiple, old_size))

        importance = _score_for_layer(scores_by_layer, layer_idx).float()
        if importance.numel() != old_size:
            raise ValueError(
                f"Layer {layer_idx}: Wanda score has {importance.numel()} entries; "
                f"expected {old_size}"
            )

        keep = torch.topk(importance, keep_size, largest=True).indices.sort().values
        device = mlp.gate_proj.weight.device
        keep = keep.to(device)
        dtype = mlp.gate_proj.weight.dtype
        hidden_size = mlp.gate_proj.in_features

        gate = nn.Linear(hidden_size, keep_size, bias=False, device=device, dtype=dtype)
        up = nn.Linear(hidden_size, keep_size, bias=False, device=device, dtype=dtype)
        down = nn.Linear(keep_size, hidden_size, bias=False, device=device, dtype=dtype)
        gate.weight.copy_(mlp.gate_proj.weight.index_select(0, keep))
        up.weight.copy_(mlp.up_proj.weight.index_select(0, keep))
        down.weight.copy_(mlp.down_proj.weight.index_select(1, keep))

        mlp.gate_proj = gate
        mlp.up_proj = up
        mlp.down_proj = down
        mlp.intermediate_size = keep_size

        pruned = old_size - keep_size
        total_pruned += pruned
        per_layer.append({"layer": layer_idx, "original": old_size, "kept": keep_size})

    report: Dict[str, object] = {
        "ratio_requested": ratio,
        "friendly_multiple": friendly_multiple,
        "layers": len(per_layer),
        "total_neurons_pruned": total_pruned,
        "per_layer": per_layer,
    }
    logger.info(
        "[FlashU-UND] Permanently pruned %d FFN neurons across %d layers",
        total_pruned,
        len(per_layer),
    )
    return report


def _gather_sequence(tensor: Optional[torch.Tensor], indices: torch.Tensor):
    if tensor is None:
        return None
    if tensor.ndim == 2:
        # Either [B, L] position ids or [L, L] attention mask.
        if tensor.shape[0] == tensor.shape[1]:
            return tensor.index_select(0, indices).index_select(1, indices)
        return tensor.index_select(-1, indices)
    if tensor.ndim == 3:
        return tensor.index_select(1, indices)
    if tensor.ndim == 4:
        # [B, H/1, Q, K]
        return tensor.index_select(-2, indices).index_select(-1, indices)
    raise ValueError(f"Unsupported sequence tensor rank: {tensor.ndim}")


def _visual_keep_indices(
    hidden_states: torch.Tensor,
    decoder_layer: nn.Module,
    visual_start: int,
    visual_length: int,
    prune_ratio: float,
    keep_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return global gather indices and raw per-visual-token V norms."""

    seq_len = hidden_states.shape[1]
    visual_end = visual_start + visual_length
    if visual_start < 0 or visual_end > seq_len:
        raise ValueError(
            f"Visual span [{visual_start}, {visual_end}) is outside sequence length {seq_len}"
        )

    # Algorithm 1 computes V at L_prune and drops the largest V norms.
    normed = decoder_layer.input_layernorm(hidden_states)
    values = decoder_layer.self_attn.v_proj(normed)
    visual_values = values[:, visual_start:visual_end, :]
    scores = visual_values.float().norm(p=2, dim=-1).mean(dim=0)

    keep_count = max(1, int((1.0 - prune_ratio) * visual_length))
    if keep_mode not in {"low", "high"}:
        raise ValueError(f"V-Norm keep mode must be 'low' or 'high', got {keep_mode!r}")
    visual_keep = torch.topk(
        scores,
        keep_count,
        largest=(keep_mode == "high"),
    ).indices.sort().values
    visual_keep = visual_keep + visual_start

    before = torch.arange(0, visual_start, device=hidden_states.device)
    after = torch.arange(visual_end, seq_len, device=hidden_states.device)
    gather = torch.cat((before, visual_keep, after))
    return gather, scores


def _select_skip_layers(
    layer_pairs: Sequence[Tuple[int, torch.Tensor, torch.Tensor]],
    num_layers: int,
    ratio: float,
    excluded: Iterable[int] = (),
) -> Tuple[List[int], List[float]]:
    excluded = set(excluded)
    candidates: List[Tuple[int, float]] = []
    similarities = [float("nan")] * num_layers
    for layer_idx, before, after in layer_pairs:
        similarity = F.cosine_similarity(before.float(), after.float(), dim=-1).mean().item()
        similarities[layer_idx] = similarity
        if layer_idx not in excluded:
            candidates.append((layer_idx, similarity))

    count = int(num_layers * ratio)
    count = min(count, len(candidates))
    chosen = sorted(idx for idx, _ in sorted(candidates, key=lambda x: x[1], reverse=True)[:count])
    return chosen, similarities


def configure_understanding_sample(
    showo2_model: nn.Module,
    visual_start: int,
    visual_length: int,
) -> None:
    """Reset per-sample dynamic state and register the visual token span."""

    qwen = showo2_model.showo
    qwen._flashu_und_visual_start = int(visual_start)
    qwen._flashu_und_visual_length = int(visual_length)
    qwen._flashu_und_decode_step = 0
    qwen._flashu_und_layers_to_skip = []
    qwen._flashu_und_last_similarities = None
    qwen._flashu_und_token_lengths = []
    qwen._flashu_und_skip_history = []
    qwen._flashu_und_cache_history = []


def patch_understanding_forward(
    showo2_model: nn.Module,
    config: UnderstandingConfig = UnderstandingConfig(),
) -> None:
    """Patch the Qwen CausalLM forward with Algorithm 1's UND pathway."""

    qwen = showo2_model.showo
    if hasattr(qwen, "_flashu_und_original_forward"):
        return

    qwen._flashu_und_original_forward = qwen.forward
    qwen._flashu_und_config = config
    qwen._flashu_und_decode_step = 0
    qwen._flashu_und_layers_to_skip = []
    qwen._flashu_und_token_lengths = []
    qwen._flashu_und_skip_history = []
    qwen._flashu_und_cache_history = []

    def accelerated_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if output_attentions:
            raise NotImplementedError("Attention outputs are not needed by FlashU UND")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        hidden_states = inputs_embeds
        original_seq_len = hidden_states.shape[1]
        cache_enabled = bool(use_cache or past_key_values is not None)
        # A cache rebuild processes the full current sequence.  Incremental
        # decoding receives one token and a row of the original omni mask.
        cache_is_empty = True
        if cache_enabled and past_key_values is not None:
            # DLS may skip layer 0, leaving its slot empty while later active
            # layers contain valid states.  Inspect all layers instead of the
            # global layer-0 counter.
            cache_is_empty = not any(
                past_key_values.get_seq_length(layer_idx) > 0
                for layer_idx in range(len(self.model.layers))
            )
        cache_rebuild = cache_enabled and (cache_is_empty or original_seq_len > 1)
        full_mask_key_length = int(attention_mask.shape[-1]) if attention_mask is not None else original_seq_len
        if cache_position is None:
            if cache_enabled and not cache_rebuild:
                # DynamicCache's global counter is not reliable when layer 0
                # is selected by DLS, so derive the absolute position from the
                # unpruned omni-attention row supplied by mmu_generate.
                cache_position = torch.tensor(
                    [full_mask_key_length - 1], device=hidden_states.device
                )
            else:
                cache_position = torch.arange(original_seq_len, device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        step = int(getattr(self, "_flashu_und_decode_step", 0))
        recalc = step % config.layer_skip_interval == 0
        active_skip = set() if recalc else set(getattr(self, "_flashu_und_layers_to_skip", []))
        prune_idx = config.token_prune_layer - 1
        visual_start = int(getattr(self, "_flashu_und_visual_start"))
        visual_length = int(getattr(self, "_flashu_und_visual_length"))
        layer_pairs: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        all_hidden_states = () if output_hidden_states else None

        for layer_idx, decoder_layer in enumerate(self.model.layers):
            if layer_idx in active_skip:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                continue

            if layer_idx == prune_idx and (not cache_enabled or cache_rebuild):
                gather, scores = _visual_keep_indices(
                    hidden_states,
                    decoder_layer,
                    visual_start,
                    visual_length,
                    config.visual_token_prune_ratio,
                    config.visual_token_keep_mode,
                )
                hidden_states = hidden_states.index_select(1, gather)
                attention_mask = _gather_sequence(attention_mask, gather)
                position_ids = _gather_sequence(position_ids, gather)
                cache_position = cache_position.index_select(0, gather)
                position_embeddings = tuple(x.index_select(1, gather) for x in position_embeddings)
                self._flashu_und_last_vnorm = scores.detach().cpu()
                if cache_enabled:
                    # Indices refer to the full sequence used for this rebuild.
                    # Deeper layers cache only these positions; later generated
                    # tokens are appended after this fixed base.
                    self._flashu_und_cache_gather = gather.detach().clone()
                    self._flashu_und_cache_full_length = original_seq_len

            elif layer_idx == prune_idx and cache_enabled:
                base_gather = getattr(self, "_flashu_und_cache_gather").to(
                    attention_mask.device
                )
                base_full_length = int(
                    getattr(self, "_flashu_und_cache_full_length")
                )
                extra = torch.arange(
                    base_full_length,
                    full_mask_key_length,
                    device=attention_mask.device,
                )
                deep_key_indices = torch.cat((base_gather, extra))
                # Incremental decoding has one query row, so only the key axis
                # is gathered.  Its absolute RoPE position remains unpruned.
                attention_mask = attention_mask.index_select(-1, deep_key_indices)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            before = hidden_states[:, -1, :].detach() if recalc else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=cache_enabled,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if recalc:
                layer_pairs.append((layer_idx, before, hidden_states[:, -1, :].detach()))

        hidden_states = self.model.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if recalc:
            chosen, similarities = _select_skip_layers(
                layer_pairs,
                len(self.model.layers),
                config.layer_skip_ratio,
                excluded=(prune_idx,),
            )
            self._flashu_und_layers_to_skip = chosen
            self._flashu_und_last_similarities = similarities
        self._flashu_und_skip_history.append((step, recalc, list(self._flashu_und_layers_to_skip)))
        self._flashu_und_token_lengths.append((original_seq_len, hidden_states.shape[1]))
        if cache_enabled:
            cache_lengths = [
                int(past_key_values.get_seq_length(layer_idx))
                for layer_idx in range(len(self.model.layers))
            ]
            self._flashu_und_cache_history.append(
                (step, bool(cache_rebuild), cache_lengths)
            )
        self._flashu_und_decode_step = step + 1

        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            values = (logits, None, all_hidden_states, None)
            output = tuple(x for x in values if x is not None)
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if cache_enabled else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    qwen.forward = types.MethodType(accelerated_forward, qwen)
    logger.info("[FlashU-UND] Patched Qwen forward with %s", config)


def unpatch_understanding_forward(showo2_model: nn.Module) -> None:
    qwen = showo2_model.showo
    if hasattr(qwen, "_flashu_und_original_forward"):
        qwen.forward = qwen._flashu_und_original_forward
        del qwen._flashu_und_original_forward
