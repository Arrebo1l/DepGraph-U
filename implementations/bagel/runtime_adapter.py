"""BAGEL-specific dependency dispatch. Policy pruning is attributed separately.

No generation vocabulary benefit is credited: BAGEL already omits that head.
Understanding preserves native KV and uses a guarded append-only buffer.
"""
import hashlib, json, types
import torch
import torch.nn.functional as F

def sha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

def append_kv(cache, layer, old_k, old_v, new_k, new_v):
    """Only called after the model-level contiguous layout and fixed-graph guard."""
    buffers = getattr(cache, '_dgu_buffers', None)
    if buffers is None:
        buffers = {}; cache._dgu_buffers = buffers
    n, q = old_k.shape[0], new_k.shape[0]
    current = buffers.get(layer)
    if current is None or current[0].shape[0] < n + q or current[2] != n or current[3] != old_k.data_ptr():
        capacity = ((n + q + 127) // 128) * 128
        k = old_k.new_empty((capacity, *old_k.shape[1:]))
        v = old_v.new_empty((capacity, *old_v.shape[1:]))
        k[:n].copy_(old_k); v[:n].copy_(old_v)
    else:
        k, v = current[:2]
    k[n:n+q].copy_(new_k); v[n:n+q].copy_(new_v)
    out_k, out_v = k[:n+q], v[:n+q]
    buffers[layer] = (k, v, n+q, out_k.data_ptr())
    cache._dgu_append_calls = getattr(cache, '_dgu_append_calls', 0) + 1
    return out_k, out_v

class PolicyAdapter:
    def __init__(self, model, task):
        self.model = model; self.task = task; self.enabled = False
        self.full = False; self.calibrating = False; self.stats = {}; self.pruned = {}
        self.calls = 0; self.targets = []
        for i, layer in enumerate(model.language_model.model.layers):
            mlp = layer.mlp_moe_gen if task == 'generation' else layer.mlp
            self.targets.append((i, mlp))
            original = mlp.forward
            def forward(this, x, original=original, i=i):
                if self.enabled and not self.full:
                    self.calls += 1
                    gate, up, down = self.pruned[i]
                    return F.linear(this.act_fn(F.linear(x, gate)) * F.linear(x, up), down)
                return original(x)
            mlp.forward = types.MethodType(forward, mlp)
        self.hooks = []

    def start_calibration(self):
        self.enabled = False
        for i, mlp in self.targets:
            def collect(module, inputs, output, i=i):
                x = inputs[0].detach().float()
                values = x.square().sum(dim=tuple(range(x.ndim - 1)))
                self.stats[i] = self.stats.get(i, 0) + values
            self.hooks.append(mlp.down_proj.register_forward_hook(collect))

    def finish_calibration(self):
        for hook in self.hooks: hook.remove()
        self.hooks = []
        summary = {}
        for i, mlp in self.targets:
            assert i in self.stats and torch.isfinite(self.stats[i]).all(), i
            # FlashU/Wanda structured down-projection column importance.
            importance = mlp.down_proj.weight.float().abs().sum(0) * self.stats[i].sqrt()
            keep_n = int(importance.numel() * 0.8) // 128 * 128
            keep = importance.topk(keep_n).indices.sort().values
            self.pruned[i] = (mlp.gate_proj.weight.detach()[keep].contiguous(),
                              mlp.up_proj.weight.detach()[keep].contiguous(),
                              mlp.down_proj.weight.detach()[:, keep].contiguous())
            summary[str(i)] = dict(original=importance.numel(), retained=keep_n, keep_sha256=sha(keep))
        self.stats = {}
        return summary

    def configure(self, arm):
        self.enabled = arm != 'native'; self.full = False; self.calls = 0
        lm = self.model.language_model.model
        lm._tg_cadd = arm == 'ours' and self.task == 'generation'
        lm._tg_counts = {}; lm._dgu_kv = arm == 'ours' and self.task == 'understanding'
        lm.enable_taylorseer = False
        for layer in lm.layers:
            layer.enable_taylorseer = False
            layer._tg_last_layer_only = True
            for attr in ['cache_dic', 'current']:
                if hasattr(layer, attr): delattr(layer, attr)
        for attr in ['cache_dic', 'current']:
            if hasattr(lm, attr): delattr(lm, attr)

def setup_model(root, task):
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer
    device = json.loads((root/'DEVICE.json').read_text())
    from pathlib import Path
    weights = Path(device['weights'])
    llm = Qwen2Config.from_json_file(str(weights/'llm_config.json'))
    llm.qk_norm=True; llm.tie_word_embeddings=False; llm.layer_module='Qwen2MoTDecoderLayer'
    vit=SiglipVisionConfig.from_json_file(str(weights/'vit_config.json')); vit.rope=False; vit.num_hidden_layers-=1
    vae, vaec=load_ae(str(weights/'ae.safetensors')); vae=vae.cuda().eval()
    config=BagelConfig(visual_gen=True,visual_und=True,llm_config=llm,vit_config=vit,vae_config=vaec,vit_max_num_patch_per_side=70,connector_act='gelu_pytorch_tanh',latent_patch_size=2,max_latent_size=64)
    with init_empty_weights():
        model=Bagel(Qwen2ForCausalLM(llm),SiglipVisionModel(vit),config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit,meta=True)
    model=load_checkpoint_and_dispatch(model,checkpoint=str(weights/'ema.safetensors'),device_map={'':'cuda:0'},dtype=torch.bfloat16,force_hooks=True).eval()
    tok=Qwen2Tokenizer.from_pretrained(str(weights)); tok,ids,_=add_special_tokens(tok)
    transform=ImageTransform(980,378,14,max_pixels=2007040)
    inf=InterleaveInferencer(model,vae,tok,ImageTransform(1024,512,16),transform,ids)
    # Match the validated one-device BAGEL setup in all arms.
    for name in ['prepare_vae_latent','prepare_vae_latent_cfg']:
        original=getattr(model,name)
        def prep(*args, _original=original, **kwargs):
            return {k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in _original(*args,**kwargs).items()}
        setattr(model,name,prep)
    return model,inf,tok,ids,transform,PolicyAdapter(model,task)
