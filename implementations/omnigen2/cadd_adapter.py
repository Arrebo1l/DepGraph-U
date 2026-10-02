"""Small, guarded refiner-dispatch adaptation of the pinned OmniGen2 source.

B is the upstream method; C only applies last-layer Taylor history/prediction.
D adds CADD to C. E/F apply identical static-text hoisting to C/D respectively.
The upstream source remains unchanged on disk. Generated code and diff are saved.
"""
import difflib
import hashlib
import inspect
import textwrap
from pathlib import Path
import torch


def install(evidence_dir):
    import omnigen2.models.transformers.transformer_omnigen2 as mod
    cls = mod.OmniGen2Transformer2DModel
    original = textwrap.dedent(inspect.getsource(cls.forward))
    updated = original
    old = "    if enable_taylorseer:\n        cal_type(self.cache_dic, self.current)"
    new = """    if enable_taylorseer:
        self.cache_dic['_tg_last_only'] = getattr(self, '_tg_mode', 'B') in ('C', 'D', 'E', 'F')
        self.cache_dic['_tg_last_layer'] = len(self.layers) - 1
        cal_type(self.cache_dic, self.current)"""
    assert updated.count(old) == 1
    updated = updated.replace(old, new)

    start = updated.index('    # 2. Context refinement')
    stop = updated.index('    if self.enable_teacache:', start)
    refiner_block = updated[start:stop]
    static_old = """    for layer in self.context_refiner:
        text_hidden_states = layer(text_hidden_states, text_attention_mask, context_rotary_emb)"""
    static_new = """    tg_static = getattr(self, '_tg_mode', 'B') in ('E', 'F')
    tg_branch = id(self.cache_dic) if enable_taylorseer else None
    if tg_static and tg_branch in self._tg_context_cache:
        text_hidden_states = self._tg_context_cache[tg_branch]
    else:
        for layer in self.context_refiner:
            text_hidden_states = layer(text_hidden_states, text_attention_mask, context_rotary_emb)
        if tg_static:
            self._tg_context_cache[tg_branch] = text_hidden_states"""
    assert refiner_block.count(static_old) == 1
    refiner_block = refiner_block.replace(static_old, static_new)
    new_block = """    # CADD: keep current timestep embedding and output layout. Input packing
    # and RoPE are deliberately retained in this conservative first adapter.
    tg_skip = False
    if enable_taylorseer and getattr(self, '_tg_mode', 'B') in ('D', 'F'):
        factors = self.cache_dic['cache'][-1]['layers_stream'][len(self.layers)-1].get('total', {})
        cached = factors.get(0)
        tg_skip = (self.current['type'] == 'Taylor' and
                   not self.enable_teacache and not self.training and
                   not torch.is_grad_enabled() and cached is not None and
                   cached.shape == (batch_size, max(seq_lengths), self.config.hidden_size) and
                   cached.dtype == temb.dtype and cached.device == temb.device)
    self._tg_last_skipped = tg_skip
    if tg_skip:
        hidden_states = None
        attention_mask = None
    else:
""" + textwrap.indent(refiner_block, '    ')
    updated = updated[:start] + new_block + updated[stop:]

    # Match only the non-TeaCache loop. Full steps still run every layer.
    old = """        for layer_idx, layer in enumerate(self.layers):
            if enable_taylorseer:
                layer.current = self.current"""
    new = """        tg_iterator = enumerate(self.layers)
        if enable_taylorseer and self.cache_dic.get('_tg_last_only') and self.current['type'] == 'Taylor':
            tg_iterator = [(len(self.layers)-1, self.layers[-1])]
        for layer_idx, layer in tg_iterator:
            if enable_taylorseer:
                layer.current = self.current"""
    assert updated.count(old) == 1
    updated = updated.replace(old, new)
    ns = dict(mod.__dict__)
    exec(compile(updated, '<TaskGraph guarded OmniGen2 forward>', 'exec'), ns)
    cls.forward = ns['forward']
    original_derivative = mod.derivative_approximation

    def derivative(cache_dic, current, feature):
        if cache_dic.get('_tg_last_only') and current['layer'] != cache_dic['_tg_last_layer']:
            return
        return original_derivative(cache_dic, current, feature)

    # Block.forward resolves this global in its original module.
    mod.derivative_approximation = derivative
    out = Path(evidence_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/'upstream_forward.py').write_text(original)
    (out/'candidate_forward.py').write_text(updated)
    (out/'forward.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),fromfile='upstream_forward.py',tofile='candidate_forward.py')))
    return hashlib.sha256(updated.encode()).hexdigest()


def reset(pipe, mode):
    assert mode in ('A', 'B', 'C', 'D', 'E', 'F')
    pipe.enable_taylorseer = mode != 'A'
    model = pipe.transformer
    model.enable_taylorseer = mode != 'A'
    model.enable_teacache = False
    model._tg_mode = mode
    model._tg_context_cache = {}
    model._tg_last_skipped = False
    # Upstream leaves attributes on modules after a call; explicitly reset them
    # before changing arms to prevent a cache-on -> cache-off contamination.
    for layer in model.layers:
        layer.enable_taylorseer = mode != 'A'
        for name in ('current', 'cache_dic'):
            if hasattr(layer, name):
                delattr(layer, name)
    for name in ('current', 'cache_dic'):
        if hasattr(model, name):
            delattr(model, name)
    model.eval()
    torch.cuda.empty_cache()
