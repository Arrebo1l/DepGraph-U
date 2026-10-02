"""BAGEL-native output-dependent dispatch, without any approximate cache provider.

Only stateless text-to-image requests with image guidance disabled are rewritten.
The unused image-CFG producer has no consumer and no externally returned future state.
All other requests delegate to the unchanged official high-level API.
"""
import math

def eligible(image, text, options):
    allowed={'image_shapes','num_timesteps','cfg_text_scale','cfg_img_scale','cfg_interval','cfg_renorm_type','timestep_shift','enable_taylorseer'}
    return (image is None and isinstance(text,str) and len(text)>0 and
            set(options)<=allowed and options.get('enable_taylorseer',False) is False and
            isinstance(options.get('cfg_img_scale'),(int,float)) and options['cfg_img_scale']==1.0 and
            isinstance(options.get('cfg_text_scale'),(int,float)) and math.isfinite(options['cfg_text_scale']) and options['cfg_text_scale']>1.0)

def guarded_infer(inf,image=None,text=None,**options):
    if not eligible(image,text,options):
        return inf(image=image,text=text,**options)
    import torch
    shapes=options.get('image_shapes',(1024,1024))
    with torch.autocast(device_type='cuda',enabled=True,dtype=torch.bfloat16):
        context=inf.init_gen_context()
        text_cfg=inf.init_gen_context()
        # Same live prompt cache; no duplicate prompt prefill for disabled image CFG.
        context=inf.update_context_text(text,context)
        inputs=inf.model.prepare_vae_latent(curr_kvlens=context['kv_lens'],curr_rope=context['ropes'],image_sizes=[shapes],new_token_ids=inf.new_token_ids)
        cfg=inf.model.prepare_vae_latent_cfg(curr_kvlens=text_cfg['kv_lens'],curr_rope=text_cfg['ropes'],image_sizes=[shapes])
        settings={k:v for k,v in options.items() if k not in ['image_shapes','enable_taylorseer']}
        latents=inf.model.generate_image(past_key_values=context['past_key_values'],cfg_text_past_key_values=text_cfg['past_key_values'],cfg_img_past_key_values=None,
            **inputs,cfg_text_packed_position_ids=cfg['cfg_packed_position_ids'],cfg_text_packed_query_indexes=cfg['cfg_packed_query_indexes'],cfg_text_key_values_lens=cfg['cfg_key_values_lens'],cfg_text_packed_key_value_indexes=cfg['cfg_packed_key_value_indexes'],
            enable_taylorseer=False,**settings)
        image_out=inf.decode_image(latents[0],shapes)
    return {'image':image_out,'text':None}
