"""Fresh paired C/D example. This launcher has not been GPU-validated at release time.

Use pinned local OmniGen2 source and weights. No downloads or remote jobs are
started by this script. Each invocation requires a new output directory.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from artifact_utils import verify_hf_files, ratio_of_means


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prompt',default='A red apple on a white plate.')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--repeats',type=int,default=3)
    return p.parse_args()


def main():
    a=arguments()
    if a.repeats<1 or not a.prompt.strip():raise ValueError('Positive repeats and a nonempty prompt required')
    source,weights,out=a.source.resolve(),a.weights.resolve(),a.output.resolve()
    if out.exists():raise ValueError('Output already exists; choose a fresh directory')
    protocol=json.loads((ROOT/'configs/protocols/omnigen2_geneval.json').read_text())
    rev=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True)
    if rev!=protocol['source_revision'] or dirty:raise ValueError('Use a clean checkout of the pinned OmniGen2 commit')
    print('Verifying local weight and scheduler identity...',flush=True)
    verify_hf_files(weights,json.loads((ROOT/'configs/omnigen2_weights.json').read_text()))
    sys.path.insert(0,str(source));sys.path.insert(0,str(ROOT/'implementations/omnigen2'))
    import numpy as np
    import torch
    import cadd_adapter
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    from diffusers import AutoencoderKL
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(8)
    out.mkdir(parents=True)
    def save(name,obj):(out/name).write_text(json.dumps(obj,indent=2)+'\n')
    save('run_config.json',{'source_revision':rev,'weight_revision':protocol['weight_revision'],
        'prompt':a.prompt,'seed':a.seed,'repeats':a.repeats,'python':platform.python_version(),
        'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'scope':'Fresh example; not a reproduction of a full benchmark table'})
    try:
        transformer=OmniGen2Transformer2DModel.from_pretrained(str(weights),subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True)
        # The verified weight-repository scheduler has different defaults from
        # the source checkout. Load that exact file, not a library replacement.
        spec=importlib.util.spec_from_file_location('depgraph_pinned_scheduler',weights/'scheduler/scheduling_flow_match_euler_discrete.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        scheduler=module.FlowMatchEulerDiscreteScheduler.from_pretrained(str(weights),subfolder='scheduler',local_files_only=True)
        mllm=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(weights/'mllm'),torch_dtype=torch.bfloat16,local_files_only=True)
        processor=Qwen2_5_VLProcessor.from_pretrained(str(weights/'processor'),local_files_only=True)
        vae=AutoencoderKL.from_pretrained(str(weights),subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True)
        pipe=OmniGen2Pipeline(mllm=mllm,processor=processor,scheduler=scheduler,transformer=transformer,vae=vae).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        for m in [pipe.transformer,pipe.mllm,pipe.vae]:m.eval()
        cadd_adapter.install(out/'adapter_diff')

        def tensor_hash(t):return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()
        def run(mode,seed,label,diagnostic=False):
            cadd_adapter.reset(pipe,mode)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            generator=torch.Generator(device='cuda').manual_seed(seed)
            skips=[]
            handle=pipe.transformer.register_forward_hook(lambda m,args,output:skips.append(bool(m._tg_last_skipped))) if diagnostic else None
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            try:
                with torch.no_grad():
                    image=pipe(prompt=a.prompt,negative_prompt=protocol['negative_prompt'],input_images=None,
                        width=protocol['width'],height=protocol['height'],num_inference_steps=protocol['steps'],
                        max_sequence_length=protocol['max_sequence_length'],text_guidance_scale=protocol['text_cfg'],
                        image_guidance_scale=protocol['image_cfg'],cfg_range=protocol['cfg_range'],
                        num_images_per_prompt=1,generator=generator,output_type='pil').images[0]
                torch.cuda.synchronize();seconds=time.perf_counter()-start
            finally:
                if handle is not None:handle.remove()
            record={'mode':mode,'seed':seed,'seconds':None if diagnostic else seconds,
                'diagnostic':diagnostic,'cadd_hits':sum(skips) if diagnostic else None,
                'pixel_sha256':hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
                'image_size':list(image.size),'image_mode':image.mode,
                'rng':{'generator':tensor_hash(generator.get_state()),'torch_cpu':tensor_hash(torch.get_rng_state()),
                       'torch_cuda':tensor_hash(torch.cuda.get_rng_state())},
                'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            image.save(out/f'{label}.png');save(f'{label}.json',record)
            return record

        def matches(left,right):
            return all(left[k]==right[k] for k in ['pixel_sha256','image_size','image_mode','rng'])

        warm={mode:run(mode,733,f'warmup_{mode}',True) for mode in ['C','D']}
        if not matches(warm['C'],warm['D']) or warm['D']['cadd_hits']==0:
            raise RuntimeError('Preflight output/RNG equality or CADD-hit check failed; records preserved')
        pairs=[]
        for repeat in range(a.repeats):
            order=['C','D'] if repeat%2==0 else ['D','C']
            row={mode:run(mode,a.seed,f'r{repeat}_{mode}') for mode in order}
            row['equal']=matches(row['C'],row['D']);row['order']=order
            save(f'pair_{repeat}.json',row);pairs.append(row)
            if not row['equal']:raise RuntimeError('Paired output/RNG mismatch; both records preserved')
        baseline=[r['C']['seconds'] for r in pairs];candidate=[r['D']['seconds'] for r in pairs]
        result={'pairs':len(pairs),'reference':'C: last-layer Taylor control','candidate':'D: C + CADD',
            'reference_mean_s':sum(baseline)/len(baseline),'candidate_mean_s':sum(candidate)/len(candidate),
            'speedup':ratio_of_means(baseline,candidate),'all_tested_pairs_equal':True,
            'slower_candidate_pairs':sum(d>c for c,d in zip(baseline,candidate)),
            'scope':'One prompt, repeated timings, fixed seed. No benchmark quality score or universal equality claim.'}
        save('summary.json',result);print(json.dumps(result,indent=2))
    except Exception as exc:
        save('failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise


if __name__=='__main__':main()
