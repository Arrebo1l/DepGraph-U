import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from artifact_utils import verify_hashes, verify_hf_files, ratio_of_means, sha256

spec=importlib.util.spec_from_file_location('bagel_dispatch',ROOT/'implementations/bagel/generation_dispatch.py')
dispatch=importlib.util.module_from_spec(spec);spec.loader.exec_module(dispatch)


class ArtifactTests(unittest.TestCase):
    def test_detects_changed_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'record.json';p.write_text('{"n": 3}')
            expected={'record.json':sha256(p)}
            self.assertEqual(verify_hashes(temp,expected),1)
            p.write_text('{"n": 4}')
            with self.assertRaises(ValueError):verify_hashes(temp,expected)

    def test_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):verify_hashes(temp,{'../outside':'0'*64})

    def test_hf_blob_and_lfs_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            value=b'known bytes';p=Path(temp)/'small';p.write_bytes(value)
            item={'path':'small','size':len(value),'type':'file','oid':hashlib.sha1(b'blob 11\0'+value).hexdigest()}
            verify_hf_files(temp,{'files':[item]})
            item['lfs']={'oid':hashlib.sha256(value).hexdigest()}
            verify_hf_files(temp,{'files':[item]})
            p.write_bytes(b'wrong bytes')
            with self.assertRaises(ValueError):verify_hf_files(temp,{'files':[item]})

    def test_ratio_is_not_mean_of_ratios(self):
        self.assertEqual(ratio_of_means([1,9],[1,3]),2.5)
        for a,b in [([],[]),([1],[0]),([1],[float('nan')]),([1,2],[1])]:
            with self.assertRaises(ValueError):ratio_of_means(a,b)

    def test_corrected_mmstar_and_paired_quality(self):
        data=json.loads((ROOT/'results/summary.json').read_text())
        for arm,r in data['mmstar_rescoring']['results'].items():
            self.assertGreaterEqual(r['official_accuracy_percent'],0)
            self.assertLessEqual(r['official_accuracy_percent'],100)
        rows=[r for r in data['understanding'] if r['table']==3 and r['benchmark']=='MMStar']
        self.assertEqual(len(rows),2)
        self.assertAlmostEqual(float(rows[0]['score']),757/1500*100)
        self.assertEqual(rows[0]['score'],rows[1]['score'])


class BagelBoundaryTests(unittest.TestCase):
    def test_only_supported_generation_rewrites(self):
        options={'cfg_img_scale':1.0,'cfg_text_scale':4.0,'enable_taylorseer':False}
        self.assertTrue(dispatch.eligible(None,'a red apple',options))
        for image,text,extra in [(object(),'a red apple',{}),(None,'',{}),
                                (None,'apple',{'cfg_img_scale':2}),
                                (None,'apple',{'enable_taylorseer':True}),
                                (None,'apple',{'cfg_text_scale':float('nan')}),
                                (None,'apple',{'return_history':True})]:
            self.assertFalse(dispatch.eligible(image,text,{**options,**extra}))

    def test_fallback_preserves_arguments(self):
        seen=[]
        def original(**kwargs):seen.append(kwargs);return 'reference'
        self.assertEqual(dispatch.guarded_infer(original,text='apple',return_history=True),'reference')
        self.assertEqual(seen,[{'image':None,'text':'apple','return_history':True}])


if __name__=='__main__':unittest.main()
