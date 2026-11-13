import tempfile
import unittest
from pathlib import Path
import torch
import test_sampler
from longvideo.segment_cache import SegmentCache, fingerprint
from unittest.mock import patch
import sys
import types
from longvideo.segment_cache import cached_sample
from longvideo.segment_cache import workflow_identity
from longvideo.segment_cache import exclusive_task


class CacheTests(unittest.TestCase):
    def test_failed_task_keeps_prefix_and_unlocks(self):
        class Job:
            @exclusive_task
            def run(self, cache_directory=''):
                SegmentCache(cache_directory).save('prefix', {'video':torch.ones(1)})
                raise RuntimeError('simulated interruption')
        with tempfile.TemporaryDirectory() as directory:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    Job().run(cache_directory=directory)
            self.assertIsNotNone(SegmentCache(directory).load('prefix'))

    def test_same_directory_cannot_run_concurrently(self):
        class Job:
            @exclusive_task
            def run(self, cache_directory=''):
                return self.run(cache_directory=cache_directory)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, '正在运行'):
                Job().run(cache_directory=directory)
    def test_persistent_identity_changes_with_model_content(self):
        nodes = types.ModuleType('nodes')
        nodes.NODE_CLASS_MAPPINGS = {'Loader': CacheTests}
        paths = types.ModuleType('folder_paths')
        with tempfile.TemporaryDirectory() as directory:
            paths.folder_names_and_paths = {'models':([directory], set())}
            model_file = Path(directory) / 'model.bin'
            model_file.write_bytes(b'version-one')
            graph = {'1':{'class_type':'Loader', 'inputs':{'model':'model.bin'}},
                     '2':{'class_type':'Sampler', 'inputs':{'model':['1',0]}}}
            with patch.dict(sys.modules, {'nodes':nodes, 'folder_paths':paths}):
                first = workflow_identity(graph, '2', ['model'])
                self.assertEqual(first, workflow_identity(graph, '2', ['model']))
                model_file.write_bytes(b'version-two')
                self.assertNotEqual(first, workflow_identity(graph, '2', ['model']))

    def test_restart_reuses_with_verified_identity(self):
        nested = types.ModuleType('comfy.nested_tensor')
        nested.NestedTensor = test_sampler.Nested
        def sample(*args):
            return {'samples':test_sampler.Nested((torch.ones(1,24,7,2,2),torch.ones(1,32,2,37)))}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'comfy.nested_tensor':nested}):
            params = dict(model=object(), cond=[], negative=[], latent={}, sampler=object(),
                          sigmas=torch.tensor([1.,0.]), seed=4, cfg=1, stage='ordinary-v2', identity='verified-test-graph')
            cached_sample(directory, sample, **params)
            with patch('longvideo.segment_cache.SESSION', 'new-engine-session'):
                _, hit = cached_sample(directory, sample, **dict(params, model=object(), sampler=object()))
            self.assertTrue(hit)
            import json
            manifest = json.loads((Path(directory)/'manifest.json').read_text())
            self.assertEqual(manifest['stages']['ordinary-v2:0']['status'], 'reused')
    def test_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = SegmentCache(directory)
            self.assertIsNone(cache.load('key'))
            cache.save('key', {'video':torch.ones(2,3)})
            self.assertTrue(torch.equal(cache.load('key')['video'], torch.ones(2,3)))
            next(Path(directory).glob('*.pt')).write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, '完整性'):
                cache.load('key')

    def test_content_invalidates(self):
        self.assertNotEqual(fingerprint(torch.ones(2)), fingerprint(torch.zeros(2)))
        self.assertNotEqual(fingerprint(['prompt-a',42]), fingerprint(['prompt-b',42]))

    def test_sampling_resume_and_explicit_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            test_sampler.Tests().run_fake(False, cache_directory=directory, explicit_plan=True)

    def test_explicit_plan_whole_refine(self):
        test_sampler.Tests().run_fake(True, explicit_plan=True)

    def test_selflift_boundary_cache_preserves_x0_and_audio_state(self):
        nested = types.ModuleType('comfy.nested_tensor')
        nested.NestedTensor = test_sampler.Nested
        calls = []
        def sample(*args):
            calls.append(True)
            return {'samples':test_sampler.Nested((torch.ones(1,24,7,2,2), torch.full((1,32,2,37), .4))),
                    '_denoised_samples':test_sampler.Nested((torch.full((1,24,7,2,2), .7), torch.full((1,32,2,37), .9)))}
        params = dict(model=object(), cond=[], negative=[], latent={}, sampler=object(),
                      sigmas=torch.tensor([1.,.5]), seed=4, cfg=1, stage='selflift-boundary-v1')
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'comfy.nested_tensor':nested}):
            first, hit = cached_sample(directory, sample, **params)
            self.assertFalse(hit)
            resumed, hit = cached_sample(directory, sample, **params)
            self.assertTrue(hit)
            self.assertEqual(len(calls), 1)
            self.assertTrue(torch.equal(first['samples'].tensors[1], resumed['samples'].tensors[1]))
            self.assertTrue(torch.equal(first['_denoised_samples'].tensors[0], resumed['_denoised_samples'].tensors[0]))
            _, hit = cached_sample(directory, sample, **dict(params, stage='ordinary-v1'))
            self.assertFalse(hit)


if __name__ == '__main__':
    unittest.main()
