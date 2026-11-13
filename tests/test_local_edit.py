"""CPU regressions for planning, guide placement, cropping and media assembly."""
import sys
import types
import unittest
from collections.abc import Mapping
from unittest.mock import patch

import torch

from test_sampler import Nested, empty, VAE
from longvideo import local_edit as e
from longvideo import sampler as s


def video(n, value=0., height=32, width=32):
    return torch.full((n, height, width, 3), value)


def sound(n, value=0., rate=24000):
    return {'waveform': torch.full((1, 2, round(n*rate/24)), value), 'sample_rate': rate}


class LocalEditTests(unittest.TestCase):
    def plan(self, mode='replace_range', **overrides):
        kw = dict(source_video=video(100, .1), mode=mode, start_frame=24, end_frame=72,
                  new_frames=30, context_frames=22, audio_mode='preserve_source', audio_fade_ms=0)
        kw.update(overrides)
        return e.SolarLH3LocalEditPlan().build(**kw)[0]

    def test_modes_and_exclusive_range(self):
        for mode, expected in [('replace_range', (24, 48, 28)), ('insert_at', (24, 30, 76)),
                               ('extend', (100, 30, 0)), ('bridge', (100, 30, 50))]:
            with self.subTest(mode=mode):
                p = self.plan(mode, following_video=video(50))
                self.assertEqual((len(p['left']), p['frames'], len(p['right'])), expected)

    def test_invalid_ranges(self):
        for kw in (dict(start_frame=0), dict(end_frame=101), dict(start_frame=72),
                   dict(mode='insert_at', start_frame=101),
                   dict(mode='extend', new_frames=0), dict(context_frames=23)):
            with self.subTest(kw=kw), self.assertRaises(ValueError): self.plan(**kw)

    def test_optional_following_video_falls_back_to_extend(self):
        for options in ({}, {'following_video': None}, {'following_video': video(0)}):
            with self.subTest(options=list(options)):
                plan = self.plan('bridge', **options, following_audio=sound(50, .4))
                self.assertEqual((plan['requested_mode'], plan['mode']), ('bridge', 'extend'))
                self.assertEqual(len(plan['right']), 0)
                self.assertIsNone(plan['right_audio'])
                result, guides, _ = self.run_fake(plan)
                self.assertEqual((result[5], len(result[0])), (130, 30))
                self.assertEqual([(p, n) for p, n, _ in guides], [(0, 22)])
                self.assertEqual(result[3]['waveform'].shape[-1], 130000)

    def test_malformed_following_video_is_not_treated_as_absent(self):
        for following in ('', [], torch.zeros(0)):
            with self.subTest(type=type(following).__name__), self.assertRaises(ValueError):
                self.plan('bridge', following_video=following)

    def test_bridge_fits_different_following_sizes_and_preserves_timing(self):
        for height, width in ((64, 64), (32, 64), (64, 32)):
            with self.subTest(size=(height, width)):
                following = video(50, .4, height=height, width=width)
                p = self.plan('bridge', following_video=following,
                              source_audio=sound(100, .1), following_audio=sound(50, .2))
                result, guides, _ = self.run_fake(p)
                self.assertEqual(p['right'].shape, (50, 32, 32, 3))
                self.assertEqual(result[1].shape, (180, 32, 32, 3))
                self.assertTrue(torch.equal(result[1][130:], p['right']))
                self.assertTrue(torch.all(result[1][:100] == .1))
                self.assertTrue(torch.all(result[3]['waveform'][..., 130000:] == .2))
                self.assertEqual([(pos, n) for pos, n, _ in guides], [(0, 22), (52, 22)])
                self.assertTrue(torch.equal(guides[1][2], p['right'][:22]))
                self.assertTrue(torch.all(following == .4))
                self.assertIn('等比例适配', result[-1])
                if width > height:
                    self.assertTrue(torch.all(p['right'][:, :8] == 0))
                    self.assertTrue(torch.all(p['right'][:, 8:24] == .4))
                elif height > width:
                    self.assertTrue(torch.all(p['right'][:, :, :8] == 0))
                    self.assertTrue(torch.all(p['right'][:, :, 8:24] == .4))
                else:
                    self.assertTrue(torch.all(p['right'] == .4))

    def test_optional_audio_inputs_are_independent(self):
        for left in (None, sound(0), sound(100, .1)):
            for right in (None, sound(0), sound(50, .2)):
                with self.subTest(left=left is not None, right=right is not None):
                    p = self.plan('bridge', following_video=video(50),
                                  source_audio=left, following_audio=right, audio_fade_ms=20)
                    result, guides, calls = self.run_fake(p)
                    expected_left = left is not None and left['waveform'].shape[-1] > 0
                    expected_right = right is not None and right['waveform'].shape[-1] > 0
                    anchors = calls[0][0][0][1]['minimax_keyframes']
                    self.assertEqual([g['audio'] is not None for g in anchors],
                                     [expected_left, expected_right])
                    self.assertEqual(len(guides), 2)
                    self.assertEqual(result[5], 180)
                    wave = result[3]['waveform']
                    self.assertEqual(wave.shape[-1], 180000)
                    if not expected_left:
                        self.assertIsNone(p['source_audio'])
                        self.assertEqual(wave[..., :100000].count_nonzero(), 0)
                    if not expected_right:
                        self.assertIsNone(p['right_audio'])
                        self.assertEqual(wave[..., 130000:].count_nonzero(), 0)
                    self.assertTrue(torch.all(wave[..., 100000:130000] == .7))

    def test_audio_omitted_on_both_sides(self):
        p = self.plan('bridge', following_video=video(50))
        self.assertIsNone(p['source_audio'])
        self.assertIsNone(p['right_audio'])
        result, _, calls = self.run_fake(p)
        self.assertTrue(all(g['audio'] is None for g in calls[0][0][0][1]['minimax_keyframes']))
        self.assertEqual(result[5], 180)

    def test_lazy_audio_mapping_reaches_both_guides(self):
        class LazyAudio(Mapping):
            def __init__(self, frames):
                self.frames, self.data, self.loads = frames, None, 0

            def __getitem__(self, key):
                if self.data is None:
                    self.data = sound(self.frames, .3)
                    self.loads += 1
                return self.data[key]

            def __iter__(self):
                return iter(('waveform', 'sample_rate'))

            def __len__(self):
                return 2

        source, following = LazyAudio(100), LazyAudio(50)
        p = self.plan('bridge', following_video=video(50),
                      source_audio=source, following_audio=following)
        result, _, calls = self.run_fake(p)
        anchors = calls[0][0][0][1]['minimax_keyframes']
        self.assertTrue(all(g['audio'] is not None for g in anchors))
        self.assertTrue(torch.all(result[3]['waveform'][..., :100000] == .3))
        self.assertTrue(torch.all(result[3]['waveform'][..., 130000:] == .3))
        self.assertEqual((source.loads, following.loads), (1, 1))
        empty_audio = LazyAudio(0)
        self.assertIsNone(self.plan('extend', source_audio=empty_audio)['source_audio'])

    def test_malformed_audio_is_not_silently_ignored(self):
        for bad in ('', {}, {'waveform': torch.zeros(0), 'sample_rate': 24000},
                    dict(sound(0), sample_rate=0)):
            for name in ('source_audio', 'following_audio'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.plan('bridge', following_video=video(50), **{name: bad})

    def test_context_lengths(self):
        self.assertEqual([e.context_length(n, 22) for n in (1, 4, 5, 21, 22, 100)], [1, 1, 5, 5, 22, 22])

    def run_fake(self, plan, positive=None, negative=None):
        names = ['comfy', 'comfy.nested_tensor', 'comfy.model_management', 'comfy_extras',
                 'comfy_extras.nodes_minimax_h3', 'comfy_extras.nodes_audio', 'node_helpers']
        modules = {name: types.ModuleType(name) for name in names}
        modules['comfy'].nested_tensor = modules['comfy.nested_tensor']
        modules['comfy.nested_tensor'].NestedTensor = Nested
        modules['comfy.model_management'].throw_exception_if_processing_interrupted = lambda: None
        h3 = modules['comfy_extras.nodes_minimax_h3']
        h3.FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
        h3.align_frame_count = lambda n: n+(5-n)%17
        h3.video_latent_t = lambda n: ((n-5)//17)*5+2
        h3._empty_av_latent = empty
        guides, samples = [], []
        def guide(cond, latent, pos, **kw):
            guides.append((pos, len(kw['image']), kw['image'].clone()))
            metadata = dict(cond[0][1])
            metadata['minimax_keyframes'] = list(metadata.get('minimax_keyframes', [])) + [
                {'resolved_frame_index': pos, 'audio': kw.get('audio')}]
            return ([[cond[0][0], metadata]],)
        h3.MiniMaxH3AddGuide = types.SimpleNamespace(execute=guide)
        modules['comfy_extras.nodes_audio'].vae_decode_audio = lambda *args: sound(1000, .7)
        def sample(model, cond, neg, latent, sampler, sigmas, seed, cfg, mask=None):
            samples.append((cond, neg, latent))
            return dict(samples=latent['samples'], _denoised_samples=latent['samples'])
        decoder = VAE()
        original_decode = decoder.decode
        decoder.decode = lambda z: original_decode(z)+.9
        with patch.dict(sys.modules, modules), patch.object(s, 'sample_once', sample):
            result = e.SolarLH3LocalEditSampler().run(
                edit_plan=plan, model=object(), positive=positive or [[torch.ones(1, 2, 3), {}]],
                latent=empty(32, 32, 124)[0], vae=decoder, audio_vae=object(),
                sampler=object(), sigmas=torch.tensor([1., .5, 0.]), seed=42, cfg=1., negative=negative)
        return result, guides, samples

    def test_replace_preserves_pixels_duration_and_source_audio(self):
        original = video(100, .1)
        audio = sound(100, .3)
        result, guides, samples = self.run_fake(self.plan(source_video=original, source_audio=audio))
        patch_images, composed, patch_audio, composed_audio, fps, count, window, report = result
        self.assertEqual([(pos, n) for pos, n, _ in guides], [(0, 22), (70, 22)])
        self.assertEqual(len(samples), 1)
        self.assertEqual(patch_images.shape, (48, 32, 32, 3))
        self.assertTrue(torch.equal(composed[:24], original[:24]))
        self.assertTrue(torch.equal(composed[72:], original[72:]))
        self.assertTrue(torch.allclose(composed[24:72], torch.full_like(patch_images, .9)))
        self.assertTrue(torch.equal(composed_audio['waveform'], audio['waveform']))
        self.assertEqual((fps, count), (24., 100))
        self.assertEqual(window['frame_count'], 107)
        self.assertEqual((window['patch_start_frame'], window['patch_frame_count']), (22, 48))
        self.assertEqual(patch_audio['waveform'].shape[-1], 48000)
        self.assertFalse(torch.count_nonzero(original - .1))

    def test_extend_has_one_guide_and_exact_new_frames(self):
        result, guides, samples = self.run_fake(self.plan('extend'))
        self.assertEqual([(p, n) for p, n, _ in guides], [(0, 22)])
        self.assertEqual((len(result[0]), len(result[1]), result[5]), (30, 130, 130))
        self.assertEqual(result[3]['waveform'].shape[-1], 130000)
        self.assertEqual(result[3]['waveform'][..., :100000].count_nonzero(), 0)
        self.assertTrue(torch.all(result[3]['waveform'][..., 100000:] == .7))

    def test_bridge_preserves_both_videos_and_audio(self):
        p = self.plan('bridge', following_video=video(50, .2), source_audio=sound(100, .1), following_audio=sound(50, .2))
        result, guides, samples = self.run_fake(p)
        self.assertEqual([(pos, n) for pos, n, _ in guides], [(0, 22), (52, 22)])
        self.assertEqual(result[5], 180)
        self.assertTrue(torch.all(result[1][:100] == .1))
        self.assertTrue(torch.all(result[1][130:] == .2))
        wave = result[3]['waveform']
        self.assertTrue(torch.all(wave[..., :100000] == .1))
        self.assertTrue(torch.all(wave[..., 100000:130000] == .7))
        self.assertTrue(torch.all(wave[..., 130000:] == .2))

    def test_insert_and_tiny_patch(self):
        p = self.plan('insert_at', new_frames=1, start_frame=3)
        result, guides, _ = self.run_fake(p)
        self.assertEqual([(pos, n) for pos, n, _ in guides], [(0, 1), (2, 22)])
        self.assertEqual((len(result[0]), result[5]), (1, 101))

    def test_generated_patch_audio_and_original_right_offset(self):
        audio = sound(100)
        audio['waveform'][..., 72000:] = .4
        result, _, _ = self.run_fake(self.plan(source_audio=audio, audio_mode='generated_patch'))
        wave = result[3]['waveform']
        self.assertTrue(torch.all(wave[..., 24000:72000] == .7))
        self.assertTrue(torch.all(wave[..., 72000:] == .4))

    def test_generation_size_can_differ_from_original(self):
        p = self.plan(source_video=video(100, .2, height=64, width=48))
        result, _, _ = self.run_fake(p)
        self.assertEqual(result[0].shape[1:3], (32, 32))
        self.assertEqual(result[1].shape[1:3], (64, 48))

    def test_reject_external_guides_and_unmapped_video_refs(self):
        for metadata in ({'minimax_keyframes': [{}]}, {'minimax_refs': [{'kind': 'video'}]}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.run_fake(self.plan(), [[torch.ones(1, 2, 3), metadata]])

    def test_explicit_negative_forwarded(self):
        neg = [[torch.zeros(1, 2, 3), {}]]
        _, _, samples = self.run_fake(self.plan(), negative=neg)
        self.assertIs(samples[0][1], neg)

    def test_extend_audio_guide_uses_synced_tail(self):
        audio = sound(100)
        audio['waveform'].copy_(torch.arange(100000).reshape(1, 1, -1)/100000)
        _, _, calls = self.run_fake(self.plan('extend', source_audio=audio))
        guide = calls[0][0][0][1]['minimax_keyframes'][0]['audio']
        self.assertEqual(guide['sample_rate'], 24000)
        self.assertTrue(torch.equal(guide['waveform'], audio['waveform'][..., 78000:100000]))

    def test_both_audio_guides_use_correct_source_offsets(self):
        audio = sound(100)
        audio['waveform'].copy_(torch.arange(100000).reshape(1, 1, -1)/100000)
        _, _, calls = self.run_fake(self.plan(source_audio=audio))
        guides = calls[0][0][0][1]['minimax_keyframes']
        self.assertTrue(torch.equal(guides[0]['audio']['waveform'], audio['waveform'][..., 2000:24000]))
        self.assertTrue(torch.equal(guides[1]['audio']['waveform'], audio['waveform'][..., 72000:94000]))
        following = sound(50, .4)
        _, _, calls = self.run_fake(self.plan('bridge', source_audio=audio,
                                               following_video=video(50), following_audio=following))
        guides = calls[0][0][0][1]['minimax_keyframes']
        self.assertTrue(torch.equal(guides[1]['audio']['waveform'], following['waveform'][..., :22000]))

    def test_audio_context_can_be_disabled(self):
        _, _, calls = self.run_fake(self.plan(source_audio=sound(100), audio_context='off'))
        self.assertTrue(all(g['audio'] is None for g in calls[0][0][0][1]['minimax_keyframes']))

    def test_missing_audio_not_replaced_with_silent_guide(self):
        _, _, calls = self.run_fake(self.plan('extend', source_audio=sound(50)))
        self.assertIsNone(calls[0][0][0][1]['minimax_keyframes'][0]['audio'])

    def test_crossfade_does_not_move_audio_or_modify_sources(self):
        source, following = sound(100, .1), sound(50, .2)
        result, _, _ = self.run_fake(self.plan('bridge', following_video=video(50), source_audio=source,
                                              following_audio=following, audio_fade_ms=20))
        wave = result[3]['waveform']
        self.assertEqual(wave.shape[-1], 180000)
        self.assertTrue(torch.all(wave[..., :99520] == .1))
        self.assertTrue(torch.all(wave[..., 130480:] == .2))
        self.assertAlmostEqual(float(wave[0, 0, 99999]), .7)
        self.assertAlmostEqual(float(wave[0, 0, 100000]), .7)
        self.assertAlmostEqual(float(wave[0, 0, 129999]), .7)
        self.assertAlmostEqual(float(wave[0, 0, 130000]), .7)
        self.assertTrue(torch.all(source['waveform'] == .1))
        self.assertTrue(torch.all(following['waveform'] == .2))

    def test_preserve_source_bypasses_fade(self):
        source = sound(100, .3)
        result, _, _ = self.run_fake(self.plan(source_audio=source, audio_fade_ms=20))
        self.assertTrue(torch.equal(result[3]['waveform'], source['waveform']))

    def test_audio_options_validate(self):
        for options in (dict(audio_context='unknown'), dict(audio_fade_ms=-1), dict(audio_fade_ms=201)):
            with self.subTest(options=options), self.assertRaises(ValueError): self.plan(**options)

    def test_audio_guide_upmixes_mono(self):
        audio = {'waveform': torch.ones(1, 1, 100000), 'sample_rate': 24000}
        guide = e.audio_guide(audio, 78, 22)
        self.assertEqual(guide['waveform'].shape, (1, 2, 22000))

    def test_fade_stays_inside_short_tail_context(self):
        result, _, _ = self.run_fake(self.plan('bridge', following_video=video(4),
            source_audio=sound(100, .1), following_audio=sound(4, .2), audio_fade_ms=200))
        self.assertIn('audio_blend_samples=4800/1000', result[-1])
        self.assertTrue(torch.all(result[3]['waveform'][..., 131000:] == .2))


if __name__ == '__main__': unittest.main()
