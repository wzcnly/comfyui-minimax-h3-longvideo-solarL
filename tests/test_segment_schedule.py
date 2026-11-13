"""Frame-grid regressions for explicit long-video schedules."""
import unittest
import torch

from test_sampler import s
from longvideo.planning import parse_segment_schedule


class ScheduleTests(unittest.TestCase):
    def test_variable_segments_have_exact_total_and_aligned_latents(self):
        for overlap in (5, 22, 39, 56):
            rows = parse_segment_schedule([
                {'frames': 124, 'prompt': 'first'},
                {'frames': 102}, {'frames': 77}], overlap, 'common')
            self.assertEqual([r['prompt'] for r in rows], ['first', 'common', 'common'])
            total = sum(r['frames'] for r in rows)
            parts, cursor = [], 0
            for i, row in enumerate(rows):
                trim = overlap if i else 0
                count = row['frames'] + trim
                length = max(22, count) + (5-max(22, count)) % 17
                tokens = 2 if length <= 5 else (length-5)//17*5+2
                trim_tokens = 2 if trim == 5 else (trim-5)//17*5+2 if trim else 0
                parts.append(torch.zeros(1, 24, tokens-trim_tokens, 2, 2))
                cursor += row['frames']
            full = max(22, total)+(5-max(22, total))%17
            self.assertEqual(sum(p.shape[2] for p in parts), (full-5)//17*5+2)
            self.assertEqual(cursor, 303)

    def test_invalid_middle_boundary_rejected(self):
        with self.assertRaisesRegex(ValueError, '网格'):
            parse_segment_schedule([{'frames':120}, {'frames':30}], 22)
        with self.assertRaisesRegex(ValueError, '网格'):
            parse_segment_schedule([{'frames':124}, {'frames':96}, {'frames':30}], 22)
        with self.assertRaisesRegex(ValueError, '重叠'):
            parse_segment_schedule([{'frames':22}, {'frames':30}], 56)


if __name__ == '__main__':
    unittest.main()
