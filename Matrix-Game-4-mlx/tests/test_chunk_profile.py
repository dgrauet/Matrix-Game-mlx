"""Chunk timer and warm-up summary (port of upstream tests/test_chunk_profile.py::test_timer_and_warmup)."""
import unittest
from unittest.mock import patch

from matrix_game_mlx.profiling import ChunkTimer, summarize


class ProfileTests(unittest.TestCase):
    def test_timer_and_warmup(self):
        timer = ChunkTimer(True)
        with patch('matrix_game_mlx.profiling.time.perf_counter', side_effect=[0., .001, .002, .004]):
            with timer.measure('work_ms'):
                pass
            with timer.measure('work_ms'):
                pass
        self.assertAlmostEqual(timer.values['work_ms'], 3.)
        disabled = ChunkTimer(False)
        with disabled.measure('work_ms'):
            pass
        self.assertEqual(disabled.values, {})
        result = summarize([dict(chunk=i, processing_ms=float(i)) for i in range(30)])
        self.assertEqual(result['chunks'], 5)
        self.assertEqual(result['stages']['processing_ms']['mean'], 27.)
        self.assertEqual(summarize([])['stages'], {})
