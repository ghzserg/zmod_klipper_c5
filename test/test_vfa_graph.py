# Проверки диагностического режима без принтера.
import importlib.util
import pathlib
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    'vfa_graph', pathlib.Path(__file__).resolve().parents[1]
    / 'klippy/extras/vfa_graph.py')
vfa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vfa)


def fft(t, a, freqs, fs):
    return [float(np.max(np.abs(a))) for _ in freqs], freqs


class TestVfaGraph(unittest.TestCase):
    def test_frequency_limit_and_cruise(self):
        t = np.arange(0., 1.02, 1. / 1600.)
        x = np.sin(2. * np.pi * 100. * t)
        samples = np.column_stack((t, x, np.zeros_like(t), np.zeros_like(t)))
        result = vfa.spectrum(samples, 0., 1.02, 100., 5000., 100.,
                              [100., 400., 800.], fft)
        self.assertTrue(np.isfinite(result[:2]).all())
        self.assertTrue(np.isnan(result[2]))
        self.assertIsNone(vfa.spectrum(samples, 0., 2., 100., 5000., 100.,
                                       [100.], fft))
        self.assertIsNone(vfa.spectrum([], 0., 1.02, 100., 5000., 100.,
                                       [100.], fft))

    def test_restore_attempts_all_harmonics(self):
        calls = []

        class Motor:
            def set_resonance_damping(self, h, *value):
                calls.append((h, value))
                if h == 1:
                    raise RuntimeError('test')

        values = {'x': {h: (h, 2, 3) for h in vfa.HARMONICS}}
        with self.assertRaises(RuntimeError):
            vfa.restore_damping({'x': Motor()}, values)
        self.assertEqual(calls, [(h, (h, 2, 3)) for h in vfa.HARMONICS])


if __name__ == '__main__':
    unittest.main()
