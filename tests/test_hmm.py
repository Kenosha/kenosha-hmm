import contextlib
import importlib
import io
import itertools
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from kenosha_hmm import GaussianHMM, fit, classify, decode, parse_grammar, read_grammar
from kenosha_hmm.HRest import log_forward, log_backward, elnsum
from kenosha_hmm.HERest import HERest
from kenosha_hmm.utils import calculate_emission_densities


def one_state(mean=0, variance=1, survival=.8):
    return GaussianHMM.from_arrays([[0, 1, 0], [0, survival, 1-survival], [0, 0, 0]],
                                   [[mean]], [[[variance]]])


class NumericalTests(unittest.TestCase):
    def test_forward_backward_against_enumerated_paths(self):
        A = np.array([[0, .7, .3, 0], [0, .5, .3, .2],
                      [0, .1, .6, .3], [0, 0, 0, 0]])
        emissions = np.array([[.1, .5], [.7, .2], [.3, .4]])
        paths = []
        for path in itertools.product(range(2), repeat=3):
            prob = A[0, path[0]+1] * emissions[0, path[0]]
            for t in range(1, 3):
                prob *= A[path[t-1]+1, path[t]+1] * emissions[t, path[t]]
            paths.append(prob * A[path[-1]+1, -1])
        alpha, logp = log_forward(emissions, A)
        beta = log_backward(emissions, A)
        self.assertAlmostEqual(math.exp(logp), sum(paths), places=14)
        np.testing.assert_allclose(np.exp(alpha+beta-logp).sum(axis=1), 1, atol=1e-13)

    def test_gaussian_density_and_total_likelihood(self):
        model = one_state(mean=2, variance=4)
        data = pd.DataFrame({'f0': [2., 4.]})
        actual = calculate_emission_densities(data, model.avg, model.cov).to_numpy().ravel()
        expected = np.array([1, math.exp(-.5)]) / math.sqrt(8*math.pi)
        np.testing.assert_allclose(actual, expected, rtol=1e-14)
        self.assertAlmostEqual(model.score([2, 4]), math.log(expected.prod() * .8 * .2))

    def test_stable_log_arithmetic(self):
        for module_name in ['HRest', 'HERest']:
            logsum = importlib.import_module('kenosha_hmm.'+module_name).elnsum
            with np.errstate(over='raise', invalid='raise'):
                self.assertAlmostEqual(float(logsum(1000, -1000)), 1000)
                self.assertEqual(float(logsum(-np.inf, -np.inf)), -np.inf)
                self.assertEqual(float(logsum(-np.inf, -10000)), -10000)

    def test_import_does_not_change_numpy_policy_or_load_atman(self):
        script = '''import numpy as np, sys
before=np.geterr().copy()
import kenosha_hmm
import kenosha_hmm.HERest
assert np.geterr()==before
assert not any(n.startswith(('ATMANpy', 'cv2', 'torch')) for n in sys.modules)
'''
        subprocess.run([sys.executable, '-c', script], check=True, cwd='/tmp', capture_output=True)

    def test_single_state_training_matches_analytic_moments(self):
        rng = np.random.default_rng(11)
        sequences = [rng.normal(3, .7, (20, 1)) for _ in range(6)]
        before = [v.copy() for v in sequences]
        with contextlib.redirect_stdout(io.StringIO()):
            model = fit(sequences, n_states=1, init_iterations=1, iterations=4)
        expected = np.concatenate(sequences)
        self.assertAlmostEqual(model.avg.iloc[0, 0], expected.mean(), places=12)
        self.assertAlmostEqual(model.cov.iloc[0, 0], expected.var(), places=12)
        self.assertAlmostEqual(model.A[1, -1], 1/20, places=12)
        for old, new in zip(before, sequences):
            np.testing.assert_array_equal(old, new)

    def test_training_and_classifying_distinct_sequences(self):
        rng = np.random.default_rng(12)
        low = [rng.normal(0, .5, (20, 1)) for _ in range(5)]
        high = [rng.normal(6, .5, (20, 1)) for _ in range(5)]
        with contextlib.redirect_stdout(io.StringIO()):
            models = {name: fit(sequences, n_states=2, init_iterations=2, iterations=3)
                      for name, sequences in [('low', low), ('high', high)]}
        self.assertEqual(classify(models, rng.normal(0, .5, 15)).index[0], 'low')
        self.assertEqual(classify(models, rng.normal(6, .5, 15)).index[0], 'high')

    def test_embedded_reestimation_keeps_valid_models_and_learns_means(self):
        rng = np.random.default_rng(24)
        frames = []
        for sequence in range(6):
            frames.append(pd.DataFrame({'f0': np.r_[rng.normal(0, .4, 8), rng.normal(6, .4, 8)],
                                        'IndexSF': sequence, 'Event': ['low']*8+['high']*8},
                                       index=pd.Index(range(16), name='Frame')))
        models = {'low': one_state(-.5).as_legacy(), 'high': one_state(6.5).as_legacy()}
        with contextlib.redirect_stdout(io.StringIO()):
            trained = HERest(models, pd.concat(frames), num_cycles=2, tikhonov_regularizer=1e-6)
        self.assertLess(abs(trained['low']['avg'].iloc[0, 0]), .3)
        self.assertLess(abs(trained['high']['avg'].iloc[0, 0]-6), .3)
        for params in trained.values():
            self.assertTrue(np.isfinite(params['A']).all())
            np.testing.assert_allclose(params['A'][:-1].sum(axis=1), 1, atol=1e-10)
            self.assertGreater(params['cov'].iloc[0, 0], 0)
        self.assertEqual(models['low']['avg'].iloc[0, 0], -.5)


class PublicInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.models = {'low': one_state(0), 'high': one_state(6)}

    def test_grammar_sequence_and_alternative(self):
        sequence = np.array([0, .1, -.1, 6, 6.1, 5.9])
        self.assertEqual(decode(self.models, sequence, 'all ::= low high'), [(2, 'low'), (5, 'high')])
        self.assertEqual(decode(self.models, [6, 6, 6], 'all ::= low | high'), [(2, 'high')])
        self.assertEqual(decode(self.models, sequence, 'all ::= <pair>\npair ::= low   high'),
                         [(2, 'low'), (5, 'high')])

    def test_recursive_grammar(self):
        sequence = [0, 0, 6, 6, 0, 0, 6, 6]
        grammar = 'all ::= <pair> | <pair> <all>\npair ::= low high'
        self.assertEqual(decode(self.models, sequence, grammar),
                         [(1, 'low'), (3, 'high'), (5, 'low'), (7, 'high')])

    def test_grammar_file_has_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'grammar.txt';file.write_text('# test\nall ::= low high\n')
            self.assertEqual(read_grammar(file), {'all': 'low high'})
        for bad in ['', 'all low', 'all ::= low\nall ::= high', 'all ::=']:
            with self.subTest(grammar=bad), self.assertRaises(ValueError):parse_grammar(bad)

    def test_round_trip_and_feature_order(self):
        model = GaussianHMM.from_arrays([[0,1,0],[0,.5,.5],[0,0,0]], [[2,5]],
                                         [[[2,.3],[.3,1]]], ['x','y'])
        observations = pd.DataFrame({'y': [4,5], 'x': [2,3]})
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'model.npz';model.save(file);loaded=GaussianHMM.load(file)
            self.assertAlmostEqual(loaded.score(observations), model.score(observations))
            np.testing.assert_array_equal(loaded.A, model.A)
            pd.testing.assert_frame_equal(loaded.avg, model.avg)
            pd.testing.assert_frame_equal(loaded.cov, model.cov)

    def test_invalid_parameters_and_inputs_are_rejected(self):
        for bad in [[], [np.nan], [[np.inf]], np.zeros((2,2,2))]:
            with self.subTest(values=bad), self.assertRaises(ValueError): self.models['low'].score(bad)
        with self.assertRaises(ValueError): self.models['low'].score([0], density_floor=0)
        with self.assertRaises(ValueError): fit([], n_states=1)
        with self.assertRaises(ValueError): fit([[1,2]], n_states=2)
        with self.assertRaises(ValueError): decode(self.models,[1,2], 'all ::= missing')
        with self.assertRaises(ValueError): decode(self.models,[1,2], 'all ::= <missing>')
        with self.assertRaises(ValueError): decode(self.models,[1,2], 'all ::= low |')
        with self.assertRaises(ValueError): decode(self.models,[1,2], 'all ::= <all> | low')
        with self.assertRaises(ValueError): GaussianHMM.from_arrays([[0,1,0],[0,.8,.2],[0,0,0]],[[0]],[[[0]]])

    def test_impossible_grammar_path_is_reported(self):
        with self.assertRaises(ValueError): decode(self.models,[0], 'all ::= low high')


if __name__ == '__main__':
    unittest.main()
