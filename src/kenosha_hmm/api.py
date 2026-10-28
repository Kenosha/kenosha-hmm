"""Small array-oriented interface to the original Gaussian HMM algorithms."""
from dataclasses import dataclass
from collections.abc import Mapping
import json

import numpy as np
import pandas as pd

from .HInit import HInit
from .HRest import HRest, log_forward
from .HVite import token_pass_single_word
from .TokenPass import infer, parse_grammar
from .utils import calculate_emission_densities


def _observations(values, features=None):
    if isinstance(values, pd.DataFrame):
        if features is None:
            features = list(values.columns)
        if set(values.columns) != set(features):
            raise ValueError('Observation features must match the model')
        array = values.loc[:, features].to_numpy(dtype=float)
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array[:, None]
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError('Observations must be a nonempty (frames, features) array')
    if not np.isfinite(array).all():
        raise ValueError('Observations must be finite')
    if features is None:
        features = [f'f{i}' for i in range(array.shape[1])]
    if len(features) != array.shape[1] or len(set(features)) != len(features):
        raise ValueError('Feature names must be unique and match the array width')
    if not all(isinstance(name, str) for name in features):
        raise ValueError('Feature names must be strings')
    result = pd.DataFrame(array.copy(), columns=features)
    result.columns.name = 'Var'
    result.index.name = 'Frame'
    return result


@dataclass
class GaussianHMM:
    """One full-covariance Gaussian per emitting state, with entry/exit states.

    ``A`` includes the non-emitting entry (first) and exit (last) states.
    ``avg`` and ``cov`` use the original labelled pandas representation.
    Use ``from_arrays`` when supplying parameters directly.
    """
    A: np.ndarray
    avg: pd.DataFrame
    cov: pd.DataFrame

    @classmethod
    def from_arrays(cls, transitions, means, covariances, features=None):
        means = np.asarray(means, dtype=float)
        covariances = np.asarray(covariances, dtype=float)
        transitions = np.asarray(transitions, dtype=float)
        if means.ndim != 2 or min(means.shape, default=0) < 1:
            raise ValueError('Means must have shape (states, features)')
        n, d = means.shape
        if transitions.shape != (n + 2, n + 2):
            raise ValueError('Transitions must include entry and exit states')
        if covariances.shape != (n, d, d):
            raise ValueError('Covariances must have shape (states, features, features)')
        if not all(np.isfinite(a).all() for a in (transitions, means, covariances)):
            raise ValueError('Parameters must be finite')
        if (transitions < 0).any() or not np.allclose(transitions[:-1].sum(axis=1), 1):
            raise ValueError('Non-exit transition rows must sum to one')
        if np.any(transitions[-1]) or np.any(transitions[:, 0]) or transitions[0, -1]:
            raise ValueError('Entry/exit transitions do not match the model topology')
        for covariance in covariances:
            if not np.allclose(covariance, covariance.T):
                raise ValueError('Covariances must be symmetric')
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError('Covariances must be positive definite') from exc
        features = list(features) if features is not None else [f'f{i}' for i in range(d)]
        _observations(means, features)  # Validate feature names and width.
        states = pd.Index(range(1, n + 1), name='State')
        avg = pd.DataFrame(means.copy(), index=states, columns=features)
        avg.columns.name = 'Var'
        cov = pd.DataFrame(covariances.reshape(n * d, d).copy(),
                           index=pd.MultiIndex.from_product([states, features], names=['State', 'Var']),
                           columns=features)
        cov.columns.name = 'Var'
        return cls(transitions.copy(), avg, cov)

    def as_legacy(self):
        """Return the parameter mapping accepted by the original numerical API."""
        return {'A': self.A, 'avg': self.avg, 'cov': self.cov}

    def score(self, observations, density_floor=1e-100):
        """Return total log likelihood, summing over all possible state paths."""
        data = _observations(observations, list(self.avg.columns))
        _check_floor(density_floor)
        emissions = calculate_emission_densities(data, self.avg, self.cov, density_floor)
        return float(log_forward(emissions.to_numpy(), self.A)[1])

    def save(self, path):
        """Write arrays and feature names to an explicit NPZ path (no pickle)."""
        n, d = self.avg.shape
        with open(path, 'wb') as handle:
            np.savez_compressed(handle, A=self.A, means=self.avg.to_numpy(),
                                covariances=self.cov.to_numpy().reshape(n, d, d),
                                features=np.array(json.dumps(list(self.avg.columns))))

    @classmethod
    def load(cls, path):
        """Load a model saved by :meth:`save` and validate its parameters."""
        with np.load(path, allow_pickle=False) as data:
            return cls.from_arrays(data['A'], data['means'], data['covariances'],
                                   json.loads(str(data['features'])))


def _check_floor(value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError('density_floor must be finite and positive')


def fit(sequences, n_states=3, *, init_iterations=5, iterations=10, features=None,
        covariance_regularization=1e-6, density_floor=1e-100):
    """Train on separate observed sequences using Viterbi initialization and EM.

    Each sequence is a (frames, features) array or DataFrame. Frames within each
    sequence are ordered; sequence boundaries are never joined. The historical
    routines report their iteration progress to stdout. This is a local optimum,
    not a guarantee of the best model or calibrated out-of-domain performance.
    """
    if not isinstance(n_states, int) or n_states < 1:
        raise ValueError('n_states must be a positive integer')
    for name, value in [('init_iterations', init_iterations), ('iterations', iterations)]:
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not np.isfinite(covariance_regularization) or covariance_regularization < 0:
        raise ValueError('covariance_regularization must be finite and nonnegative')
    _check_floor(density_floor)
    frames = []
    for sequence in sequences:
        data = _observations(sequence, features)
        if features is None:
            features = list(data.columns)
        if len(data) < n_states * 2:
            raise ValueError('Each training sequence needs at least two frames per state')
        frames.append(data)
    if not frames:
        raise ValueError('At least one training sequence is required')
    words = pd.concat(frames, keys=range(len(frames)), names=['IndexWord'])
    A, avg, cov = HInit(words, num_states=n_states, min_iterations=1,
                        max_iterations=init_iterations + 1, density_floor=density_floor)
    d = len(avg.columns)
    covariance_array = cov.to_numpy().reshape(n_states, d, d)
    covariance_array = covariance_array + covariance_regularization * np.eye(d)[None, :, :]
    initial = GaussianHMM.from_arrays(A, avg.to_numpy(), covariance_array, features)
    A, avg, cov = HRest(words, initial_A=initial.A, initial_avg=initial.avg,
                        initial_cov=initial.cov, min_iterations=1,
                        max_iterations=iterations + 1, density_floor=density_floor)
    return GaussianHMM.from_arrays(A, avg.to_numpy(), cov.to_numpy().reshape(n_states, d, d), features)


def _models(models):
    if not isinstance(models, Mapping) or not models:
        raise ValueError('Provide a nonempty mapping of names to GaussianHMM models')
    if not all(isinstance(model, GaussianHMM) for model in models.values()):
        raise TypeError('Every model must be a GaussianHMM')
    features = list(next(iter(models.values())).avg.columns)
    if any(list(model.avg.columns) != features for model in models.values()):
        raise ValueError('Models must use the same ordered features')
    return {name: model.as_legacy() for name, model in models.items()}, features


def classify(models, observations, *, density_floor=1e-100):
    """Rank named models by their best (Viterbi) path log score, highest first.

    Unlike ``GaussianHMM.score``, these scores maximise over state paths instead
    of summing their probabilities.
    """
    params, features = _models(models)
    _check_floor(density_floor)
    with np.errstate(divide='ignore'):
        return token_pass_single_word(params, _observations(observations, features), density_floor)


def decode(models, observations, grammar, *, start='all'):
    """Decode a grammar-constrained sequence as (inclusive_end_frame, label) pairs.

    ``grammar`` is BNF text or a mapping of bare nonterminal names to expressions.
    Rules support sequences, ``|`` alternatives and ``<nonterminal>`` references.
    No epsilon productions or left recursion are supported. The outer production must be completable
    by the observations; an impossible path raises ValueError.
    """
    params, features = _models(models)
    rules = parse_grammar(grammar) if isinstance(grammar, str) else dict(grammar)
    start = start.strip('<>')
    if start not in rules:
        raise ValueError(f'Missing start rule: {start}')
    for name, expression in rules.items():
        if not isinstance(name, str) or not isinstance(expression, str):
            raise ValueError('Grammar names and expressions must be strings')
        alternatives = [' '.join(part.split()) for part in expression.split('|')]
        if any(not part for part in alternatives):
            raise ValueError('Empty grammar alternatives are unsupported')
        rules[name] = ' | '.join(alternatives)
        for token in ' '.join(alternatives).split():
            if token.startswith('<') and token.endswith('>'):
                if token[1:-1] not in rules:
                    raise ValueError(f'Unknown grammar rule: {token}')
            elif token not in params:
                raise ValueError(f'Unknown model: {token}')
    # Left recursion would expand indefinitely before consuming a frame.
    visiting, visited = set(), set()
    def check_left_recursion(name):
        if name in visiting:
            raise ValueError('Left-recursive grammar rules are unsupported')
        if name in visited:
            return
        visiting.add(name)
        for alternative in rules[name].split('|'):
            first = alternative.split()[0]
            if first.startswith('<') and first.endswith('>'):
                check_left_recursion(first[1:-1])
        visiting.remove(name)
        visited.add(name)
    for name in rules:
        check_left_recursion(name)
    data = _observations(observations, features)
    with np.errstate(divide='ignore'):
        result = infer(data, params, rules, outer_grammar_keyword=f'<{start}>')
    if not result or result[-1][0] != len(data) - 1:
        raise ValueError('No complete path through the grammar for these observations')
    return result
