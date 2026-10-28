# Kenosha HMM

Gaussian hidden Markov models for labelled sequences, with model training,
classification and grammar-constrained decoding. Written by Gianluca Vagliano
from the algorithm literature in 2022, originally used to recover tennis events
from video/audio observations in ATMAN.

The standalone package depends on **NumPy and pandas**. It needs no ATMAN checkout,
OpenCV, PyTorch, model weights or tennis data.

## Install

Use Python 3.10 or later. Install the tagged release from GitHub:

```bash
python -m pip install "kenosha-hmm @ git+https://github.com/Kenosha/kenosha-hmm.git@v0.1.0"
```

For local development and the synthetic example:

```bash
git clone https://github.com/Kenosha/kenosha-hmm.git
cd kenosha-hmm
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python examples/synthetic.py
```

The package is distributed through this repository; it is not published on PyPI.

## Train and classify

Each training item is a separate `(frames, features)` NumPy array or pandas
DataFrame. One-dimensional arrays represent a single feature. Sequence boundaries
are preserved. DataFrames can name features; arrays use `f0`, `f1`, and so on.

```python
import numpy as np
from kenosha_hmm import fit, classify, decode

rng = np.random.default_rng(7)
models = {
    label: fit([rng.normal(mean, .4, (20, 1)) for _ in range(8)], n_states=2)
    for label, mean in [('low', 0), ('high', 5)]
}
print(classify(models, rng.normal(5, .4, (12, 1))))

observations = np.r_[rng.normal(0, .4, 10), rng.normal(5, .4, 10)]
print(decode(models, observations, 'all ::= low high'))
# [(9, 'low'), (19, 'high')] — inclusive, zero-based end frames
```

`fit` uses Viterbi-based initialization followed by Baum–Welch re-estimation.
`classify` ranks the **best state-path** log scores. `model.score(observations)`
returns the **total** log likelihood, summing over possible state paths. They
measure different quantities; neither returns a calibrated class probability.

For supplied parameters use
`GaussianHMM.from_arrays(transitions, means, covariances, features=None)`.
The transition matrix includes silent entry/exit states, so its shape is
`(n_states + 2, n_states + 2)`; means have shape `(n_states, n_features)` and full
covariances `(n_states, n_features, n_features)`. Exit rows are zero, other rows
sum to one, and the model does not skip directly from entry to exit.

Models expose `A`, `avg`, `cov` and `as_legacy()` for the original pandas-based
numerical API. Save/load a model with an explicit path:

```python
from kenosha_hmm import GaussianHMM
models['low'].save('low.npz')
restored = GaussianHMM.load('low.npz')
```

The NPZ representation contains arrays and feature names; loading disables pickle.

## Grammars

Rules use bare nonterminal names, `<references>`, sequences and `|` alternatives:

```text
all ::= <pair> | <pair> <all>
pair ::= low high
```

`decode` accepts this text or a mapping such as `{'all': 'low high'}`. It supports
right recursion, rejects unknown symbols, empty alternatives and left recursion,
and reports when observations cannot complete a path. `read_grammar(path)` takes
an explicit UTF-8 file path; it never searches an ATMAN directory. Blank lines
and lines beginning with `#` are ignored. Observation rows are decoded in their
supplied order; returned frame numbers are zero-based row positions.

## Numerical API and scope

The original entry points remain importable from `kenosha_hmm.HInit`, `HRest`,
`HERest`, `HVite`, `TokenPass` and `high_level`. See [legacy data formats](docs/API.md)
for their pandas schemas and embedded-training example. Those names are retained
for compatibility; the smaller top-level API is recommended for new callers.

This implementation uses one full-covariance Gaussian per emitting state. It does
not provide Gaussian mixtures, missing-value imputation, automatic feature
scaling, GPU training or a scikit-learn estimator interface. Observations must be
finite and features consistently ordered. Training needs at least two frames per
state in each sequence and enough variation for positive-definite covariance
estimates. The covariance regularizer applies to the initializer; degenerate later
estimates are rejected. Training finds a local optimum and the original routines
print their progress. A density floor prevents zero Gaussian densities before the
log recursions; extreme-tail scores consequently depend on that floor.

Tennis postprocessing, ATMAN filesystem layout and its model/data CSV adapters
remain in ATMAN. The original authorship/algorithm references are retained. See
[origin](docs/ORIGIN.md) and [corrections made during extraction](docs/CHANGELOG.md).

## Check and build

```bash
python -m unittest discover -s tests -v
python examples/synthetic.py
python -m pip install build
python -m build
```

Tests check likelihoods against enumerated paths, Gaussian densities and analytic
single-state estimates, embedded training, recursive grammar backtraces, model
persistence, invalid inputs and absence of ATMAN/import-side effects. These are
software/numerical checks, not a claim of benchmark superiority or revalidation
of historical tennis accuracy. The CI configuration runs tests and builds; it
does not publish packages.

## Licence

[MIT](LICENSE). Original implementation by Gianluca Vagliano, 2022; extracted as
a standalone library in October 2026. See [origin and references](docs/ORIGIN.md).
