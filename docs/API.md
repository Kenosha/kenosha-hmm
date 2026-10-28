# Original numerical interface

The preserved numerical modules accept the original labelled pandas structures.
They have no ATMAN imports. The top-level array API performs validation and format
conversion; callers of these lower-level routines supply the described invariants.

- `HInit.HInit(words, num_states=3, ...)`: a numeric DataFrame indexed by
  `IndexWord, Frame`, with unique sequence/frame pairs; columns named `Var`.
  Returns `(A, avg, cov)`. `avg` has state index `State` (1 through N), `cov` has
  `State, Var` index and one row per state/feature, using the same feature order.
- `HRest.HRest(words, initial_A=..., initial_avg=..., initial_cov=..., ...)`:
  isolated-sequence Baum–Welch training with the same format/result.
- `HERest.HERest(models, data, num_cycles=..., tikhonov_regularizer=...)`:
  embedded re-estimation of a model dictionary `{label: {'A':..., 'avg':...,
  'cov':...}}`. Data is a numeric-feature DataFrame indexed by `Frame`, with
  extra `IndexSF` (sequence identifier) and `Event` (model label) columns. Repeated
  contiguous labels define each sequence's transcription; the algorithm estimates
  boundaries. Frames are ordered within each sequence. It returns a new dictionary.
- `HVite.token_pass_single_word(models, observations)` returns ranked best-path
  scores. `HVite.token_pass_sequence` retains the original unconstrained token
  passing interface. Use the tested top-level `classify` and `decode` where possible.
- `TokenPass.infer(data, models, grammar, outer_grammar_keyword='<all>')`:
  grammar-constrained decoding on an integer frame index. The top-level `decode`
  validates grammar/input and converts row positions before calling this routine.
- `high_level.init_all_single_mixture` retains the historical batch initialization
  convenience routine on an `Event, IndexWord, Frame` MultiIndex.

Example of embedded training using existing models:

```python
import pandas as pd
from kenosha_hmm.HERest import HERest

# models: a mapping of labels to fitted GaussianHMM objects
# x: ordered numeric observations; labels: same-length frame labels
training = pd.DataFrame({'f0': x, 'IndexSF': 0, 'Event': labels})
training.index.name = 'Frame'
parameters = HERest({name: model.as_legacy() for name, model in models.items()},
                    training, num_cycles=3, tikhonov_regularizer=1e-6)
```

No training observations, fitted ATMAN models or video-derived samples ship with
this package. The executable example and test fixtures are generated synthetically.
