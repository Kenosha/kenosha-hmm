"""Train two models and decode an ordered sequence, using generated data only."""
import numpy as np
from kenosha_hmm import fit, classify, decode


def main():
    rng = np.random.default_rng(7)
    training = {
        'low': [rng.normal(0, .4, (20, 1)) for _ in range(8)],
        'high': [rng.normal(5, .4, (20, 1)) for _ in range(8)],
    }
    models = {label: fit(sequences, n_states=2, init_iterations=3, iterations=5)
              for label, sequences in training.items()}
    sample = rng.normal(5, .4, (12, 1))
    ranked = classify(models, sample)
    assert ranked.index[0] == 'high'
    print('Classification (best-path log scores):')
    print(ranked)
    observations = np.r_[rng.normal(0, .4, 10), rng.normal(5, .4, 10)]
    events = decode(models, observations, 'all ::= low high')
    assert events == [(9, 'low'), (19, 'high')]
    print('Decoded inclusive end frames and labels:', events)
    print('High-model total log likelihood:', models['high'].score(sample))


if __name__ == '__main__':
    main()
