"""Gaussian HMM training, classification and grammar-constrained decoding."""
from .api import GaussianHMM, fit, classify, decode
from .TokenPass import parse_grammar, read_grammar

__version__ = '0.1.0'
__all__ = ['GaussianHMM', 'fit', 'classify', 'decode', 'parse_grammar', 'read_grammar']
