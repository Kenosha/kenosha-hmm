"""Small pandas-index helper used by the historical numerical routines."""
import pandas as pd


def prepend_index_level(frame, name, value):
    return pd.concat([frame], keys=[value], names=[name])
