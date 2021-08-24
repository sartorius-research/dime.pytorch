import pytest
import torch

from dime import DIME


def test_creation():
    # Should work.
    DIME()
    DIME(explained_variance_threshold=10)
    DIME(explained_variance_threshold=0.9)
    DIME(n_percentiles=10)


def test_raises_on_bad_arguments():
    with pytest.raises(ValueError):
        DIME(explained_variance_threshold=1.1)
    with pytest.raises(ValueError):
        DIME(explained_variance_threshold=-1)
    with pytest.raises(ValueError):
        DIME(explained_variance_threshold=-.1)
    with pytest.raises(ValueError):
        DIME(n_percentiles=-1)