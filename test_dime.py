# Copyright (c) 2021 Sartorius AG
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest
import torch

from dime import DIME, NotCalibrated, NotFitted


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


def test_fit_embedding():
    p = 30
    x = torch.randn(2000, p)
    modelled_embedding = DIME().fit(x)

    assert modelled_embedding.hyperplane_basis_vectors.shape[0] == p


def test_fit_embedding_with_specific_rank():
    p = 30
    rank = 5
    x = torch.randn(2000, p)
    modelled_embedding = DIME(explained_variance_threshold=5).fit(x)

    assert modelled_embedding.hyperplane_basis_vectors.shape[1] == rank


def test_fit_embedding_with_specific_ratio():
    p = 30
    ratio = 0.5
    x = torch.randn(2000, p)
    modelled_embedding = DIME(explained_variance_threshold=ratio).fit(x)

    assert modelled_embedding.explained_variance <= ratio


def test_get_distance_raises_when_not_fitted():
    x = torch.randn(2000, 30)
    modelled_embedding = DIME()

    with pytest.raises(NotFitted):
        modelled_embedding.distance_to_hyperplane(x)

    with pytest.raises(NotFitted):
        modelled_embedding.distance_within_hyperplane(x)


def test_get_distance_to_hyperplane():
    n = 2000
    x = torch.randn(n, 30)
    modelled_embedding = DIME().fit(x)

    dime = modelled_embedding.distance_to_hyperplane(x)
    assert dime.shape == (2000, )


def test_get_distance_within_hyperplane():
    n = 2000
    x = torch.randn(n, 30)
    modelled_embedding = DIME().fit(x)

    d_within_hyperplane = modelled_embedding.distance_within_hyperplane(x)
    assert d_within_hyperplane.shape == (2000, )


def test_get_probabilities_raises_when_not_calibrated():
    x = torch.randn(2000, 30)
    modelled_embedding = DIME().fit(x)

    with pytest.raises(NotCalibrated):
        modelled_embedding.distance_to_hyperplane(x, return_probabilities=True)

    with pytest.raises(NotCalibrated):
        modelled_embedding.distance_within_hyperplane(x, return_probabilities=True)


def test_get_distance_to_hyperplane_as_probability():
    n = 2000
    x = torch.randn(n, 30)
    modelled_embedding = DIME().fit(x).calibrate(x)
    probabilities = modelled_embedding.distance_to_hyperplane(x, return_probabilities=True)

    assert probabilities.shape == (2000, )
    assert (probabilities <= 1.0).all() and (probabilities >= 0.0).all()


def test_get_distance_within_hyperplane_as_probability():
    n = 2000
    x = torch.randn(n, 30)
    modelled_embedding = DIME().fit(x).calibrate(x)
    probabilities = modelled_embedding.distance_within_hyperplane(x, return_probabilities=True)

    assert probabilities.shape == (2000, )
    assert (probabilities <= 1.0).all() and (probabilities >= 0.0).all()
