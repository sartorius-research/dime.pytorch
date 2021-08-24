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

from typing import Union, Tuple

import torch
import numpy as np

__version__ = '1.0.0'


class NotFitted(Exception):
    """ Exception indicating that DIME hyperplane-approximation is not fitted. """


class NotCalibrated(Exception):
    """ Exception indicating that DIME percentiles are not calibrated. """


class DIME:

    """ Distance to Modelled Embedding (DIME)
    
    This is a scikit-learn-esque PyTorch-implementation of DIME as described by Sjögren & Trygg and
    is used to enable ANNs detect out-of distribution observations.

    Parameters
    ----------
    explained_variance_threshold : float, int (default 0.99)
        Either a float between 0 and 1, which indicate the ratio of explained
        variance threshold used to determine the rank of the hyperplane approximation,
        or an int that specifies the rank directly.
    n_percentiles : int (default 5000)
        Number of discrete percentiles that will be used for probability lookups. A higher
        number indicate more fine-grained probability estimation. A value of 100 indicate
        that percentiles correspond to whole percentages.

    Examples
    --------
    Given a 2D-tensor, fit the hyperplane.

    >>> x = torch.tensor(...) # N x P torch 2D float-tensor.
    >>> modelled_embedding = DIME().fit(x)

    To obtain probabilities, calibrate percentiles. Preferably against
    separate dataset. Chaining is fine.:

    >>> x_cal = torch.tensor(...)  # N_cal x P torch 2D float-tensor.
    >>> modelled_embedding = DIME().fit(x).calibrate(x_cal)

    Given fitted hyperplane, you can calculate distances on new observations:

    >>> x_new = torch.tensor(...)  # N_new x P 2D float-tensor.
    >>> modelled_embedding.distance_to_hyperplane(x_new)  # -> 1D float-tensor, length N_new

    To obtain probabilities of that the new observations have a distance
    calibration set observations are equal or less than the new distance,
    you need to have calibrated the percentiles as shown above. Then you
    receive the probablities by passing `return_probablities`-keyword:

    >>> modelled_embedding.distance_to_hyperplane(x_new, return_probabilites=True) # -> 1D float-tensor, length N_new

    You can also use the alternative formulation of distance within the hyperplane, optionally as probabilities:

    >>> modelled_embedding.distance_within_hyperplane(x_new)  # -> 1D float-tensor, length N_new
    """
    
    def __init__(self, explained_variance_threshold: Union[float, int] = 0.99, n_percentiles: int = 5000):
        if isinstance(explained_variance_threshold, float) and not (0 <= explained_variance_threshold <= 1):
            raise ValueError('float param explained_variance_threshold should be between 0 and 1 when float')
        if isinstance(explained_variance_threshold, int) and explained_variance_threshold < 1:
            raise ValueError('integer param explained_variance_threshold should be positive')
        if isinstance(n_percentiles, int) and n_percentiles < 1:
            raise ValueError('param n_percentiles should be positive')
        self.explained_variance_threshold = explained_variance_threshold
        self.hyperplane_basis_vectors = None
        self.explained_variance = None
                
        self._embedded_mean = None
        self._d_within_histogram = None
        self._d_from_histogram = None
        self._precision = None

        # Specify the percentiles that will be available for probability lookups.
        self._histogram_percentiles = torch.linspace(0, 100, n_percentiles)
        
    def fit(self, x: torch.Tensor, calibrate_against_trainingset: bool = False) -> "DIME":
        """ Fit hyperplane and optionally calibrate percentiles against training-set. """
        scores, self.hyperplane_basis_vectors, self.explained_variance = fit_svd(x, self.explained_variance_threshold)
        self._embedded_mean = torch.mean(scores, dim=0)
        cov = covariance(scores - self._embedded_mean[None], assume_centered=True)
        self._precision = torch.inverse(cov)

        if calibrate_against_trainingset:
            self.calibrate(x)

        return self

    def calibrate(self, x: torch.Tensor) -> "DIME":
        """ Calibrate percentiles to enable probabilities. """
        percentiles = self._histogram_percentiles.cpu().numpy()
        rss = self.residual_sum_of_squares(x, dim=1).detach().cpu().numpy()
        self._d_from_histogram = torch.FloatTensor(np.percentile(np.sqrt(rss), percentiles))

        scores = self.transform(x) - self._embedded_mean[None]
        mahal = squared_mahalanobis_distance(scores, self._precision).detach().cpu().numpy()
        self._d_within_histogram = torch.FloatTensor(np.percentile(np.sqrt(mahal), percentiles))
        return self
        
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """ Project observations on hyperplane. """
        return torch.mm(x, self.hyperplane_basis_vectors)
    
    def inverse_transform(self, scores: torch.Tensor) -> torch.Tensor:
        """ Project observations projected on hyperplane back to data-space. """ 
        return torch.mm(scores, self.hyperplane_basis_vectors.t())
    
    def residual_sum_of_squares(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """ Calculate sum-of-squares residual of reconstruction based on hyperplane. """
        residuals = x - self.inverse_transform(self.transform(x))
        rss = (residuals ** 2).sum(dim=dim)
        return rss
    
    def distance_to_hyperplane(self, x: torch.Tensor, return_probabilities: bool = False) -> torch.Tensor:
        """ Distance to hyperplane (DIME), optionally given as probabilities. """
        if not self._is_fitted:
            raise NotFitted('Hyperplane-approximation must be fitted using DIME.fit(x: torch.Tensor) before '
                            'obtaining distance to hyperplane')
        dime = torch.sqrt(self.residual_sum_of_squares(x, dim=1))

        if return_probabilities:
            return self._calculate_probability(dime, self._d_from_histogram)
        else:
            return dime
        
    def distance_within_hyperplane(self, x: torch.Tensor, return_probabilities: bool = False) -> torch.Tensor:
        """ Distance withing hyperplane (D-within), optionally given as probabilities. """
        if not self._is_fitted:
            raise NotFitted('Hyperplane-approximation must be fitted using DIME.fit(x: torch.Tensor) before '
                            'obtaining distance within hyperplane')
        scores = self.transform(x) - self._embedded_mean[None]
        squared_mahal = squared_mahalanobis_distance(scores, self._precision)
        mahal = torch.sqrt(squared_mahal)
        if return_probabilities:
            return self._calculate_probability(mahal, self._d_within_histogram)
        else:
            return mahal

    def _calculate_probability(self, distances: torch.Tensor, distance_histogram: torch.Tensor) -> torch.Tensor:
        if not self._is_calibrated:
            raise NotCalibrated('Percentiles must be calibrated using DIME.calibrate(x: torch.Tensor) before '
                                'obtaining probability estimates.')
        n_bins = len(distance_histogram)
        repeated_distances = distances.repeat(n_bins, 1)

        histogram_thresholded_distances = (repeated_distances < distance_histogram[:, None])
        cdf_indices = histogram_thresholded_distances.int().argmin(0)

        probabilities = self._histogram_percentiles[cdf_indices] / 100
        return probabilities

    @property
    def _is_calibrated(self):
        is_calibrated = (self._d_within_histogram is not None) and (self._d_from_histogram is not None)
        return is_calibrated

    @property
    def _is_fitted(self):
        is_fitted = self.hyperplane_basis_vectors is not None
        return is_fitted


def covariance(x: torch.Tensor, assume_centered: bool = False) -> torch.Tensor:
    """ Calculate empirical covariance matrix.. """
    n_samples, n_features = x.shape
    if not assume_centered:
        x = x - torch.mean(x, 0).view(-1, n_features)
        
    cov = (1 / (n_samples - 1)) * torch.mm(x.t(), x)
    return cov


def squared_mahalanobis_distance(x: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
    mahal = (torch.mm(x, precision) * x).sum(dim=1)
    return mahal


def fit_svd(x: torch.Tensor, n_components: Union[int, float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Fit hyperplane using singular value decomposition. 
    
    Parameters
    ----------
    x : tensor
        2D N x C tensor of observations.
    n_components : float, int
        Either a float between 0 and 1, which indicate the ratio of explained
        variance threshold used to determine the rank of the hyperplane approximation,
        or an int that specifies the rank directly.
    """
    u, s, v = torch.svd(x)
    explained_variance = (s.data ** 2) / (len(x) - 1)
    r2 = explained_variance / explained_variance.sum()
    
    if isinstance(n_components, float):
        cumulative_r2 = torch.cumsum(r2, 0)
        if n_components > r2[0]:
            n_components = (cumulative_r2 < n_components).int().argmax() + 1
        else:
            n_components = 1
        
    v = v[:, :n_components]
    scores = (u * s)[:, :n_components]
    return scores, v, r2[:n_components]
