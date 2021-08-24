import logging

import torch
import numpy as np


class DIME:

    """ Distance to Modelled Embedding (DIME)
    
    This is a scikit-learn-esque PyTorch-implementation of DIME as described by Sjögren & Trygg and
    is used to enable ANNs detect out-of distribution observations.

    Parameters
    ----------
    r2_threshold : float, int (default 0.99)
        Either a float between 0 and 1, which indicate the ratio of explained
        variance threshold used to determine the rank of the hyperplane approximation,
        or an int that specifies the rank directly.

    Examples
    --------
    Given a 2D-tensor, fit the hyperplane.
    >>> x  # N x C torch 2D float-tensor.
    >>> modelled_embedding = DIME().fit(x)

    To obtain probabilities, calibrate percentiles. Preferably against
    separate dataset. Chaining is fine.
    >>> x  # N x P torch 2D float-tensor.
    >>> x_cal  # N x P torch 2D float-tensor.
    >>> modelled_embedding = DIME().fit(x).calibrate(x_cal)

    Given fitted hyperplane, you can calculate distances on new observations
    >>> x_new  # N x P torch 2D float-tensor.
    >>> modelled_embedding.distance_to_hyperplane(x_new)

    Or within hyperplane.
    >>> x_new  # N x P torch 2D float-tensor.
    >>> modelled_embedding.distance_within_hyperplane(x_new)  # -> 1D float-tensor. 

    To obtain probabilities of that the new observations have a distance
    calibration set observations are equal or less than the new distance, 
    you need to have calibrated the percentiles as shown above. Then you
    receive the probablities by passing `return_probablities`-keyword:
    
    >>> x_new  # N x P torch 2D float-tensor.
    >>> modelled_embedding.distance_within_hyperplane(x_new, return_probabilites=True) # -> 1D float-tensor. 
    """
    
    def __init__(self, r2_threshold=0.99):
        self.r2_threshold = r2_threshold
        self.v = None
        self.r2 = None
                
        self._embedded_mean = None
        self._d_within_histogram = None
        self._d_from_histogram = None
        self._histogram_percentiles = torch.FloatTensor(np.concatenate([
            np.linspace(0, 10, 1000),
            np.linspace(10, 90, 800),
            np.linspace(90, 100, 1000)
        ]))
        
    def fit(self, x, calibrate_against_trainingset=False):
        """ Fit hyperplane and optionally calibrate percentiles against training-set. """
        n_samples, n_features = x.shape
        scores, self.v, self.r2 = fit_svd(x, self.r2_threshold)
        self._embedded_mean = torch.mean(scores, dim=0)
        cov = covariance(scores - self._embedded_mean[None], assume_centered=True)
        self.precision = torch.inverse(cov)

        if calibrate_against_trainingset:
            self.calibrate(x)

        return self

    def calibrate(self, x: torch.FloatTensor):
        """ Calibrate percentiles to enable probabilities. """
        percentiles = self._histogram_percentiles.cpu().numpy()
        rss = self.residual_sum_of_squares(x, dim=1).detach().cpu().numpy()
        self._d_from_histogram = torch.FloatTensor(np.percentile(np.sqrt(rss), percentiles))

        scores = self.transform(x) - self._embedded_mean[None]
        mahal = squared_mahalanobis_distance(scores, self.precision).detach().cpu().numpy()
        self._d_within_histogram = torch.FloatTensor(np.percentile(np.sqrt(mahal), percentiles))
        return self
        
    def transform(self, x):
        """ Project observations on hyperplane. """
        return torch.mm(x, self.v)
    
    def inverse_transform(self, scores):
        """ Project observations projected on hyperplane back to data-space. """ 
        return torch.mm(scores, self.v.t())
    
    def residual_sum_of_squares(self, x, dim=1):
        """ Calculate sum-of-squares residual of reconstruction based on hyperplane. """
        residuals = x - self.inverse_transform(self.transform(x))
        rss = (residuals ** 2).sum(dim=dim)
        return rss
    
    def distance_to_hyperplane(self, x, return_probablities=False):
        """ Distance to hyperplane (DIME), optionally given as probabilities. """
        dime = torch.sqrt(self.residual_sum_of_squares(x, dim=1))

        if return_probablities:
            return self._calculate_probability(dime, self._d_from_histogram)
        else:
            return dime
        
    def distance_within_hyperplane(self, x, return_probablities=False):
        """ Distance withing hyperplane (D-within), optionally given as probabilities. """
        scores = self.transform(x) - self._embedded_mean[None]
        squared_mahal = squared_mahalanobis_distance(scores, self.precision)
        mahal = torch.sqrt(mahal)
        if return_probablities:
            return self._calculate_probability(mahal, self._d_within_histogram)
        else:
            return mahal

    def _calculate_probability(self, distances, distance_histogram):
        n_bins = len(distance_histogram)
        repeated_distances = distances.repeat(n_bins, 1)

        histogram_thresholded_distances = (repeated_distances < distance_histogram[:, None])
        cdf_indices = histogram_thresholded_distances.int().argmin(0)

        probabilities = self._histogram_percentiles[cdf_indices] / 100
        return probabilities


def covariance(x, assume_centered=False):
    """ Calculate empirical covariance matrix.. """
    n_samples, n_features = x.shape
    if not assume_centered:
        x = x - torch.mean(x, 0).view(-1, n_features)
        
    cov = (1 / (n_samples - 1)) * torch.mm(x.t(), x)
    return cov


def squared_mahalanobis_distance(x, precision):
    mahal = (torch.mm(x, precision) * x).sum(dim=1)
    return mahal


def fit_svd(x, n_components):
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
