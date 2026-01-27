import ctypes
import os
from time import time
from typing import Any, Optional, Sequence, Union, cast

import numpy as np
from sklearn import get_config
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
)
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
    check_non_negative,
    validate_data,
)

import flspp._core  # type: ignore

_DLL = ctypes.cdll.LoadLibrary(flspp._core.__file__)


class FLSpp(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator
):
    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 100,
        local_search_iterations: int = 20,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.local_search_iterations = local_search_iterations
        self.random_state = random_state

    def check_metadata_routing(self) -> None:
        if get_config().get("enable_metadata_routing", False):
            raise NotImplementedError(
                "FLSpp has not yet been tested with metadata routing."
            )

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Any = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "FLSpp":
        self._validate_flspp_params()

        _X = validate_data(
            self,
            X,
            reset=True,
            accept_sparse=False,
            dtype=np.float64,
            order="C",
            accept_large_sparse=False,
            copy=False,
        )

        _sample_weight = _validate_sample_weight(sample_weight, _X)
        self._n_threads = self._effective_n_threads()

        n_samples = _X.shape[0]
        self.n_features_in_ = _X.shape[1]

        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}."
            )

        assert isinstance(_X, np.ndarray), type(_X)

        _seed = int(time()) if self.random_state is None else self.random_state

        _X = np.ascontiguousarray(_X)
        # Declare c types
        c_array = _X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_weight = _sample_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_n = ctypes.c_uint(n_samples)
        c_d = ctypes.c_uint(self.n_features_in_)
        c_k = ctypes.c_uint(self.n_clusters)
        c_ll_iterations = ctypes.c_int(self.max_iter)
        c_ls_iterations = ctypes.c_int(self.local_search_iterations)
        c_random_state = ctypes.c_size_t(_seed)

        centers = np.empty(
            (self.n_clusters, self.n_features_in_), dtype=np.float64, order="C"
        )
        labels = np.empty(n_samples, dtype=np.int32, order="C")

        c_centers = centers.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_labels = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        c_iter = ctypes.c_int()

        _DLL.cluster.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # X
            ctypes.POINTER(ctypes.c_double),  # weight
            ctypes.c_uint,  # n_samples
            ctypes.c_uint,  # n_features_in_
            ctypes.c_uint,  # n_clusters
            ctypes.c_int,  # max_iter
            ctypes.c_int,  # ls_iterations
            ctypes.c_size_t,  # random_state
            ctypes.POINTER(ctypes.c_int),  # labels
            ctypes.POINTER(ctypes.c_double),  # centers
            ctypes.POINTER(ctypes.c_int),  # n_iter
        ]

        # Set the return type to double
        _DLL.cluster.restype = ctypes.c_double
        cost = _DLL.cluster(
            c_array,
            c_weight,
            c_n,
            c_d,
            c_k,
            c_ll_iterations,
            c_ls_iterations,
            c_random_state,
            c_labels,
            c_centers,
            ctypes.byref(c_iter),
        )

        self.inertia_ = cost
        self.cluster_centers_ = centers
        self.labels_ = labels
        self._n_features_out = len(self.cluster_centers_)
        self.n_iter_ = c_iter.value

        return self

    def fit_predict(
        self,
        X: Sequence[Sequence[float]],
        y: Any = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        return self.fit(X, sample_weight=sample_weight).labels_

    def fit_transform(
        self,
        X: Sequence[Sequence[float]],
        y: Any = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        return self.fit(X, sample_weight=sample_weight)._transform(X)

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        check_is_fitted(self)
        X = self._check_test_data(X)
        labels, _ = pairwise_distances_argmin_min(X, self.cluster_centers_)
        return cast(np.ndarray, labels)

    def transform(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        check_is_fitted(self)

        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        return cast(np.ndarray, euclidean_distances(X, self.cluster_centers_))

    def score(
        self,
        X: Sequence[Sequence[float]],
        y: Any = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> Any:
        check_is_fitted(self)
        X = self._check_test_data(X)

        _, distances = pairwise_distances_argmin_min(X, self.cluster_centers_)
        sample_weight = _validate_sample_weight(sample_weight, X)

        return -np.sum(distances**2 * sample_weight)

    def _check_test_data(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            reset=False,
            dtype=np.float64,
            order="C",
            accept_large_sparse=False,
        )
        return X

    def _validate_flspp_params(self) -> None:
        if not isinstance(self.n_clusters, (int, np.integer)):
            raise TypeError(
                f"n_clusters must be an integer, got {type(self.n_clusters).__name__}"
            )
        if self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {self.n_clusters}")

        if not isinstance(self.max_iter, (int, np.integer)):
            raise TypeError(
                f"max_iter must be an integer, got {type(self.max_iter).__name__}"
            )
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")

        if not isinstance(self.local_search_iterations, (int, np.integer)):
            raise TypeError(
                f"local_search_iterations must be an integer,"
                f"got {type(self.local_search_iterations).__name__}"
            )
        if self.local_search_iterations < -1:
            raise ValueError(
                f"local_search_iterations must be >= -1, got {self.local_search_iterations}"
            )

        if self.random_state is not None:
            if not isinstance(self.random_state, (int, np.integer)):
                raise TypeError(
                    f"random_state must be None or an integer,"
                    f"got {type(self.random_state).__name__}"
                )
            if self.random_state < 0:
                raise ValueError(f"random_state must be >= 0, got {self.random_state}")

    def _effective_n_threads(self) -> int:
        return os.cpu_count() or 1


def _validate_sample_weight(
    sample_weight: Optional[Union[float, np.ndarray, Sequence[float]]],
    X: np.ndarray,
) -> np.ndarray:
    n_samples = X.shape[0]

    if sample_weight is None:
        return np.ones(n_samples, dtype=np.float64)

    if isinstance(sample_weight, (int, float, np.integer, np.floating)):
        if sample_weight < 0:
            raise ValueError("sample_weight must be non-negative")
        return np.full(n_samples, sample_weight, dtype=np.float64)

    sw_array = check_array(
        sample_weight,
        accept_sparse=False,
        ensure_2d=False,
        dtype=np.float64,
        order="C",
        input_name="sample_weight",
    )

    assert isinstance(sw_array, np.ndarray)

    if sw_array.ndim != 1:
        raise ValueError(
            f"sample_weight must be 1D array or scalar, got {sw_array.ndim}D"
        )

    check_consistent_length(sw_array, X)
    check_non_negative(sw_array, "sample_weight")

    return sw_array
