import ctypes
from numbers import Integral
from time import time
from typing import Any, Optional, Sequence

import numpy as np
from sklearn._config import get_config
from sklearn.cluster import KMeans
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import _check_sample_weight

import flspp._core  # type: ignore

_DLL = ctypes.cdll.LoadLibrary(flspp._core.__file__)


class FLSpp(KMeans):

    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "lloyd_iterations": [Interval(Integral, 1, None, closed="left")],
        "local_search_iterations": [Interval(Integral, 1, None, closed="left")],
        "random_state": [None, Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        n_clusters: int,
        lloyd_iterations: int = 100,
        local_search_iterations: int = 100,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.lloyd_iterations = lloyd_iterations
        self.local_search_iterations = local_search_iterations
        self.random_state = random_state

    def check_metatadata_routing(self) -> None:
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
        if sample_weight is not None:
            raise NotImplementedError("Sample weights are not yet supported.")

        self._validate_params()

        _X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )

        sample_weight = _check_sample_weight(sample_weight, X, dtype=type(X))
        self._n_threads = _openmp_effective_n_threads()

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
        c_n = ctypes.c_uint(n_samples)
        c_d = ctypes.c_uint(self.n_features_in_)
        c_k = ctypes.c_uint(self.n_clusters)
        c_ll_iterations = ctypes.c_uint(self.lloyd_iterations)
        c_ls_iterations = ctypes.c_uint(self.local_search_iterations)
        c_random_state = ctypes.c_size_t(_seed)
        c_labels = (ctypes.c_int * n_samples)()
        c_centers = (ctypes.c_double * self.n_features_in_ * self.n_clusters)()

        c_iter = ctypes.c_int()

        # Set the return type to double
        _DLL.cluster.restype = ctypes.c_double
        cost = _DLL.cluster(
            c_array,
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

        self.cluster_centers_ = np.ctypeslib.as_array(
            c_centers, shape=(self.n_clusters, self.n_features_in_)
        )

        self.labels_ = np.ctypeslib.as_array(c_labels)

        self._n_features_out = len(self.cluster_centers_)
        self.n_iter_ = c_iter.value

        return self
