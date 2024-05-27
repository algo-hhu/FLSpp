import unittest
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import InvalidParameterError

from flspp import FLSpp


def manual_transform(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.sqrt(((X - center) ** 2).sum(axis=1)) for center in centers], axis=-1
    )


def calculate_costs(
    points: np.ndarray, centers: np.ndarray
) -> Tuple[np.ndarray, float]:
    cost = 0
    labels = np.zeros(len(points), dtype=int)
    for i, p in enumerate(points):
        mincost = np.inf
        for k, c in enumerate(centers):
            dist = ((p - c) ** 2).sum()
            if dist < mincost:
                mincost = dist
                labels[i] = k
        cost += mincost
    return labels, cost


def assert_equals_computed(flspp: FLSpp, data: np.ndarray) -> None:
    labels, cost = calculate_costs(np.array(data), flspp.cluster_centers_)
    assert flspp.inertia_ is not None and np.isclose(
        flspp.inertia_, cost
    ), f"Inertia: {flspp.inertia_} vs. cost {cost}"

    score = flspp.score(data)
    assert np.isclose(
        flspp.inertia_, -score
    ), f"Inertia: {flspp.inertia_} vs. score {-score}"

    pred_labels = flspp.predict(data)
    assert all(pred_labels == labels)


class TestFLSPP(unittest.TestCase):

    example_data = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.1, 1.1, 1.1],
            [1.2, 1.2, 1.2],
            [2.0, 2.0, 2.0],
            [2.1, 2.1, 2.1],
            [2.2, 2.2, 2.2],
        ]
    )

    def test_fit(self) -> None:

        flspp = FLSpp(n_clusters=2, random_state=42)
        flspp.fit(self.example_data)

        labels = flspp.predict(self.example_data)

        assert_equals_computed(flspp, self.example_data)

        assert all(labels == [1, 1, 1, 0, 0, 0])

    def test_fit_predict(self) -> None:
        flspp = FLSpp(n_clusters=2, random_state=42)
        labels = flspp.fit_predict(self.example_data)
        assert all(labels == [1, 1, 1, 0, 0, 0])

    def test_fit_transform(self) -> None:
        flspp = FLSpp(n_clusters=2)
        flspp.set_output(transform="pandas")
        transformed = flspp.fit_transform(self.example_data)

        transformed_manual = manual_transform(self.example_data, flspp.cluster_centers_)
        assert np.allclose(transformed, transformed_manual)

    def test_score(self) -> None:
        flspp = FLSpp(n_clusters=2)
        flspp.fit(self.example_data)

        score = flspp.score(self.example_data)

        assert score is not None and np.isclose(score, -0.12)

        new_arr = np.random.rand(100, 3)

        score = flspp.score(new_arr)

        _, cost = calculate_costs(new_arr, flspp.cluster_centers_)

        assert np.isclose(score, -cost), f"Score: {score}, Cost: {cost}"

    def test_transform(self) -> None:
        flspp = FLSpp(n_clusters=2)
        flspp.fit(self.example_data)

        transformed = flspp.transform(self.example_data)

        transformed_manual = manual_transform(self.example_data, flspp.cluster_centers_)
        assert np.allclose(transformed, transformed_manual)

    def test_dataframe(self) -> None:
        feature_names = ["A", "B", "C"]
        data = pd.DataFrame(self.example_data, columns=feature_names)

        flspp = FLSpp(n_clusters=2)
        flspp.fit(data)

        assert_equals_computed(flspp, data.values)

        assert flspp.feature_names_in_ is not None and all(
            flspp.feature_names_in_ == feature_names
        )

        _ = flspp.get_feature_names_out()

        _ = flspp.get_params()

    def test_rectangles(self) -> None:
        with open("datasets/rectangles.txt") as f:
            data = [list(map(float, line.split())) for line in f]

        flspp = FLSpp(n_clusters=36)
        flspp.fit(data)

        assert_equals_computed(flspp, data)

    def test_pr91(self) -> None:
        with open("datasets/pr91.txt") as f:
            data = [list(map(float, line.split())) for line in f.readlines()[1:]]

        flspp = FLSpp(n_clusters=16)
        flspp.fit(data)

        assert_equals_computed(flspp, data)

    def test_phy(self) -> None:
        with open("datasets/phy_test_features.dat") as f:
            data = [list(map(float, line.split())) for line in f]

        flspp = FLSpp(n_clusters=6)
        flspp.fit(data)

        assert_equals_computed(flspp, data)

    def test_n_clusters(self) -> None:
        flspp = FLSpp(n_clusters=0)

        self.assertRaises(InvalidParameterError, flspp.fit, self.example_data)


if __name__ == "__main__":
    unittest.main()
