[![Build Status](https://github.com/algo-hhu/FLSpp/actions/workflows/mypy-flake-test.yml/badge.svg)](https://github.com/algo-hhu/FLSpp/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Supported Python version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Stable Version](https://img.shields.io/pypi/v/flspp?label=stable)](https://pypi.org/project/flspp/)

# FLS++

An implementation of the FLS++ algorithm for k-means Clustering, as presented in [1]. This is an improvement of the LS++ algorithm presented by Lattanzi and Sohler in [2].

One can think of the algorithm as working in 3 phases:

1. Initializing centers with k-means++ (as presented in [3]).
2. Improving the center set by performing local search swaps (for details, see [1]).
3. Converging the solution with LLoyd's algorithm (also known as "the" k-means algorithm).

In [2] it is shown that `O(k log log k)` local search iterations yield a constant factor approximation. However, in both [1] and [2] it is shown that, in practice, a very small number of iterations (e.g. 20) already yields very good results, at very reasonable runtime.

The interface is built in the same way as scikit-learn's KMeans for better compatibility.

In the following plots, we compare the performance of FLS++ (GFLS++) with various local search steps (5, 10, 15) with k-means++ (kM++), greedy k-means++ (GkM++) and the local search algorithm [2] with 25 local search steps (GLS++). The results are computed for the [KDD Phy Test](https://www.kdd.org/kdd-cup/view/kdd-cup-2004/data) and the [Tower](https://www.worldscientific.com/doi/abs/10.1142/S0218195908002787) datasets and averaged over 50 runs.


<p align="center">
  <img src="https://github.com/algo-hhu/FLSpp/blob/main/images/boxplots.png" alt="Boxplot Comparison for FLS++"/>
</p>

## References

[1] Theo Conrads, Lukas Drexler, Joshua Könen, Daniel R. Schmidt and Melanie Schmidt. "Local Search k-means++ with Foresight" (2024)

[2] Silvio Lattanzi and Christian Sohler. A better k-means++ algorithm via local search. In Proc.444
of the 36th ICML, volume 97 of Proceedings of Machine Learning Research, pages 3662–3671.445
PMLR, 09–15 Jun 2019

[3] David Arthur and Sergei Vassilvitskii. K-means++: The advantages of careful seeding. In409
Proceedings of the 18th SODA, page 1027–1035, USA, 2007

## Installation

```bash
pip install flspp
```

## Example

```python
from flspp import FLSpp

example_data = [
    [1.0, 1.0, 1.0],
    [1.1, 1.1, 1.1],
    [1.2, 1.2, 1.2],
    [2.0, 2.0, 2.0],
    [2.1, 2.1, 2.1],
    [2.2, 2.2, 2.2],
]

flspp = FLSpp(n_clusters=2)
labels = flspp.fit_predict(example_data)
centers = flspp.cluster_centers_

print(labels) # [1, 1, 1, 0, 0, 0]
print(centers) # [[2.1, 2.1, 2.1], [1.1, 1.1, 1.1]]
```

## Development

Install [poetry](https://python-poetry.org/docs/#installation)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Install clang
```bash
sudo apt-get install clang
```

Set clang variables
```bash
export CXX=/usr/bin/clang++
export CC=/usr/bin/clang
```

Install the package
```bash
poetry install
```

If the installation does not work and you do not see the C++ output, you can build the package to see the stack trace
```bash
poetry build
```

Run the tests
```bash
poetry run python -m unittest discover tests -v
```

## Citation

If you use this code, please cite the following paper:

```
Theo Conrads, Lukas Drexler, Joshua Könen, Daniel R. Schmidt and Melanie Schmidt. "Local Search k-means++ with Foresight" (2024). Accepted at SEA 2024.
```
