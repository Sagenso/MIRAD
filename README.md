# MIRAD: A Method for Interpretable Ransomware Attack Detection

In this repository we open-source a method for training interpretable models,
originally developed for ransomware attack detection.

Key features:
- model and prediction interpretability
- concise implementation based on `numpy` and `pytorch`
- efficiency: training can be executed on a CPU, evaluation is fast
- extensibility: model structure allows easy enforcement of task-specific requirements by additional regularization

We also include everything necessary to reproduce the experiments we conducted to publish our results.

## How it works

The entire implementation can be found in [lib/embedding_sum.py](lib/embedding_sum.py) and consists of three classes:
 - Digitizer
 - EmbeddingSumModule
 - EmbeddingSumClassifier

The method can be summarized as follows:

1. For each feature, the `Digitizer` defines quantile-based bins and encodes feature values as bin ordinals,
2. The `EmbeddingSumModule` defines a trainable parameter for each bin of each feature
   and implements evaluation as sum of the trainable parameters corresponding to bin ordinals in the input vector,
3. The `EmbeddingSumClassifier` trains the `EmbeddingSumModule` using gradient descent and a compound loss function
   combining binary cross-entropy with regularization terms that encourage desirable properties of the model.

In other words, we train an [additive model](https://en.wikipedia.org/wiki/Additive_model) composed of
[step functions](https://en.wikipedia.org/wiki/Step_function).
The model can be interpreted by plotting the step functions.
Usage example can be found in the [tutorial notebook](notebooks/tutorial.ipynb).

## Experiments

The [data](data) directory contains data sets related to ransomware detection.
Each row corresponds to a moment in time during one of many simulated user sessions with ransomware attacks.
The target variable indicates whether the attack was already ongoing at that moment.
Each feature is based on a number of system events of a specific kind in a preceding fixed-length time window.
Helpers for loading this data are provided in [lib/data.py](lib/data.py).

Experiments that compare MIRAD to popular interpretable models are conducted in the
[experiments notebook](notebooks/experiments.ipynb).

## Running the notebooks

First, clone the repository.
Then, either install it directly and run jupyter

```shell
poetry install  # or: pip install -e .
jupyter notebook
```

or build and run a docker image
```shell
 docker build -t mirad .
 docker run --rm -it -p 8888:8888 mirad
```
