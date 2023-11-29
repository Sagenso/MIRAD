# MIRAD: A Method for Interpretable Ransomware Attack Detection

In this repository we open-source a method for training interpretable models,
originally developed for ransomware attack detection.

Key features:
- model and prediction interpretability
- concise implementation based on `numpy` and `pytorch`
- efficiency: training can be executed on a CPU, evaluation is fast
- extensibility: model structure allows easy enforcement of task-specific requirements by additional regularization

We also include everything necessary to reproduce experiments we conducted to publish our results.

## How it works

All the implementation is in [lib/embedding_sum.py](lib/embedding_sum.py) and consists of three classes:
 - Digitizer
 - EmbeddingSumModule
 - EmbeddingSumClassifier

The method can be summarized as follows:

1. For each feature, the `Digitizer` defines quantile-based bins and encodes feature values as bin ordinals,
2. The `EmbeddingSumModule` defines a parameter for each bin of each feature and implements evaluation as sum of the
   parameters corresponding to bin ordinals in the input vector,
3. The `EmbeddingSumClassifier` trains the `EmbeddingSumModule` using gradient descent and a compound loss function
   combining binary cross-entropy with regularization terms that encourage desirable properties of the model.

In other words, we train an [additive model](https://en.wikipedia.org/wiki/Additive_model) composed of
[step functions](https://en.wikipedia.org/wiki/Step_function).
The model can be interpreted by plotting the step functions.
Usage example can be found in the [tutorial notebook](tutorial.ipynb).


## Running the code

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
