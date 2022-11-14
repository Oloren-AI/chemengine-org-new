---
layout: ../layouts/Content.astro
---

# Getting Started

Oloren ChemEngine (OCE) is an open-source Python library for developing and using machine learning models for molecular property prediction, validated on many [benchmarks](https://github.com/Oloren-AI/OCE-TDC). OCE also provides tools for uncertainty quantification, model-agnostic interpretability, interactive visualizations, model evaluation and management, and hyperparameter optimization, among many other useful tools for molecular property prediction.

This guide illustrates the main features that OCE provides. It assumes basic knowledge of machine learning techniques (e.g. model fitting, predicting, and evaluation) and chemistry related to *in silico* modeling (e.g. SMILES, molecular representations). Refer to our [installation guide](https://chemengine.org/install) for installing OCE.

## Datasets

OCE provides a dataset wrapper, BaseDataset, which simplifies the handling of tabular data for model fitting and predicting. OCE also provides various dataset splitting mechanisms, such as RandomSplit and ScaffoldSplit. While BaseDataset makes most workflows cleaner, it is not required to wrap datasets in a BaseDataset for use in OCE.

Here we load a demonstration dataset consisting of molecules and their pIC50 inhibition values for the VEGFR1 protein, compiled from the CHEMBL database.

```python
import olorenchemengine as oce
import pandas as pd

df = pd.read_csv("https://storage.googleapis.com/oloren-public-data/CHEMBL%20Datasets/997_2298%20-%20VEGFR1%20(CHEMBL1868).csv")
dataset = oce.BaseDataset(data = df.to_csv(), structure_col = "Smiles", property_col = "pChEMBL Value") + oce.ScaffoldSplit(split_proportions=[0.8,0.0,0.2])
```

The BaseDataset object accepts 3 parameters:
- A csv object corresponding to the data, `data`.
- The column name of molecule structures, `structure_col`. Molecules should be in SMILES string format.
- The column name of target values, `property_col`.

## Dataset Splitters
As in the above example, OCE implements dataset splitters such as ScaffoldSplit which automatically handle dataset splitting when combined with a BaseDataset. Adding the objects `BaseDataset + ScaffoldSplit` will split the dataframe in BaseDataset using `ScaffoldSplit`'s `split` function. In the above example, `dataset` is split by molecular scaffold into a roughly 80/20 train/test split.

The train, valid and test subsets can be accessed with the BaseDataset methods `train_dataset` and `test_dataset`.
```python
dataset.train_dataset
dataset.valid_dataset
dataset.test_dataset
```

### Data Visualization
Datasets can be visualized using a variety of built-in `BaseVisualization` classes for exploratory data analysis.

Here we plot the molecules of `dataset` in a Jupyter Notebook by taking the first 2 principal components of their corresponding Morgan Fingerprint.

```python
oce.ChemicalSpacePlot(dataset = dataset, rep = oce.MorganVecRepresentation()).render_ipynb()
```

![](https://chemengine.org/static/images/GS_CSpace.png)

We can hover over dots with the cursor and get a 2D structure preview of molecules at each dot.

Other data visualization classes include `VisualizeCompounds` which renders 2D molecular structures, `VisualizeTrainTest` which gives a `ChemicalSpacePlot`-like plot colored by train/test membership, and `VisualizeMoleculePerturbations` which gives a `ChemicalSpacePlot`-like plot for a molecule and several perturbations of the molecule.

## Model Creation and Fitting
OCE implements and unifies many molecular property prediction models into a single API. Models range from basic models such as SVMs and decision trees to advanced, state-of-art graph neural networks and gradient boosting learners.

Here we initialize a random forest model with 1000 estimators.

```python
model = oce.RandomForestModel(oce.MorganVecRepresentation(), n_estimators=1000)
model.fit(*dataset.train_dataset)
```

Models that intake a specific feature set (e.g. random forest, SVM, as opposed to a graph neural network) have the `representation` parameter, a `BaseRepresentation` object corresponding to the method in which molecules are featurized. In this case, we use Morgan Fingerprint which featurizes each molecule as a 2048 bit vector.

All model objects inherit from `BaseModel` and implement the `fit` method. When fitting a model, calculations to featurize molecules are performed and cached automatically. Unless specified with the setting parameter of `BaseModel`, models will automatically determine whether to train a classification or regression model. pIC50 is a continuous value so our model will train a random forest regressor.

If not using a BaseDataset, `fit` will accept parameters `X` and `y`, corresponding to the structures and property values of the data.

### Model Predicting and Evaluation
BaseModel defines a predict function which outputs predictions given SMILES strings. Hence we predict values for the test set with

```python
preds = model.predict(*dataset.test_dataset)
```

For model evaluation, we can use the `test` function of `BaseModel`. This evaluates various accuracy metrics for the model on the test set.

```python
model.test(*dataset.test_dataset)
```
```python
{'r2': 0.49527708861270214,
    'Spearman': 0.61408826476857,
    'Explained Variance': 0.5105688901771117,
    'Max Error': 2.4375165820677642,
    'Mean Absolute Error': 0.590457028157271,
    'Mean Squared Error': 0.6146704583132537,
    'Root Mean Squared Error': 0.7840092208088204}
```

## Hyperparameter Search with HyperOpt
OCE implements HyperOpt for hyperparameter optimization. HyperOpt uses the Tree of Parzen Estimators algorithm to find optimal hyperparameters. The following example optimizes our model using HyperOpt.

```python
model = oce.RandomForestModel(
   representation = oce.OptChoice("descriptor1", [oce.Mol2Vec(), oce.DescriptastorusDescriptor("rdkit2dnormalized"), oce.MorganVecRepresentation()]),
   n_estimators = oce.OptChoice("n_estimators1", [10, 500, 1000, 2000]),
   max_features = oce.OptChoice("max_features1", ["log2", "auto"]),
   max_depth = oce.OptChoice("max_depth1", [None, 3, 5, 10]),
   bootstrap = oce.OptChoice("bootstrap1", [True, False]),
   criterion = oce.OptChoice("criterion1", ["mse", "mae"]),
)

manager = oce.ModelManager(dataset, metrics="Root Mean Squared Error")
best = oce.optimize(model, manager, max_evals=100)

manager.get_model_database().sort_values(by="Root Mean Squared Error", ascending=True)

best_params = manager.get_model_database().sort_values(by="Root Mean Squared Error", ascending=True)['Model Parameters'][0]

print(best_params)

best_model = manager.best_model

>>>"{'BC_class_name': 'RandomForestModel', 'args': [], 'kwargs': {'representation': {'BC_class_name': 'MorganVecRepresentation', 'args': [], 'kwargs': {'radius': 2, 'nbits': 1024, 'log': True}}, 'n_estimators': 2000, 'max_features': 'log2', 'max_depth': 3, 'bootstrap': False, 'criterion': 'mse', 'class_weight': None}}"
```

To specify a hyperparameter search space, initialize a `BaseModel` with parameters of form:
```python
oce.OptChoice("Parameter Name", [Parameter Options])
```

OCE includes other options for defining the parameter search space, such as `OptRandInt` and `OptUniform`.

`ModelManager` is an object which can run and record test accuracy for many different models on the same dataset. For HyperOpt, we create a `ModelManager` with dataset and optimize it over the predefined parameter space. `get_model_database()` returns a  dataframe listing all model parameters and associated accuracy metrics. We also take the best performing model with the `best_model` attribute.

## Prediction Visualization
As with datasets, OCE provides various utilities to visualize results of a trained model. Here we use `VisualizeModelSim`, an object which plots a model’s predicted target value versus the true value for each molecule in the test set.
```
oce.VisualizeModelSim(dataset, best_model).render_ipynb()
```
![](https://chemengine.org/static/images/GS_PredVTru.png)

Molecules are colored by their similarity to compounds of the train set using Tanimoto similarity of their Morgan Fingerprints. Darker colors represent greater similarity of the molecule to the train set.

## Next Steps
We have covered briefly the use of OCE’s dataset wrapper, dataset splitting tools, model fitting and predicting, hyperparameter optimization, and visualization of datasets and models. This guide gives an overview of the main features of the library and serves as a good starting point for experimentation with your own datasets, but there is much more to OCE! You can refer to the example notebooks of the OCE library or the [documentation](https://docs.oloren.ai).