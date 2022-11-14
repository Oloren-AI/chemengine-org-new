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