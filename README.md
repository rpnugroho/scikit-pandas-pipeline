# Scikit pandas pipeline
Simple scikit-learn pipeline helper to return pandas dataframe.

## Install
``` cmd
pip install git+https://github.com/rpnugroho/scikit-pandas-pipeline.git#egg=skpdspipe
```
## How to use
Create pipeline for numerical data
``` python
numerical_pipeline = Pipeline([
    ('slctr', DFColumnSelector(['fare', 'mile'])),
    ('scale', DFScaler(scaler_type='standard'))
])
```

Join with another pipeline
``` python
final_union = DFFeatureUnion([
    ('num', numerical_pipeline),
    ('cat', categorical_pipeline)
])
```

## Credits
- [source 1](https://github.com/dpmcgonigle/bamboo-pipeline)
- [source 2](https://github.com/Kgoetsch/sklearn_pipeline_enhancements)
- [source 3](https://zablo.net/blog/post/pandas-dataframe-in-scikit-learn-feature-union/)