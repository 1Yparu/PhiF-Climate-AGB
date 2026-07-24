# PhiF-Climate-AGB
Code and data analysis for integrating chlorophyll fluorescence quantum yield and climatic variables to estimate winter wheat aboveground biomass.

# Lightweight trained-model inference

This repository contains only trained model artifacts and a unified inference interface for:

- XGBoost
- LightGBM
- NODE
- GANDALF

## Correct NODE/GANDALF directory structure

A directory such as `PhiF+GDD+CP` is already the complete PyTorch Tabular saved model:

```text
models/
├── NODE/
│   ├── PhiF+GDD+CP/
│   │   ├── callbacks.sav
│   │   ├── config.yml
│   │   ├── custom_params.sav
│   │   ├── datamodule.sav
│   │   └── model.ckpt
│   ├── X_sel+GDD+CP/
│   │   └── ...
│   └── PhiF+X+GDD+CP/
│       └── ...
└── GANDALF/
    ├── PhiF+GDD+CP/
    │   └── ...
    └── ...
```


The files have the following roles:

- `config.yml`: saved model and trainer configuration, including the architecture.
- `model.ckpt`: trained network weights.
- `datamodule.sav`: saved data configuration and input schema.
- `custom_params.sav` and `callbacks.sav`: supporting PyTorch Tabular state.


```
