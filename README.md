# Data-Driven Regime Detection in the U.S. Housing Market:
This Project models U.S. housing market regimes using a Hidden Markov Model (HMM) on FHFA House Price Index data to identify latent phases such as expansion, neutral, and correction.

# Team 102 - Watermelon
Members (A to Z):
- Dennis Sun
- Nick Pham
- Noe Gonzalez
- Wenrong Zheng

# Prjoect files structure:
```
cse150-final-hpi-hmm/
├─ data/                        # using cogs108 data files style
│   └─ 00-raw/         
|   |   └─...                   # original FHFA data
|   └─ 01-interim/         
|   |   └─...                   # processed temporary files/data 
|   └─ 02-clean/         
|       └─...                   # clean dataframe .csv 
├─ figures/
│   └─ ...                      # saved plots go here
├─ results/
│   └─ ...                      # any saved params / logs
├─ hmm_utils.py                 # core HMM algorithms (mini-project style)
├─ 01_final_pipeline.ipynb      # main notebook for this project
├─ 02_analysis.ipynb            # analysis the training data
└─ README.md
```

# Notebook files Note: 
01_final_pipeline.ipynb focuses on building a clean end-to-end pipeline (data loading, HMM training for different K, basic plots).

02_analysis.ipynb builds on top of that pipeline and focuses on evaluation and interpretation: regime stability across seeds and samples, macroeconomic alignment, and simple ROC-style analysis for downturn detection.