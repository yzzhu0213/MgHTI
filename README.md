# Multi-granular Herb-Target Interaction Prediction by Fusing Multi-level Data
This repository contains the code and data of MgHTI.

## Quick start
We provide an example script to run experiments on our dataset. Please run **main.py** to start MgHTI and predict herb-target interactions(HTIs)/ingredient-target interactions(ITIs).



## Code
- **main.py**: predict target-target interactions(HTIs) and ingredient-target interactions(ITIs), and evaluate the results with cross-validation.
- **model.py**: predict target-target interactions(HTIs) and ingredient-target interactions(ITIs).
- **utils.py**: data preprocess and data load.
- **args.py**: parameter settings for MgHTI.
- **evaluation.py**: evaluate the results.  
- **loss_function.py**: calculate the loss function.


## Data
- **herb.xlsx**: A list of herbs, which contains the IDs and names of 470 herbs, as well as their efficacy, property, and meridian. 
- **target.xlsx**: A list of targets，which contains the IDs and amino acid sequences of 1754 targets(protein targets).  
- **ingredient.xlsx**: A list of ingredients, which contains the IDs and SMILES representations of 881 ingredients.  
- **herb-target.xlsx**: 67429 known interactions between herbs and targets, linked by HerbID and TargetID.  
- **herb-ingredient.xlsx**: 6294 known interactions between herbs and ingredients, linked by HerbID and IngredientID.  
- **ingredient-target.xlsx**: 7501 known interactions between ingredients and targets, linked by IngredientID and TargetID.  
These data were collected from [**HIT(2.0)**](http://www.badd-cao.net:2345/) database, [**SymMap(v2)**](http://www.symmap.org/) database, [**KEGG**](https://www.genome.jp/kegg/) database and [**Chinese Pharmacopoeia(2020)**](https://ydz.chp.org.cn/#/main).


## Environmental requirements
- Python 3.10
- Pytorch 1.10.1
- CUDA 11.4
- Ubuntu 22.04
