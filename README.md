# Dynamical system modelling of somatic mutations accumulation 

This repository contains code and all of the calculations from the paper: [LINK to the preprint](https://www.biorxiv.org/content/10.1101/2025.11.23.689982v1)
## Description
![alt text](https://github.com/ComputationalAgingLab/somatic-mutations-simulation/blob/main/graphabstract.jpg "Graphical abstract")
This repository contains the code and calculations for dynamical systems models of somatic mutation accumulation in tissues.  
We have developed an incremental approach for the evaluation of somatic mutations impact on the aging process, in which multiple models, that are gradually becoming more complex, allow us to model individual tissue aging trajectory based on the several biological parameters. 

## Installation
### 1. Clone the repo
```
git clone https://github.com/ComputationalAgingLab/somatic-mutations-simulation.git
cd somatic-mutations-simulation
```

### 2. Create conda env
```
conda env create -f environment.yml
```

### 3. Activate
```
conda activate somatic-sim
```

### 4. Install the package
```
pip install -e .
```

### 5. OR if you want to use pyproject.toml
```
git clone https://github.com/ComputationalAgingLab/somatic-mutations-simulation.git
cd somatic-mutations-simulation
pip install -e .
```

## Usage

### Run brain (Model II)
```
python run_pipeline.py --organ brain
```

### Run liver with LPC (Model IIIB)
```
python run_pipeline.py --organ liver --organ_s LPC
```

### Also compute hazard rates
```
python run_pipeline.py --organ brain --compute_hazard
```

### You can also run multiple organs in one command
```
python run_pipeline.py --organ heart liver --n_mc 1000 --n_traj 1000
```

### Frech\'et - Hoeffding bounds calculation

Once the four organ hazards are computed, it is possible to calculate the organism survival bounds.

If the package is installed, the function is called as:

```
from utils import frechet_hoeffding
organism_survival = frechet_hoeffding(data_brain, 
                                      data_heart, 
                                      data_lungs,
                                      save_path)                                  
```

The function returns ```pd.Dataframe``` with bounds of $S(t)$. The argument ```save_path``` is optional. 

## Contact

For questions about the code, models, or their use in your own work, you may contact the corresponding author listed in pyproject.toml:

Vlad Fedotov – Vlad.Fedotov@skoltech.ru 

Or open an issue on the GitHub repository

## Citation
If you found this library useful in your research, please cite us with the following plain citation or bibtex.

*Efimov, E., Fedotov, V., Malaev, L., Khrameeva, E. E., & Kriukov, D. (2025). Somatic mutations impose an entropic upper bound on human lifespan. bioRxiv, 2025-11.*

```
@article {Efimov2025.11.23.689982,
	author = {Efimov, Evgeniy and Fedotov, Vlad and Malaev, Leonid and Khrameeva, Ekaterina E. and Kriukov, Dmitrii},
	title = {Somatic mutations impose an entropic upper bound on human lifespan},
	elocation-id = {2025.11.23.689982},
	year = {2025},
	doi = {10.1101/2025.11.23.689982},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
}
```
