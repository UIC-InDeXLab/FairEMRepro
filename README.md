# Fair Entity Matching (Availability and Reproducibility for VLDB 2024)

## A fairness suite for auditing Entity Matching approaches
Companion repository for reproducing the results of the paper "Through the Fairness Lens: Experimental Analysis and Evaluation of Entity Matching".

### Publication(s) to cite:
[1] Nima Shahbazi, Nikola Danevski, Fatemeh Nargesian, Abolfazl Asudeh, and Divesh Srivastava. "Through the Fairness Lens: Experimental Analysis and Evaluation of Entity Matching." Proceedings of the VLDB Endowment 16, no. 11 (2023): 3279-3292.

[VLDB Publication] <a href="https://dl.acm.org/doi/abs/10.14778/3611479.3611525">https://dl.acm.org/doi/abs/10.14778/3611479.3611525</a> <be>

## Requirements:
- Python 3.8
- 300GB of storage (for training process)
- Cuda-supported machine with NVIDIA GPUs

## Instructions:
We tried our best to make the reproducibility process as simple as possible. Please be advised that to run the experiments, you need a Cuda-supported machine with NVIDIA GPUs. Please follow the three steps below:

 
### Step 1: Installation
- Clone the repo: ```git clone https://github.com/UIC-InDeXLab/FairEMRepro.git```
- Enter the project's main directory: ```cd FairEMRepro/```
- Create a virtual environment: ```python -m venv venv```
- Activate the virtual environment: ```source venv/bin/activate``` 
- Install required packages: ```pip install -r requirements.txt```
- To run Jupyter notebook in local machine: ```jupyter notebook```
- To run Jupyter notebbok on server without browser: ```jupyter notebook --no-browser```

#### :warning: Notice 1:
Due to the long running time of the matchers, we have provided the prediction results based on a run in the repository. If you want to use the existing predictions and directly move to running the analysis, you can skip step 2. Otherwise, run the ```bash remove_script.sh``` script and move to step 2.

### Step 2:  Generating Matching Results
- Make sure that you have docker properly installed with non-root user permissions, 
- Make sure you have an NVIDIA GPU available and docker has access to GPU. See [here](https://docs.docker.com/config/containers/resource_constraints/#gpu) for more information.
- Run the jupyter notebook ```train.ipynb``` to train all the matching models and create the predictions for all datasets.
- Please note that when the run is over, it is needed to enter your root password to change permissions to the current user in the notebook.

#### :warning: Notice 2:
Please be advised that depending on the matcher, dataset, and the number of epochs each training task could take between a few minutes to a few days. Running all tasks using each matcher with the default parameters (epoch=10) took us about a week to finish (with GNEM being the slowest due to the high Cuda memory requirements). That being said, we have provided the results of a full run (with 10 epochs) in the repository in case anyone needs to skip the tedious training step.

### Step 3: Analysis and Visualization of The Results
After step 2 is over, run the jupyter notebook ```experiments.ipynb```. The results for Single and Pairwise Fairness, sensitivity to the matching threshold heatmaps and tables using PPVP and TPRP measures can be observed in the notebook.

## Contact
Feel free to contact the authors or leave an issue in case of any complications. We will try to respond as soon as possible.

## License

This project is licensed under the MIT License &mdash; see the [LICENSE.md](LICENSE.md) file for details.

<p align="center"><img width="20%" src="https://www.cs.uic.edu/~indexlab/imgs/InDeXLab2.gif"></p>
