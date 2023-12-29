In order to run the code, first is needed to install a virtual environment with the following command: 

```bash
conda env create -f environment.yml
```
Be sure to run the previous command inside the Work1 directory

Activate the environment with the command: 
```bash
conda activate CI
```

You can then open the `CI-lab2.ipynb` file and run all cells. The main part of the code is in the `src\ultis.py` file where we implemented a class to easily run all the experiments. The results of each experiment will be saved in the corresponding CSV file in the `results` folder, as well as the generated graph in each of the subfolders.