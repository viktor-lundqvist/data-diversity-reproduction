### Training
Setup the conda env using `environment.yml`
Create a config.yaml file or change the parameters in the existing one.
Run `python src/train.py --config <your config path>`.
This also automatically creates the relevant data.

### Validation graphs (code should be restructured)
Enter the run_ids and number of training steps in the second cell of `validation.ipynb`, then run all cells.

### Notes/TODOs
- The wandb project used is hardcoded in train.py
