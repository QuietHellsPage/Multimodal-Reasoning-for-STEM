# Main README of the project

**All needed steps were passed**:

1. Training code is in file `experiments.ipynb` - it evaluates zero-shot and one-shot models, then tunes 2 models - one with single dataset and another with 2 datasets.

2. Simple streamlit app is done, the way it works is shown in `report.md` and the app itelf lies in `app.py`

3. Checkpoints are saved (I hope they will not be downloading too long in case of cloning the repo)

4. Results and parameters are in `report.md` with all other information

**Important**: my computational resourses allowed me to use only about 1000 dataset samples for both fine-tuned models. Moreover,
when I tuned the model on 2 datasets, I had to reduce the amount of samples to 800. And I think that's the reason that model works a bit worse (`0.6454` CER vs `0.5741` CER on model with 1 dataset). However, I think that if fine-tuned on equal amount of samples, approach
with 2 datasets will be better.

**!!!** I ran all the experiments in the GoogleColab environment with TPU. That's why this notebook is not likely to run on GPU
