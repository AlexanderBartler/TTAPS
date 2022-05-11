# TTAPS: Test-Time Adaption by Aligning Prototypes using Self-Supervision
Official implementation of the TTAPS accepted at IJCNN 2022.
## Requirements

Simply run

```
pip install -r requirements.txt
```

to install all requirements.

## WandB
Metrics from runs are logged using [Weights and Biases](https://www.wandb.ai).
In order to perform runs yourself, [sign up](https://app.wandb.ai/login?signup=true) there for a free account, and run
```
wandb login
```
from the Python environment that you installed the requirements into, and enter the API key you can find on your [Settings page](https://wandb.ai/settings).  
After that, you are fully setup to start executing runs.

## Training

[Scripts for training](scripts/training) are provided for both CIFAR-10 and CIFAR-100, for each of the methods, with the hyperparameters thet were used to obtain the results presented in the thesis.  
However, before using these scripts, you might need/want to change some things to fit your specific needs:
- Name of the python environment
- Number of GPUs (don't forget to adapt batch size accordingly)
- --data_dir: the directory where the dataset is located, or should be downloaded to
- --wandb_log_dir: the directory you want wandb's logs to go
- --wandb_project: your desired project name

## Testing

Likewise, [Scripts for testing](scripts/testing) are also provided.  
Like for training, the parameters mentioned above should be adjusted to fit your needs. Note that multi-gpu execution is not supported for the evaluation.  
Also, there are additional parameters that need to be specified:

- --wandb_entity: the username that you signed up with at Weights and Biases
- --wandb_project: your wandb project
- --artifact_name: the id of the artifact that contains the trained model that should be evaluated. For a finished training run, this can be found under "Artifacts" -> "Output artifacts" in the WandB web interface. Note that the specified project name has to match with the project in which the training run was done.
- --artifact_dir: the directory to locally save downloaded artifacts in  
  
The example scripts `*_example.sh` perform the evaluation on one single corruption of the CIFAR-10-Corrupted/CIFAR-100-Corrupted dataset.  
It is however recommended to use a [WandB sweep](https://docs.wandb.ai/guides/sweeps) to manage and group together the evaluation on all of the corruptions.  
The `*_sweep.yaml` files can be used to initialize a sweep by running
```
wandb sweep *_sweep.yaml
```
WandB then distributes the evaulation on the different corruptions over the the different agents that register for that sweep.
An agent can be starting using [`sweep_agent.sh`](scripts/testing/sweep_agent.sh).
The username, project name, and sweep id have to be specified there beforehand.  

## Credits

We'd like to thanks 

https://www.pytorchlightning.ai/

https://lightning-bolts.readthedocs.io/en/latest/

and especially 

Florian Bender (https://github.com/f-bender)

for providing the basis of our work. 