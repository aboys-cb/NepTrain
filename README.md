 

<h4 align="center">

 
[![PyPI Downloads](https://img.shields.io/pypi/dm/NepTrain?logo=pypi&logoColor=white&color=blue&label=PyPI)](https://pypi.org/project/NepTrain)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
 
</h4>

 
[pull request]: https://github.com/aboys-cb/NepTrain/pulls
[github issue]: https://github.com/aboys-cb/NepTrain/issues
[github discussion]: https://github.com/aboys-cb/NepTrain/discussions

 
## Installation

You can install it via `pip`:

[pypi]: https://pypi.org/project/NepTrain

```sh
pip install NepTrain
```

If you want to use the latest changes from the main branch, you can install it directly from GitHub:

```sh
pip install -U git+https://github.com/aboys-cb/NepTrain
```
### Community Support

- Join the community chat (https://qm.qq.com/q/wPDQYHMhyg)
- Raise issues and engage in discussions via GitHub issues

## Software Architecture

It is recommended to use Python 3.10 or higher. Older versions might cause type errors.
We also recommend using GPUMD version 3.9.5 or higher.

 
## Usage

Modify the `vim ~/.NepTrain` file to change the pseudopotential file path.
If this file doesn't exist, simply run `NepTrain init` once to generate it.


 
### Creating Training Set (Optional)
Generate a perturbation training set for structures or structure files.

For example, apply a 0.03 lattice distortion and 0.1 atomic perturbation:
```sh
NepTrain perturb ./structure/Cs16Ag8Bi8I48.vasp --num 2000 --cell 0.03 -d 0.1  
NepTrain select perturb.xyz -max 100  
```

### 1. Initialization
First, initialize NepTrain. This will create a submission script in the current directory:
```sh
NepTrain init slurm
```

### 3. Submit Job
After modifying the submission script and job configuration, you can submit the job by running the following command on a login node:
```sh
NepTrain train job.yaml
```
For running the job in the background, use `nohup`
```sh
nohup NepTrain train job.yaml &
```
If the job is interrupted, there will be a  `restart.yaml` file in the directory. To resume the job, run:
```sh
NepTrain train restart.yaml
```
