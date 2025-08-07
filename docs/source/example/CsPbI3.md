# CsPbI3

## Preparation Work
:::{note}
First, we create a working directory called CsPbI3. All subsequent operations are assumed to be in this directory by default.
:::

Here, we take the cubic phase of $CsPbI_3$ as an example to introduce the use of the entire automated training framework.
We download the .cif file from [Materials Project](https://next-gen.materialsproject.org/materials/mp-1069538?chemsys=Cs-Pb-I).
Then we make the supercell of [3,3,2] in VESTA as the  initial structure for subsequent training, with the file name CsPbI3.vasp.
The supercell structure is as follows:
```text
Cs1 Pb1 I3
1.0
       18.8254261017         0.0000000000         0.0000000000
        0.0000000000        18.8254261017         0.0000000000
        0.0000000000         0.0000000000        12.5502843857
   Cs   Pb    I
   18   18   54
Direct
     0.166666672         0.166666672         0.250000000
     0.166666672         0.166666672         0.750000000
     0.166666672         0.500000000         0.250000000
     0.166666672         0.500000000         0.750000000
     0.166666672         0.833333313         0.250000000
     0.166666672         0.833333313         0.750000000
     0.500000000         0.166666672         0.250000000
     0.500000000         0.166666672         0.750000000
     0.500000000         0.500000000         0.250000000
     0.500000000         0.500000000         0.750000000
     0.500000000         0.833333313         0.250000000
     0.500000000         0.833333313         0.750000000
     0.833333313         0.166666672         0.250000000
     0.833333313         0.166666672         0.750000000
     0.833333313         0.500000000         0.250000000
     0.833333313         0.500000000         0.750000000
     0.833333313         0.833333313         0.250000000
     0.833333313         0.833333313         0.750000000
     0.000000000         0.000000000         0.000000000
     0.000000000         0.000000000         0.500000000
     0.000000000         0.333333343         0.000000000
     0.000000000         0.333333343         0.500000000
     0.000000000         0.666666687         0.000000000
     0.000000000         0.666666687         0.500000000
     0.333333343         0.000000000         0.000000000
     0.333333343         0.000000000         0.500000000
     0.333333343         0.333333343         0.000000000
     0.333333343         0.333333343         0.500000000
     0.333333343         0.666666687         0.000000000
     0.333333343         0.666666687         0.500000000
     0.666666687         0.000000000         0.000000000
     0.666666687         0.000000000         0.500000000
     0.666666687         0.333333343         0.000000000
     0.666666687         0.333333343         0.500000000
     0.666666687         0.666666687         0.000000000
     0.666666687         0.666666687         0.500000000
     0.000000000         0.000000000         0.250000000
     0.000000000         0.000000000         0.750000000
     0.000000000         0.333333343         0.250000000
     0.000000000         0.333333343         0.750000000
     0.000000000         0.666666687         0.250000000
     0.000000000         0.666666687         0.750000000
     0.333333343         0.000000000         0.250000000
     0.333333343         0.000000000         0.750000000
     0.333333343         0.333333343         0.250000000
     0.333333343         0.333333343         0.750000000
     0.333333343         0.666666687         0.250000000
     0.333333343         0.666666687         0.750000000
     0.666666687         0.000000000         0.250000000
     0.666666687         0.000000000         0.750000000
     0.666666687         0.333333343         0.250000000
     0.666666687         0.333333343         0.750000000
     0.666666687         0.666666687         0.250000000
     0.666666687         0.666666687         0.750000000
     0.000000000         0.166666672         0.000000000
     0.000000000         0.166666672         0.500000000
     0.000000000         0.500000000         0.000000000
     0.000000000         0.500000000         0.500000000
     0.000000000         0.833333313         0.000000000
     0.000000000         0.833333313         0.500000000
     0.333333343         0.166666672         0.000000000
     0.333333343         0.166666672         0.500000000
     0.333333343         0.500000000         0.000000000
     0.333333343         0.500000000         0.500000000
     0.333333343         0.833333313         0.000000000
     0.333333343         0.833333313         0.500000000
     0.666666687         0.166666672         0.000000000
     0.666666687         0.166666672         0.500000000
     0.666666687         0.500000000         0.000000000
     0.666666687         0.500000000         0.500000000
     0.666666687         0.833333313         0.000000000
     0.666666687         0.833333313         0.500000000
     0.166666672         0.000000000         0.000000000
     0.166666672         0.000000000         0.500000000
     0.166666672         0.333333343         0.000000000
     0.166666672         0.333333343         0.500000000
     0.166666672         0.666666687         0.000000000
     0.166666672         0.666666687         0.500000000
     0.500000000         0.000000000         0.000000000
     0.500000000         0.000000000         0.500000000
     0.500000000         0.333333343         0.000000000
     0.500000000         0.333333343         0.500000000
     0.500000000         0.666666687         0.000000000
     0.500000000         0.666666687         0.500000000
     0.833333313         0.000000000         0.000000000
     0.833333313         0.000000000         0.500000000
     0.833333313         0.333333343         0.000000000
     0.833333313         0.333333343         0.500000000
     0.833333313         0.666666687         0.000000000
     0.833333313         0.666666687         0.500000000


```

## Generate Perturbation Training Set
We first generate 10000 perturbation training sets with 3% cell deformation and 0.2Å atomic perturbation.
:::{tip}
Here, we choose to add -f parameter to avoid generating non-physical structures from perturbations, and it was filterd based on bond length.
:::
```shell
NepTrain perturb CsPbI3.vasp -n 10000 -c 0.03 -d 0.2 -f
<!-- Output is as follows: -->
Current structure:Cs18Pb18I54 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
```
The script will output perturb.xyz. We then use select to extract 100 structures from the 10000 structures as the initial training set.
:::{note}
- You may adjust the -d parameter according to the output of selected structure, or directly use -d 0, but the selection may not be optimal
- If there is no NEP potential, SOAP descriptors will be used by default. Considering the spacing of SOAP descriptors is relatively large, so you should adjust -d parameter. 
:::
```shell
NepTrain select perturb.xyz -max 100 -d 1
<!-- Output is as follows: -->
[2024-12-28 18:15:21.436949] --  Reading trajectory perturb.xyz
[2024-12-28 18:15:28.985300] --  The file base does not exist.
[2024-12-28 18:15:28.986175] --  The file ./nep.txt does not exist.
[2024-12-28 18:15:28.986671] --  An invalid path for nep.txt was provided, using SOAP descriptors instead.
[2024-12-28 18:15:29.454689] --  Start generating structure descriptor, please wait
[2024-12-28 18:16:18.169325] --  Starting to select points, please wait...
[2024-12-28 18:16:24.492264] --  Obtained 100 structures.
[2024-12-28 18:16:25.800733] --  The point selection distribution chart is saved to ./selected.png.
[2024-12-28 18:16:25.801237] --  The selected structures are saved to ./selected.xyz.
```
Then run the following command to finish the preparation!
```shell
mv selected.xyz train.xyz
<!-- This deletion is not necessary -->
rm selected_perturb.xyz.xyz perturb.xyz selected.png
```
## Initialize the Task
:::{tip}
All template files can be saved by yourself. And all template files can be saved independently. If a corresponding template file exists in the designated directory (the default being the working directory), the creation of a new template file will be skipped, and the existing template file will be used instead.
:::
First, we use the following command to initialize
```shell
NepTrain init slurm
<!-- Output is as follows: -->
[2024-12-29 10:08:27.298365] --  For existing files, we choose to skip; if you need to forcibly generate and overwrite, please use -f or --force.
[2024-12-29 10:08:27.302688] --  Create the directory ./structure, please place the expanded structures that need to run MD into this folder!
[2024-12-29 10:08:27.320868] --  You need to check and modify the vasp_job and vasp.cpu_core in the job.yaml file.
[2024-12-29 10:08:27.321411] --  You also need to check and modify the settings for GPUMD active learning in job.yaml!
[2024-12-29 10:08:27.336706] --  Detected that there is no train.xyz in the current directory; please check the directory structure!
[2024-12-29 10:08:27.337630] --  If there is a training set but the filename is not train.xyz, please unify the job.yaml.
[2024-12-29 10:08:27.345471] --  Create run.in; you can modify the ensemble settings! Temperature and time will be modified by the program!
[2024-12-29 10:08:27.351823] --  Initialization is complete. After checking the files, you can run `NepTrain train job.yaml` to proceed.
```
Move `CsPbI3.vasp` to the `structure` directory created by the program:
```shell
mv CsPbI3.vasp structure
``` 
Let's take a look at the current directory:
```text
├── job.yaml
├── run.in
├── structure/
│   └── CsPbI3.vasp
└── train.xyz
```
### Modify MD Template [Optional]
The default run.in  is for npt, and generally, only the `ensemble` needs to be modified, the program will automatically replace the temperature.
We are not making any changes here.
### Modify NEP Training Parameters
By default, the program does not require the `nep.in` file to be provided. It will automatically generate the simplest `nep.in` based on the training set.
If you need to modify hyperparameters, simply place your own `nep.in` file in this directory(the default working directory or the directory specified for `nep.in` in the `.yaml` file). 
### Modify Task Details
In `job.yaml` we have explained each parameter through comments as much as possible. And most parameters do not need to be adjusted.
We will not explain each parameter in detail here, only showing how to modify according to your own system.

#### VASP Calculation Details
:::{tip}
`cpu_core` should match the number of CPU cores requested for VASP jobs on your cluster.
:::
To accelerate single-point energy calculations, we set the number of tasks through `vasp_job`, and the program will divide the tasks according to this number.
This depends on your own computational resources.

In addition, all calculation details are specified through INCAR, and you can create your own INCAR and place it in this directory(the default working directory or the directory specified for `nep.in` in the `.yaml` file).
We also need to modify the k-points selection. Here we choose the `kspacing` form. After modification, it is as follows:
```yaml
vasp:

  cpu_core: 64
  kpoints_use_gamma: true  #ASE defaults to using M-point k-mesh, but here we default to using the gamma-centered grid; this can be set to false.

  incar_path: ./INCAR

  use_k_stype: kspacing
  #--ka
  kpoints:
    - 20 #a
    - 20 #b
    - 20 #c
  kspacing: 0.1
```
:::{tip}
Typically, a kspacing value of 0.2 is sufficient for common calculations. However, for metals, the k-point density needs to be increased. Usually, a kspacing value of 0.1 is much sufficient for metals.
:::
#### Modify Active Learning Iteration Details
The current program's number of iterations is determined by the time of active learning.
We choose to enable bond length filtering.
We set the MD temperature range to 0-300k, with iteration times of 10ps, 100ps, 500ps, and 1000ps.
:::{tip}
`step_times`does not require a progressive relationship, for example, it can be 10 100 100 500 500 this kind of repetitive detection.
If no structures are extracted in the second 100ps, in addition to the extra time for one md, there will be no repeated training.
:::
After modification, it is as follows:
```yaml
gpumd:
#Time for iterative progressive learning in units of picoseconds.
#The first active learning is at 10ps, the second at 100ps, with a total of four active learning sessions.
  step_times:
    - 10
    - 100
    - 500
    - 1000
#Each time active learning is performed, all structures in model_path will undergo molecular dynamics (MD) simulations at the following temperatures, followed by sampling.
  temperature_every_step:
    - 50
    - 100
    - 150
    - 200
    - 250
    - 300
  model_path: ./structure
  run_in_path: ./run.in
  filter_by_bonds: true  #Enable bond length detection, and determine structures with bond lengths below 60% of the equilibrium model bond lengths as non-physical structures.
```
An additional note on `yaml` syntax.
   ```yaml
    gpumd:
    #Time for iterative progressive learning in units of picoseconds.
    #The first active learning is at 10ps, the second at 100ps, with a total of four active learning sessions.
      step_times:
        - 10
        - 100
        - 500
        - 1000
   ```
   is equivalent to
   ```yaml
    gpumd:
    #Time for iterative progressive learning in units of picoseconds.
    #The first active learning is at 10ps, the second at 100ps, with a total of four active learning sessions.
      step_times: [10, 100, 500, 1000]
   ```
#### Modify Sampling Details
We set a maximum of 80 structures to be selected each time. The minimum distance is 0.01
```yaml
select:
  #After completing this round of MD, a maximum of max_selected structures will be selected from all trajectories.
  max_selected: 80
  min_distance: 0.01   #Hyperparameters for farthest point sampling
```
The complete modified `job.yaml` is as follows
```yaml
version: 1.4.3
queue: slurm #Select the queuing system, divided into Slurm and local.
vasp_job: 10 #The number of tasks submitted when calculating single-point energy with VASP.
#All task submission root directories
work_path: ./cache  #Root directory for all task submissions.
current_job: vasp  #If the current_job has three states: nep, gpumd, vasp, and if train.xyz has not been calculated, set it to vasp; otherwise, directly set it to use nep to train the potential function, or use gpumd.
generation: 1  #Marking resume tasks.
init_train_xyz: ./train.xyz  #Initial training set; if not calculated, set current_job to vasp.
init_nep_txt: ./nep.txt  #If current_job is set to gpumd, a potential function must be provided; otherwise, it can be ignored.
nep:
  #Does it support restarting? If true, the potential function for the next step will continue from this step for nep_restart_step steps.
  #The program will automatically set lambda_1 to 0.
  #If false, retrain from scratch every time.
  nep_restart: true
  nep_restart_step: 20000
  #Optional; if you need to modify the number of steps, simply provide a file in the current path.
  #If there is no such file, the number of steps will be automatically generated based on the training set.
  nep_in_path: ./nep.in
  #Optional
  test_xyz_path: ./test.xyz
vasp:

  cpu_core: 64
  kpoints_use_gamma: true  #ASE defaults to using M-point k-mesh, but here we default to using the gamma-centered grid; this can be set to false.

  incar_path: ./INCAR

  use_k_stype: kspacing
  #--ka
  kpoints:
    - 20 #a
    - 20 #b
    - 20 #c
  kspacing: 0.1
gpumd:
#Time for iterative progressive learning in units of picoseconds.
#The first active learning is at 10ps, the second at 100ps, with a total of four active learning sessions.
  step_times:
    - 10
    - 100
    - 500
    - 1000
#Each time active learning is performed, all structures in model_path will undergo molecular dynamics (MD) simulations at the following temperatures, followed by sampling.
  temperature_every_step:
    - 50
    - 100
    - 150
    - 200
    - 250
    - 300
  model_path: ./structure
  run_in_path: ./run.in
  filter_by_bonds: true  #Enable bond length detection, and determine structures with bond lengths below 60% of the equilibrium model bond lengths as non-physical structures.

select:
  #After completing this round of MD, a maximum of max_selected structures will be selected from all trajectories.
  max_selected: 80
  min_distance: 0.01   #Hyperparameters for farthest point sampling

limit:
  force: 20  #Limit the force of the structure to between -force and force
```
## Start Training
Execute the following command in the terminal of the login node
```shell
NepTrain train job.yaml

```
Run in the background
```shell
nohup NepTrain train job.yaml &

```
