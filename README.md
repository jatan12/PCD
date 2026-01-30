## Pareto-Conditioned Diffusion models

### Installation
This code base builds on top off [Offline-moo](https://github.com/lamba-bbo/offline-moo), 
and thus one needs to install it first. To make this process easier, the exact version
of offline-moo used in our case is included in `offline_moo/data`.

Begin the process by installing requirements for offline-moo 

```bash
cd offline_moo
conda env create -f environment.yml
conda activate off-moo
conda install gxx_linux-64 gcc_linux-64
conda install --channel=conda-forge libxcrypt

# Install requirements  from pip
conda install -r requirements.txt

conda install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

pip install scipy==1.10.1
pip install scikit-learn==0.21.3
pip install --upgrade pandas
pip install --upgrade kiwisolver

# Dependencies specific to our code  base
pip install gin-config==0.5.0
pip install einops==0.8.0
pip install torchdiffeq==0.2.5
pip install pygmo==2.19.5
pip install accelerate==1.0.1
pip install wandb==0.19.6
```

This should cover the basic installation. However, some of the tasks, such as scientific design
and MORL tasks require additional setup. For these we refer the reader to the official instructions
from [Offline-moo](https://github.com/lamda-bbo/offline-moo) 

- [Mujoco](https://github.com/lamda-bbo/offline-moo/tree/main#mujoco)
- [FoldX](https://github.com/lamda-bbo/offline-moo/tree/main#foldx)
- [MONAS](https://github.com/lamda-bbo/offline-moo/tree/main#evoxbench)


> [!CAUTION]
> Due to the complicated nature of the dependencies required by the different tasks in offline-moo,
> we found that it is easier to create separate environments for each subtask that requires
> additional software. Your mileage may vary!

After installing the required dependencies, download the offline data from [google-drive](https://drive.google.com/drive/folders/1SvU-p4Q5KAjPlHrDJ0VGiU2Te_v9g3rT) and place them in `offline_moo/data`.
(Note: experiments shown in the paper utilized the data_fix_250508 version of the dataset.)


### Reproducing result
Below a few examples from the paper are shown:

Train & evaluate PCDiffusion in ZDT2. 
```bash
python train.py --task_name zdt2 --seed 1000 --domain synthetic --sampling-method 'reference-direction' --sampling-guidance-scale 2.5 --reweight-loss --experiment_name "reweight-ref-dir" --save-dir path/to/your_dir
```


Use data pruning instead of dataset reweighing in MORL
```bash
python train.py --task_name mo_hopper_v2 --seed 1000 --domain morl --sampling-method 'reference-direction' --sampling-guidance-scale 2.5 --data_pruning --experiment_name "pruning-ref-dir" --save-dir path/to/your_dir
```

Use simple condition mechanism without any data-processing (Ideal + N/A from table 2) in MONAS
```bash
python train.py --task_name c10mop2 --seed 2000 --domain monas --sampling-method 'uniform-ideal' --sampling-guidance-scale 2.5 --data_pruning --experiment_name "pruning-ref-dir" --save-dir path/to/your_dir
```

The results from the paper are performed for all tasks & with seeds 1000, 2000, 3000, 4000, 5000
