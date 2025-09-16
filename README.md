# Installation of DPMM Base Environment

Open a terminal:
```bash
conda create -n dpmm python=3.7

conda activate dpmm

pip install dm_control mujoco-py==2.0.2.8 cython==0.29.33 protobuf==3.20.0 gym==0.20.0

pip install -r requirements/dev.txt
```

# Download and install DPMM dependencies:


Install Bayesian Nonparametric Models library (bnpy) under project folder

```bash
git clone https://github.com/bnpy/bnpy.git

cd bnpy && pip install -e . && cd ..
```

Other pip install
```bash
pip install seaborn
```

# File Structure

```
-dpmm_model

--b2d_train
---scenariox, ...
----scenariox-townx-routex-weatherx, ...
-----anno, ...

--b2d_val

--model
---my_dpmm_model (model code cluding init, fit, save, load method)
---b2d_traj_fit_dpmm_by_ability (seperate b2d dataset by ability, use dpmm to fit the 6-sec-traj one by one.)
---dpmm_classify_gauss_demo.py (toy example)

--tool

--dataset
---b2d_ability_dataset.py

--requirements

--bnp

--text_enco
---scen_skill_desc_list (a list of 44 scenarios, each element contains scenario name, relative driving skill (ability), text description and encoded description.)

```

# Instruction
Run dpmm_classify_gauss_demo.py to quickly understand the dpmm fitting process and use b2d_traj_fit_dpmm_by_ability.py to learn how to train dpmm on b2d dataset.