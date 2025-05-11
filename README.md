# ARES: Attention-based REward Shaping

Anonymized repository for the ARES paper.

## 📦 Requirements

- **Python version**: 3.10.12  
- **Dependencies**:  
  Install via pip:
  ```bash
  pip install -r requirements.txt
  ```

## 📁 Datasets

The datasets used in this paper can be downloaded [here](https://drive.google.com/file/d/13ue2aqBMZLj_6IxhZYW4Azz8id5dNN-G/view?usp=sharing).

---

## 🚀 Running MuJoCo Experiments

### 🔹 Generate Random Data

Generate 10 random datasets of 1000 timesteps each using 4 processes:

```bash
python gym_random_episode_gen.py Hopper-v4 1000 4
```

### 🔹 Generate TrainingExpert Data

Use a SAC agent to generate 1,000,000 timesteps of slowly improving expert behavior on `Hopper-v4`:

```bash
python generate_SAC_expert_output.py output.txt Hopper-v4 1000000 4
```

- Output will be saved to `output.txt`

### 🔹 Generate Shaped Rewards from Output

Train a transformer reward model for 10,000 epochs on the generated output:

```bash
python output_to_unmerged_rewards.py output.txt rewards.txt 10000 512 0.00001 8 11 3
```

- `512`: embedding size  
- `0.00001`: learning rate  
- `8`: internal (final-layer) embedding size  
- `11`: state dimensions  
- `3`: action dimensions (Hopper-v4)

### 🔹 Train SAC Agent with Shaped Rewards

Train a SAC agent using the shaped rewards:

```bash
python generate_SAC_inferred_output.py rewards.txt finaloutput.txt Hopper-v4 4 1000000 11 3 5 1 1 0
```

- `4`: number of vectorized environments  
- `1000000`: timesteps  
- `11/3`: state/action dimensions  
- `5 1 1 0`: distance hyperparameters  
- Output saved to `finaloutput.txt`

---
