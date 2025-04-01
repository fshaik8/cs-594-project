# CS-594-Project: Enhancing Large Language Model Efficiency through Reverse KL Distillation

This README provides instructions for setting up the environment and preprocessing data for MiniLLM.

## Environment Setup

Install the required packages by running the commands below:

```bash
# Core dependencies
pip3 install torch
pip3 install deepspeed
pip3 install accelerate
pip3 install datasets

# Transformers fork for MiniLLM
pip3 install git+https://github.com/t1101675/transformers@minillm

# Additional utilities
pip3 install numerize
pip3 install rouge-score
pip3 install torchtyping
pip3 install rich
pip3 install peft
```

---

## Data Preprocessing

### 1. Directory Structure

```
<PROJECT_ROOT>
├── data/dolly/               # Raw dataset files
├── checkpoints/gpt2-medium/  # Tokenizer files
└── scripts/gpt2/tools/       # Preprocessing scripts
```

### 2. Download Dataset

Download the raw dataset files using:

```bash
huggingface-cli download MiniLLM/dolly --repo-type dataset --local-dir data/dolly/
```

### 3. Download Model

Create the model directory and download the GPT-2 Medium model:

```bash
mkdir -p checkpoints/gpt2-medium
huggingface-cli download openai-community/gpt2-medium --local-dir checkpoints/gpt2-medium
```

### 4. Run Preprocessing

Run the preprocessing scripts to prepare your data:

```bash
bash scripts/gpt2/tools/process_data_dolly.sh <PATH_TO_PROJECT_ROOT>

# Example:
# bash scripts/gpt2/tools/process_data_dolly.sh /home/user/LMOps/minillm
```

### Output Structure

After processing, the data will be organized as follows:

```
processed_data/
├── dolly/prompt/  # Prompt-only training data
│   ├── train.bin
│   └── train.idx
└── dolly/full/    # Full sequences (prompt + response)
    ├── train.bin
    └── train.idx
```

