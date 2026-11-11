# Backdoor Attacks in Federated Language Models

This repository contains the federated text classification implementation used to evaluate backdoor attacks and BERT-based sample filtering.

We ran our experiments on a Linux server with NVIDIA GPUs. The supplied `requirements.txt` specifies the following environment versions:

```text
cuda==12.0
python==3.12.3
pytorch==2.5.1
transformers==5.0.0
peft==0.18.1
bitsandbytes==0.49.1
accelerate==1.12.0
numpy==2.2.6
pandas==2.3.3
scikit-learn==1.7.2
```

### Code Reference

The main federated training and filtering pipeline is implemented in:

```text
federated_text_bert_detector_defense.py
```

Supported datasets:

```text
SST-2
HSOL
AG
```

Supported attacks:

```text
BadNet
AddSent
HiddenKiller
StyleBkd
```

Supported model families:

```text
bert
llama
mistral
qwen
```

LoRA is used by default for `llama`, `mistral`, and `qwen`, but not for `bert`. The script also supports `bert_filter`, `ticu`, and `bert_ticu` defense modes.

### Instructions

1. **Create and Activate an Environment**

   ```bash
   conda create -n fed_text python=3.12.3 -y
   conda activate fed_text
   ```

2. **Install Dependencies**

   Set up a compatible CUDA/PyTorch installation for your GPU, then install the Python packages. **Note:** The provided `requirements.txt` is an environment-version reference, not a directly pip-installable file: `cuda`, `python`, and `pytorch` require separate handling (`torch` is the pip package name).

   ```bash
   pip install transformers==5.0.0 peft==0.18.1 bitsandbytes==0.49.1 \
       accelerate==1.12.0 numpy==2.2.6 pandas==2.3.3 scikit-learn==1.7.2
   ```

3. **Prepare the Dataset**

   Organize the clean and poisoned CSV files as follows:

   ```text
   <helper_dir>/Data_Poisoning/<Attack>/<Dataset>/
   ├── train_poisoned.csv
   ├── train_clean.csv
   ├── test_clean.csv
   ├── test_poisoned_all.csv
   └── test_poisoned_part.csv
   ```

   Each CSV must have `text` and `label` columns (the script also accepts `sentence`, `comment_text`, or `content` in place of `text`). `train_poisoned.csv` must additionally identify poisoned samples using a `poisoned` column with `1` for poisoned rows.

4. **Run a BERT Attack Baseline and Filtering Defense**

   ```bash
   python federated_text_bert_detector_defense.py \
     --helper_dir ./helper \
     --dataset SST-2 \
     --attack BadNet \
     --model-family bert \
     --model-name bert-base-uncased \
     --rounds 5 --clients 5 --local-epochs 1 \
     --poison-rate 1.0 \
     --run both --defense bert_filter \
     --detector-threshold 0.5 --detector-rho-max 0.2 \
     --out_dir ./fed_text_bert_detector_outputs
   ```

5. **Run a LoRA Model with BERT Filtering**

   Example using Qwen:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python federated_text_bert_detector_defense.py \
     --helper_dir ./helper \
     --dataset HSOL \
     --attack HiddenKiller \
     --model-family qwen \
     --model-name Qwen/Qwen2.5-0.5B \
     --use-lora --load-in-4bit \
     --rounds 5 --clients 5 --local-epochs 1 \
     --poison-rate 1.0 \
     --run both --defense bert_filter \
     --out_dir ./fed_text_bert_detector_outputs
   ```

6. **Run Other Configurations**

   Change `--dataset`, `--attack`, `--model-family`, and `--model-name` as needed. Use `--run attack` for an undefended baseline, `--run defense` for defense only, or `--run both` to compare both. The default defense is `ticu`, so **specify `--defense bert_filter`** when reproducing the BERT filtering experiments.

### Outputs

The program prints clean accuracy, poisoned-test accuracy, label-flip rate, and detector removal statistics. Checkpoints are saved under `--out_dir` with names following:

```text
fed_text_<dataset>_<attack>_<model_family>_<model_slug>_<run_name>.pt
```

### Notes

- For the decoder-based models, place `lora_llama.py`, `lora_mistral.py`, and `lora_qwen.py` alongside the main script.
- LLaMA models may require approved Hugging Face model access and authentication.
- Quantized decoder training requires CUDA and LoRA.
- The BERT detector requires labeled clean and poisoned examples for training.
