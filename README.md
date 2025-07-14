# LLM Attacks


This is the official repository for "[Augmented Adversarial Trigger Learning](https://arxiv.org/abs/2503.12339)" by [Zhe Wang](https://scholar.google.com/citations?user=fqNkQjgAAAAJ&hl=en) and [Yanjun Qi](https://qiyanjun.github.io/Homepage/).


## Table of Contents

- [Installation](#installation)
- [Models](#models)
- [Running Attacks](#running-attacks)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Ack](#Ack)

## Installation

The project requires Python 3.8+ and PyTorch. Install the dependencies by running:

```bash
pip install -r requirements.txt
```

## Models

The implementation currently supports:
- Llama-2-7b-chat (from HuggingFace hub: meta-llama/Llama-2-7b-chat-hf)
- Vicuna-7b-v1.5 (from HuggingFace hub: lmsys/vicuna-7b-v1.5)

The models will be automatically downloaded when first used. Make sure you have accepted the terms of use for the Llama-2 model on HuggingFace.

## Running Attacks

You can run attacks using the main script with various parameters:

```bash
python main.py \
    --llm [llama2/vicuna] \
    --q_index [behavior_index] \
    --elicit [elicitation_coefficient] \
    --softmax [temperature] \
    --length [suffix_length] \
    --path [output_path]
```

Parameters:
- `llm`: Choose between 'llama2' (default) or 'vicuna'
- `q_index`: Index of the behavior to attack from harmful_behaviors.csv
- `elicit`: Elicitation coefficient for optimization
- `softmax`: Temperature parameter for softmax
- `length`: Length of the adversarial suffix
- `path`: Path to save the attack results

The script will:
1. Load the specified model and tokenizer
2. Initialize the attack with the given parameters
3. Run the optimization process
4. Save the results including loss values and generated responses

You can run attacks on Llama2 for different harmful categories:

```bash
python run_category.py \
    --category [bomb/misinformation/hacking/theft/suicide]
```

## Reproducibility

A note for hardware: all experiments we run use one or multiple NVIDIA A100 GPUs, which have 40G memory per chip. 


## Citation
If you find this useful in your research, please consider citing:

```
@inproceedings{wang-qi-2025-augmented,
    title = "Augmented Adversarial Trigger Learning",
    author = "Wang, Zhe  and
      Qi, Yanjun",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.394/",
    doi = "10.18653/v1/2025.findings-naacl.394",
    pages = "7068--7100",
    ISBN = "979-8-89176-195-7",
}
``` 

## Ack
Our code is based on the GCG's repo (https://github.com/llm-attacks/llm-attacks) and PAIR's repo (https://github.com/patrickrchao/JailbreakingLLMs/tree/main)