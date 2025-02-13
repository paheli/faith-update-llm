# Assessing the Reliability of LLMs in Faithfully Updating Text

This repository accompanies our research paper (Assessing the Reliability of LLMs in Faithfully Updating Text) on the FRUIT task – a systematic evaluation of how large language models integrate new evidence into existing documents while preserving factual consistency and coherence.

## Abstract

> Faithfully updating text with new information is a critical challenge in natural language processing. The FRUIT task (Faithfully Reflecting Updated Information in Text) evaluates models on their ability to integrate new evidence into existing documents while maintaining factual consistency and coherence. This paper systematically investigates GPT-4o and Llama-3-8B, exploring various prompting strategies, fine-tuning techniques, and evidence ranking methods. In addition to standard processing, we examine streaming evidence scenarios, where updates are applied incrementally. Our findings reveal that while LLMs can effectively incorporate new facts, they struggle with hallucinations, evidence selection, and coherence maintenance—especially when handling multiple updates or streaming input. Structured evidence presentation significantly improves performance, but iterative updates lead to degradation, highlighting challenges in maintaining consistency over successive edits. By providing a detailed analysis of existing methods, this study identifies key gaps in automatic text updating and outlines directions for improving model reliability in real-world applications.

## Table of Contents

- [Requirements](#requirements)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Dataset](#dataset)
- [Usage](#usage)
  - [Generating Predictions](#generating-predictions)
  - [Dataset Preparation](#dataset-preparation)
  - [Evaluation](#evaluation)
- [Configuration](#configuration)
- [License](#license)

## Requirements

This project requires **Python 3.9 or above**. Please ensure that you have the following packages installed:

- **vllm**
- **rouge_score**
- **matplotlib**
- **tigerscore**
- **prometheus**
- **axolotl**

You can install these packages via pip (or add them to your `requirements.txt` if you prefer):

```bash
pip install vllm rouge_score matplotlib tigerscore prometheus axolotl
```

## Repository Structure

```
.
├── base_prompts
│   ├── cot.py              # Implements chain-of-thought prompting
│   └── zero_shot.py        # Implements zero-shot prompting
├── dataset_prep
│   └── dataset.py          # Generates a .pkl file needed for predictions; change paths as required
├── evaluation
│   ├── evaluate_generated.py   # Evaluates prediction files and generates graphs/metrics
│   └── update_rouge.py         # Computes ROUGE scores for updated texts
├── evidence ordering
│   ├── factual_filter_sequence.py  # Orders evidence based on factual consistency
│   ├── filter_rank.py               # Ranks evidence candidates
│   ├── random_shuffline.py          # (For ablation) Randomizes evidence ordering
│   └── streaming_evidences.py       # Handles streaming evidence scenarios
├── few shot
│   ├── example_picking.py   # Picks few-shot examples for prompting
│   └── vllm_few_shot.py   # Implements few-shot prompting using vLLM
├── finetuning
│   └── finetuned.py         # Contains fine-tuning routines for LLMs
├── LICENSE
├── README.md
└── reflect_refine
    ├── prometheus_correct.py    # Self-correction module using Prometheus methods
    ├── prometheus_evaluate.py   # Evaluates corrections based on Prometheus scoring
    ├── self_correction.py       # Implements iterative self-correction
    ├── tiger_score_correct.py   # Corrects predictions based on Tiger score metrics
    └── tiger_score_evaluate.py  # Evaluates updated texts using Tiger score metrics
```

**Notes:**

- **Output Files:**  
  All modules—except those in **evaluation** and **dataset_prep**—output predictions as `.jsonl` files suitable for Llama-3-8B instruct inference.
  
- **Dataset Preparation:**  
  The `dataset.py` script in **dataset_prep** generates the necessary `.pkl` file to obtain all predictions.  
  The original dataset can be found in the [original fruit repository](https://github.com/Horea94/Fruit-Images-Dataset). 

## Getting Started

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/paheli/faith-update-llm.git
   cd faith-update-llm
   ```

2. **Install required packages:**  
   Make sure you have Python 3.9+ installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

   If you do not have a `requirements.txt`, please manually install the necessary packages:

   ```bash
   pip install vllm rouge_score matplotlib tigerscore prometheus axolotl
   ```

### Dataset

Download the dataset from the [original fruit repository](https://github.com/google-research/language/tree/master/language/fruit) and place it in your preferred location. **Remember:** Update the file paths in `dataset.py` and any other relevant scripts to point to your local dataset directory.

## Usage

### Generating Predictions

Run the modules (except those in **evaluation** and **dataset_prep**) to generate `.jsonl` prediction files. For example, to use the chain-of-thought prompting:

```bash
python base_prompts/cot.py
```

### Dataset Preparation

Generate the required dataset file by running:

```bash
python dataset_prep/dataset.py
```

This script outputs a `.pkl` file that is used to retrieve all predictions.

### Evaluation

After generating predictions, evaluate them by running:

```bash
python evaluation/evaluate_generated.py
```

This will produce graphs and various metrics (including ROUGE scores via `update_rouge.py`).

## Configuration

**Important:**  
Before running any scripts, please update the file paths and configuration variables in the code to match the locations of your dataset, model checkpoints, and output directories.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

