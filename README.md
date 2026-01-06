<div align="center">

# LaCoMSA: Language-Consistency Multilingual Self-Alignment
</div>

Khanh-Tung Tran, Barry O'Sullivan, Hoang D. Nguyen

*Accepted to EACL 2026 (Main Conference)*

## Overview
> Large Language Models (LLMs) have achieved impressive performance yet remain inconsistent across languages, often defaulting to high-resource outputs such as English. Existing multilingual alignment methods mitigate these issues through preference optimization but rely on external supervision, such as translation systems or English-biased signal. We propose Multilingual Self-Alignment (MSA), a preference optimization framework that leverages an LLM’s own latent representations as intrinsic supervision signals, rewarding lower-resource language outputs based on their alignment with high-resource (English) counterparts in the ``semantic hub''. We further introduce Language-Consistency MSA (LaCoMSA), which augments MSA with a final-layer language-consistency factor to prevent off-target generation. Integrated with Direct Preference Optimization, LaCoMSA improves a Llama 3 8B-based model multilingual win rates by up to 6.8% absolute (55.0% relatively) on X-AlpacaEval and achieves consistent gains across benchmarks and models. Our findings demonstrate that LaCoMSA can serve as an effective and scalable mechanism, opening a new venue toward multilingual self-alignment.

This repository contains the cleaned implementation for the LaCoMSA paper.

## Repository Structure
```
Alignment/        # DPO training scripts
Preprocess/       # Data & reward generation scripts
Data/
requirement.txt
```

## Quick Start

1. Environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Prepare preference data:
```bash
cd Preprocess
bash preprocess.sh # more details available in the bash script
```

3. Train with. DPO:
```bash
cd Alignment
bash dpo.sh example.json
```

## Citation
TBU
