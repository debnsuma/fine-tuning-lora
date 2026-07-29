# Foundations: Quantization, LoRA, and QLoRA from Scratch

A from-scratch walkthrough of the two core ideas that make modern LLM fine-tuning cheap: **quantization**, which shrinks a model's memory footprint by storing its weights in fewer bits, and **LoRA (Low-Rank Adaptation)**, which shrinks the number of parameters you actually need to train. 

We will first implement from scratch in plain PyTorch, then hand off to the standard `transformers`, `peft`, and `bitsandbytes` stack to run a real **QLoRA (Quantized Low-Rank Adaptation)** fine tune, so the library calls read as familiar rather than magic.

QLoRA, in brief, is LoRA applied on top of a quantized base model, which is exactly what [Crusoe Serverless Fine-Tuning](../02-serverless-fine-tuning-pii-redaction/) runs for you at scale. Understanding what is happening here is what lets you trust that managed service instead of treating it as a black box.

| Notebook | What it covers |
|---|---|
| [01_quantization.ipynb](01_quantization.ipynb) | Why numeric precision matters, quantization and dequantization implemented by hand, why per-channel granularity beats per-tensor on real transformer weights, mixed precision versus quantization, the NF4 format, and loading a real Hugging Face model in 4 bit with bitsandbytes |
| [02_lora_and_qlora.ipynb](02_lora_and_qlora.ipynb) | Why full fine tuning is expensive, a LoRA linear layer implemented from scratch in PyTorch and applied to a real pretrained model, confirming the adapter is genuinely low rank, then combining it with quantization using `transformers`, `peft`, and `bitsandbytes` to fine tune a real model end to end and merge the result |

## Quickstart

These notebooks share one virtual environment with the rest of this repository. See the root [README](../README.md) for creating a VM (with GPU) on Crusoe Cloud and setting up the shared environment with `uv`. 

Once that is done:

```bash
cd 01-into-lora-fine-tuning
../.venv/bin/jupyter lab 01_quantization.ipynb
```
Follow the notebooks: 

- `01_quantization.ipynb` needs an NVIDIA GPU throughout (bitsandbytes' kernels are CUDA only). 
- `02_lora_and_qlora.ipynb` starts on plain PyTorch, no GPU required, and only needs one once it gets to the QLoRA section partway through.

## Further reading

- [Hugging Face quantization overview](https://huggingface.co/docs/transformers/main/en/quantization/overview)
- [bitsandbytes documentation](https://huggingface.co/docs/bitsandbytes/main/en/index)
- [PyTorch automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/main/en/index)
- [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- [Fine-Tuning LLMs](https://leanpub.com/finetuning)

