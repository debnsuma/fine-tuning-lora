# Fine-Tuning with LoRA

[![GitHub stars](https://img.shields.io/github/stars/debnsuma/fine-tuning-lora?color=blue&style=flat-square)](https://github.com/debnsuma/fine-tuning-lora/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/debnsuma/fine-tuning-lora?color=green&style=flat-square)](https://github.com/debnsuma/fine-tuning-lora/network/members)
[![GitHub watchers](https://img.shields.io/github/watchers/debnsuma/fine-tuning-lora?color=red&style=flat-square)](https://github.com/debnsuma/fine-tuning-lora/watchers)
[![GitHub issues](https://img.shields.io/github/issues/debnsuma/fine-tuning-lora?color=orange&style=flat-square)](https://github.com/debnsuma/fine-tuning-lora/issues)

A hands-on path into LLM fine-tuning: what quantization and LoRA actually do, built from first principles, and how to run a real fine-tune, deploy, and evaluate workflow end to end on a managed service. The concepts apply anywhere; the hands-on samples run on [Crusoe Cloud](https://www.crusoe.ai/), whose Managed AI stack is OpenAI-API-compatible, so everything is driven with the standard `openai` Python SDK.

## What is in here

| Folder | What it covers |
|---|---|
| [01-into-lora-fine-tuning](./01-into-lora-fine-tuning/) | Foundations, from scratch: quantization, then LoRA and QLoRA, implemented and explained across two notebooks |
| [02-serverless-fine-tuning-pii-redaction](./02-serverless-fine-tuning-pii-redaction/) | End-to-end sample: fine-tune Qwen3 8B into a PII redaction engine on Crusoe Serverless Fine-Tuning, deploy it as a dedicated endpoint, and evaluate it against a general-purpose 70B model |

Work through `01-into-lora-fine-tuning` first if the mechanics of LoRA and quantization are new to you, then move to `02-serverless-fine-tuning-pii-redaction` to see the same ideas run, at scale, on a fully managed service.

## Getting a VM (with `GPU`) on Crusoe Cloud

 If you do not already have a GPU machine, you may like to use Crusoe Cloud and sping up a VM with the GPU of your choice.

1. Log in to the [Crusoe Console](https://console.crusoecloud.com/).
2. Go to **Compute -> Instances -> Create Instance** ([Creating a VM docs](https://docs.crusoecloud.com/quickstart/creating-a-vm/index.html)).
3. Pick a GPU type. A single GPU is plenty for every notebook in this repo, these were built and tested on one NVIDIA L40S.
4. Pick an image with NVIDIA drivers preinstalled, for example `ubuntu22.04-nvidia-pcie-docker` for PCIe GPUs like L40S or A100, or `ubuntu22.04-nvidia-sxm-docker` for SXM GPUs like H100 (see [Images overview](https://docs.crusoecloud.com/compute/images/overview)).
5. Add your SSH public key, give the boot disk enough room for model downloads (50 GB or more is comfortable), and create the instance.
6. Once it is running, copy its public IP and connect:

```bash
ssh ubuntu@<PUBLIC_IP>
```

`ubuntu` is the default user on every Ubuntu-based Crusoe image. See [Accessing your VMs](https://docs.crusoecloud.com/compute/virtual-machines/accessing-vms/index.html) for more connection options.

## Setting up the shared environment

Every notebook in this repository, runs on one shared virtual environment created at the repo root with [uv](https://docs.astral.sh/uv/). 

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  
sudo apt-get update && sudo apt-get install -y python3.12-dev
source ~/.bashrc 


# Clone the repo
git clone https://github.com/debnsuma/fine-tuning-lora.git
cd fine-tuning-lora

# Create the virtual env
uv venv --python 3.12
uv pip install -r requirements.txt

.venv/bin/python -m ipykernel install --user --name fine-tuning-lora --display-name "fine-tuning-lora (.venv)"
```

## Foundations

[The first section](./01-into-lora-fine-tuning/) builds quantization, LoRA, and QLoRA up from first principles in plain PyTorch, before reaching for any library:

```bash
cd 01-into-lora-fine-tuning
../.venv/bin/jupyter lab 01_quantization.ipynb
```

No API key needed, only a GPU as noted above. See [its README](./01-into-lora-fine-tuning/README.md) for what each notebook covers.

## Serverless Fine-Tuning with Crusoe

[In this section](./02-serverless-fine-tuning-pii-redaction/) we will go over the end-to-end walkthrough of [Crusoe Serverless Fine-Tuning](https://docs.crusoecloud.com/serverless-fine-tuning/overview) and [Self-Serve Deployments](https://docs.crusoecloud.com/self-serve-deployments/overview) by:

- Fine-tuning a [Qwen3 8B](https://huggingface.co/Qwen/Qwen3-8B) to find and mask personally identifiable information in financial documents
- Then deploying it as a dedicated endpoint, and 
- Finally, evaluate it against a general-purpose 70B model.

The recipe is model agnostic, you can swap the base model with any other [supported model](https://www.crusoe.ai/cloud/serverless-fine-tuning) in the fine-tuning registry and everything else stays the same. PII redaction is exactly the workload you cannot send to a third-party model API, because the input is the sensitive data itself, fine-tuning and serving the model inside your own cloud environment is the point, not an implementation detail.

This needs a Crusoe Inference API key:

1. Log in to the [Crusoe Console](https://console.crusoecloud.com/).
2. Go to **Organization settings -> Security -> [Inference API keys](https://console.crusoecloud.com/security/inference-api-keys)**.
3. Create a key and copy it.

```bash
cd 02-serverless-fine-tuning-pii-redaction
cp .env.example .env          # add your [CRUSOE_API_KEY]
../.venv/bin/jupyter lab finetune-deploy-inference.ipynb
```

The notebook walks through:

1. **Prepare the data**: download a dataset from Hugging Face, convert it to chat-format JSONL, and validate it before upload
2. **Fine-tune**: launch a LoRA job on Crusoe Serverless Fine-Tuning, billed per training token
3. **Deploy**: turn the checkpoint into a dedicated endpoint with a few clicks in the Crusoe Console
4. **Run inference and evaluate**: call your model through the OpenAI-compatible API and score it against a baseline
5. **Clean up**: delete the dedicated deployment and the uploaded dataset files so nothing keeps billing after you close the notebook

### Dataset format reference

Every supervised fine-tuning job takes data in the chat format (JSONL):

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is LoRA?"}, {"role": "assistant", "content": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique..."}]}
```

### Links

- Launch blog: [Crusoe Introduces Serverless Fine-Tuning](https://www.crusoe.ai/resources/blog/crusoe-introduces-serverless-fine-tuning)
- Launch blog: [Crusoe Self-Serve Deployments](https://www.crusoe.ai/resources/blog/crusoe-self-serve-deployments)
- [Serverless Fine-Tuning documentation](https://docs.crusoecloud.com/serverless-fine-tuning/overview)
- [Self-Serve Deployments documentation](https://docs.crusoecloud.com/self-serve-deployments/overview)
- [Fine-tuning API reference](https://docs.crusoecloud.com/api/managed-ai/#tag/Fine-tuning)
- [Supported models for fine-tuning](https://www.crusoe.ai/cloud/serverless-fine-tuning)
