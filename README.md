# Fine-Tuning with LoRA

A hands-on path into LLM fine-tuning: what LoRA and quantization actually do, and how to run a real fine-tune, deploy, and evaluate workflow end to end. The concepts apply anywhere; the hands-on samples run on [Crusoe Cloud](https://www.crusoe.ai/), whose Managed AI stack is OpenAI-API-compatible, so everything is driven with the standard `openai` Python SDK.

## What is in here

| Folder | What it covers | Status |
|---|---|---|
| [01-into-lora-fine-tuning](./01-into-lora-fine-tuning/) | Concepts: why fine-tune, how LoRA works, and where quantization fits | Coming soon |
| [02-pii-redaction](./02-pii-redaction/) | End-to-end sample: fine-tune Qwen3 8B into a PII redaction engine, deploy it as a dedicated endpoint, and evaluate it against a general-purpose 70B model | Available |

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A [Crusoe Cloud](https://console.crusoecloud.com/) account and an Inference API key (Organization settings -> Security -> Inference API keys) for the hands-on samples

## Setup

Each sample folder is self-contained with its own `requirements.txt` and `.env.example`. With uv:

```bash
cd 02-pii-redaction
cp .env.example .env          # then fill in CRUSOE_API_KEY
uv venv && uv pip install -r requirements.txt
.venv/bin/jupyter lab finetune-deploy-inference.ipynb
```

## The workflow at a glance

1. **Prepare the data**: download a dataset from Hugging Face and convert it to chat-format JSONL
2. **Fine-tune**: launch a LoRA job on Crusoe Serverless Fine-Tuning, billed per training token
3. **Deploy**: turn the checkpoint into a dedicated endpoint with a few clicks in the Crusoe Console
4. **Run inference and evaluate**: call your model through the OpenAI-compatible API and score it against a baseline

## Dataset format reference

Every supervised fine-tuning job takes data in the chat format (JSONL):

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is LoRA?"}, {"role": "assistant", "content": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique..."}]}
```

Rules:

- Each line is a complete JSON object (not a JSON array)
- `messages` is required; a `system` turn is optional but recommended
- Only `assistant` turns are included in the loss; `user` turns are masked
- Minimum 10 examples; 100 to 1000 high-quality examples is a practical target

## Links

- Launch blog: [Crusoe Introduces Serverless Fine-Tuning](https://www.crusoe.ai/resources/blog/crusoe-introduces-serverless-fine-tuning)
- Launch blog: [Crusoe Self-Serve Deployments](https://www.crusoe.ai/resources/blog/crusoe-self-serve-deployments)
- [Serverless Fine-Tuning documentation](https://docs.crusoecloud.com/serverless-fine-tuning/overview)
- [Self-Serve Deployments documentation](https://docs.crusoecloud.com/self-serve-deployments/overview)
- [Fine-tuning API reference](https://docs.crusoecloud.com/api/managed-ai/#tag/Fine-tuning)
- [Supported models for fine-tuning](https://www.crusoe.ai/cloud/serverless-fine-tuning)
