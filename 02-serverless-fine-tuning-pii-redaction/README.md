# Fine-Tune a PII Redaction Engine with Crusoe Serverless Fine-Tuning

An end-to-end walkthrough of [Crusoe Serverless Fine-Tuning](https://docs.crusoecloud.com/serverless-fine-tuning/overview) and [Self-Serve Deployments](https://docs.crusoecloud.com/self-serve-deployments/overview): fine-tune [Qwen3 8B](https://huggingface.co/Qwen/Qwen3-8B) to find and mask personally identifiable information in financial documents, deploy it as a dedicated endpoint, and evaluate it against a general-purpose 70B model. 

The recipe is model agnostic, you can swap the base model with any other [supported model](https://www.crusoe.ai/cloud/serverless-fine-tuning) in the fine-tuning registry and everything else stays the same.

PII redaction is exactly the workload you cannot send to a third-party model API, because the input is the sensitive data itself. Fine-tuning and serving the model inside your own cloud environment is the point, not an implementation detail.

![PII redaction pipeline on Crusoe: raw documents, Serverless Fine-Tuning, Self-Serve Deployment, redacted output](assets/pii-redaction-pipeline.svg)

## Quickstart

### Get a Crusoe Inference API key

1. Log in to the [Crusoe Console](https://console.crusoecloud.com/).
2. Go to **Organization settings -> Security -> [Inference API keys](https://console.crusoecloud.com/security/inference-api-keys)**.
3. Create a key and copy it.

### Configure and install

This notebook shares the single virtual environment for the whole repository (no GPU needed here, everything runs against Crusoe's managed API). See the root [README](../README.md) for setting up the shared environment with `uv`. Once that is done:

```bash
cp .env.example .env          # then paste your key into CRUSOE_API_KEY
../.venv/bin/jupyter lab finetune-deploy-inference.ipynb
```

The notebook walks you through the rest, covering:

1. **Prepare the data**: download a synthetic PII dataset from Hugging Face, convert it to chat-format JSONL, and validate it before upload
2. **Fine-tune**: launch a LoRA job on Serverless Fine-Tuning and watch the live loss curves
3. **Deploy**: turn the best checkpoint into a dedicated endpoint with a few clicks in the Crusoe Console
4. **Run inference**: redact documents through the OpenAI-compatible API, with and without streaming
5. **Evaluate**: score the fine-tuned model against a general-purpose 70B baseline on entity-level F1
6. **Clean up**: delete the dedicated deployment and the uploaded dataset files so nothing keeps billing after you close the notebook

## Links

- Launch blog: [Crusoe Introduces Serverless Fine-Tuning](https://www.crusoe.ai/resources/blog/crusoe-introduces-serverless-fine-tuning)
- Launch blog: [Crusoe Self-Serve Deployments](https://www.crusoe.ai/resources/blog/crusoe-self-serve-deployments)
- [Serverless Fine-Tuning documentation](https://docs.crusoecloud.com/serverless-fine-tuning/overview)
- [Self-Serve Deployments documentation](https://docs.crusoecloud.com/self-serve-deployments/overview)
- [Fine-tuning API reference](https://docs.crusoecloud.com/api/managed-ai/#tag/Fine-tuning)
- Model: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
