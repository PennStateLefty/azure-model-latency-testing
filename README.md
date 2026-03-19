# Azure AI Foundry — Model Latency Testing

Lightweight Jupyter-based project for measuring and comparing latency characteristics of models hosted in Azure AI Foundry.

## Metrics Captured

- **Time to First Token (TTFT)** — how fast the model starts responding
- **Total Response Time** — end-to-end request duration
- **Tokens per Second** — throughput of generated tokens

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <this-repo>
cd azure-model-latency-testing
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your Foundry endpoint and model name
```

### 4. Authentication

This project uses **managed identity** via `DefaultAzureCredential`. It will automatically use:

- **In Azure**: Managed identity attached to your compute
- **Locally**: Azure CLI credentials (`az login`)

No API keys needed.

### 5. Run notebooks

```bash
jupyter notebook notebooks/
```

## Project Structure

```
├── .env.example              # Required environment variables template
├── requirements.txt          # Python dependencies
├── utils/
│   ├── auth.py               # Managed identity auth (Inference + OpenAI clients)
│   ├── latency.py            # Latency measurement — Inference API
│   ├── responses_latency.py  # Latency measurement — Responses API
│   ├── judge.py              # LLM-as-judge — Inference API
│   └── responses_judge.py    # LLM-as-judge — Responses API
└── notebooks/
    ├── latency_test_template.ipynb  # Single-model latency testing
    ├── model_comparison.ipynb       # Generic side-by-side comparison + quality eval
    └── sync_async_content_safety_comparison.ipynb  # Same-model sync vs async filter comparison
```

## Supported APIs

This project supports two Azure AI Foundry API paths:

| API | SDK | Client | Best for |
|-----|-----|--------|----------|
| **Responses API** | `openai` | `get_openai_client()` | OpenAI models (GPT-5.x), also Grok, Mistral, Llama via Foundry |
| **Inference API** | `azure-ai-inference` | `get_client()` | Azure-native inference endpoint |

Switch between them in the comparison notebook by setting `API_TYPE = "responses"` or `"inference"`.

## Notebooks

### Single-Model Latency Test (`latency_test_template.ipynb`)

Test latency for a single model using the Inference API. Good for baselining or quick checks.

### Side-by-Side Comparison (`model_comparison.ipynb`)

Compare two models head-to-head:
- **API toggle**: Choose Responses API or Inference API with a single variable
- **Latency**: TTFT, total time, and tokens/sec displayed in pivot tables and bar charts
- **Quality** (optional): Uses an LLM-as-judge pattern with a separately configured judge model to score responses on Accuracy, Completeness, and Clarity (criteria are customizable)
- **Response inspection**: Full responses printed side by side for manual review

Configure models in `.env`:
```bash
AZURE_MODEL_A=gpt-5.2
AZURE_MODEL_B=gpt-5.4-mini
AZURE_JUDGE_MODEL=gpt-5.2
```

### Sync vs Async Content Safety Comparison (`sync_async_content_safety_comparison.ipynb`)

Compare the same model family deployed twice on the same Azure OpenAI or Foundry endpoint:
- **Sync deployment**: default streaming with synchronous or buffered filtering
- **Async deployment**: streaming with **Asynchronous Filter** enabled in the deployment's content filter configuration
- **Observability**: TTFT, total time, tokens per second, annotation event counts, and terminal content-filter status

Configure deployments in `.env`:
```bash
AZURE_BASE_MODEL_FAMILY=gpt-5.2
AZURE_SYNC_DEPLOYMENT=gpt-5.2-sync
AZURE_ASYNC_DEPLOYMENT=gpt-5.2-async
AZURE_JUDGE_MODEL=gpt-5.2-sync
```

Important:
- Asynchronous Filter is configured in Foundry portal as part of the deployment's content filter configuration.
- The notebook does not toggle sync versus async filtering per request.
- For a valid comparison, point both deployment names at the same underlying model family.

## Adding a New Model

1. Update `AZURE_ENDPOINT` and `AZURE_MODEL_NAME` in your `.env`, or
2. Duplicate the template notebook and configure the endpoint inline

No code changes required — just point at a different Foundry deployment.
