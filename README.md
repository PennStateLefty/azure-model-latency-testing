# Azure AI Foundry — Model Latency Testing

Lightweight Jupyter-based project for measuring and comparing latency characteristics of models hosted in Azure AI Foundry.

## Metrics Captured

- **Time to First Token (TTFT)** — how fast the model starts responding
- **Total Response Time** — end-to-end request duration
- **Tokens/sec (e2e)** — end-to-end throughput (completion tokens ÷ total time)
- **Tokens/sec (decode)** — decode-only throughput (completion tokens ÷ time after first token)
- **Reasoning Tokens** — tokens spent on chain-of-thought (when using reasoning models)
- **Statistical summaries** — median, p95, standard deviation across multiple iterations

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

## Experiment Knobs

Every notebook exposes a common set of configuration variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_ITERATIONS` | `3` | Repeat each prompt N times for statistical power |
| `WARMUP` | `1` | Throwaway requests before measured runs (avoids cold-start skew) |
| `INTERLEAVE` | `True` | Alternate A/B per prompt to reduce time-of-day bias (comparison notebooks) |
| `REASONING_EFFORT` | `None` | `"low"` / `"medium"` / `"high"` / `None` — controls reasoning depth for reasoning models |
| `MAX_OUTPUT_TOKENS` | `256` | Cap output length for consistent comparison (`None` = unlimited) |

When `NUM_ITERATIONS > 1`, charts render as box plots; otherwise bar charts.

## Project Structure

```
├── .env.example              # Required environment variables template
├── requirements.txt          # Python dependencies
├── utils/
│   ├── auth.py               # Managed identity auth (AzureOpenAI client)
│   ├── latency.py            # Latency measurement — Chat Completions API
│   ├── responses_latency.py  # Latency measurement — Responses API
│   ├── judge.py              # LLM-as-judge — Chat Completions API
│   └── responses_judge.py    # LLM-as-judge — Responses API
└── notebooks/
    ├── latency_test_template.ipynb  # Single-model latency testing
    ├── model_comparison.ipynb       # Side-by-side A/B comparison + quality eval
    └── sync_async_content_safety_comparison.ipynb  # Sync vs async content filter comparison
```

## Supported APIs

Both API paths use the `openai` SDK with an `AzureOpenAI` client (`get_client()` / `get_openai_client()` — they are aliases).

| API | Method | Best for |
|-----|--------|----------|
| **Chat Completions** | `client.chat.completions.create()` | Standard chat workloads |
| **Responses** | `client.responses.create()` | Responses API features (annotations, content safety streaming metadata) |

Switch between them in the comparison notebook by setting `API_TYPE = "responses"` or `"chat_completions"`.

## Notebooks

### Single-Model Latency Test (`latency_test_template.ipynb`)

Test latency for a single model using the Chat Completions API. Good for baselining or quick checks.

- **Metrics**: TTFT, total time, tokens/sec (e2e and decode), reasoning tokens
- **Statistics**: Average, median, p95, and standard deviation across iterations
- **Charts**: Box plots (multi-iteration) or bar charts (single iteration)

### Side-by-Side Comparison (`model_comparison.ipynb`)

Compare two models head-to-head:
- **API toggle**: Choose Responses API or Chat Completions API with a single variable
- **Interleaved A/B**: Alternates requests between models to eliminate time-of-day bias
- **Warm-up**: Configurable throwaway requests before measured runs
- **Latency**: TTFT, total time, tokens/sec (e2e and decode) with full statistical summaries
- **Quality** (optional): LLM-as-judge scoring on Accuracy, Completeness, and Clarity (criteria are customizable)
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
- **Interleaved testing**: Alternates sync/async requests to eliminate ordering bias
- **Observability**: TTFT, total time, tokens/sec (e2e and decode), annotation event counts, and terminal content-filter status
- **Safety stream views**: Annotation timing, terminal status, content-filter signals, and stream event counts

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
