import time
from datetime import datetime, timezone


def _extract_request_uri(exc, client, fallback_path: str) -> str:
    """Return the best available request URI from the exception or client config."""
    for candidate in (
        getattr(getattr(exc, "request", None), "url", None),
        getattr(getattr(getattr(exc, "response", None), "request", None), "url", None),
        getattr(getattr(exc, "response", None), "url", None),
    ):
        if candidate:
            return str(candidate)

    base_url = str(getattr(client, "base_url", getattr(client, "_endpoint", ""))).rstrip("/")
    if base_url:
        return f"{base_url}{fallback_path}"
    return f"<unknown>{fallback_path}"


def measure_latency(client, messages: list, model: str, **kwargs) -> dict:
    """Measure latency metrics for a single chat completion request using streaming.

    Returns:
        dict with keys:
            - ttft: Time to first token (seconds)
            - total_time: Total response time (seconds)
            - tokens_per_second: Output tokens per second
            - completion_tokens: Number of tokens in the response
            - response_text: The full generated text
    """
    run_timestamp = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    ttft = None
    chunks = []
    completion_tokens = 0
    reasoning_tokens = 0

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - start
                chunks.append(chunk.choices[0].delta.content)

            if chunk.usage:
                completion_tokens = chunk.usage.completion_tokens
                details = getattr(chunk.usage, "completion_tokens_details", None)
                if details:
                    reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
    except Exception as exc:
        request_uri = _extract_request_uri(exc, client, "/chat/completions")
        raise RuntimeError(
            "Chat Completions request failed. "
            f"Model: {model}. "
            f"Request URI: {request_uri}. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    total_time = time.perf_counter() - start

    # Estimate token count from chunks if usage wasn't reported
    if completion_tokens == 0:
        completion_tokens = len(chunks)

    response_text = "".join(chunks)
    tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
    decode_time = (total_time - ttft) if ttft else total_time
    decode_tokens_per_second = completion_tokens / decode_time if decode_time > 0 else 0

    return {
        "ttft": round(ttft, 4) if ttft else None,
        "total_time": round(total_time, 4),
        "tokens_per_second": round(tokens_per_second, 2),
        "decode_tokens_per_second": round(decode_tokens_per_second, 2),
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "response_text": response_text,
        "run_timestamp": run_timestamp,
    }


def run_latency_test(
    client,
    model: str,
    prompts: list[str],
    system_prompt: str = "You are a helpful assistant.",
    num_iterations: int = 1,
    warmup: int = 0,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    **kwargs,
) -> list[dict]:
    """Run latency measurements across multiple prompts.

    Args:
        client: AzureOpenAI client
        model: Model deployment name
        prompts: List of user prompt strings to test
        system_prompt: System message for all requests
        num_iterations: Number of times to repeat the full prompt set
        warmup: Number of throwaway requests before measured runs
        reasoning_effort: Reasoning effort level ("low", "medium", "high") or None
        max_output_tokens: Cap on output tokens per request, or None for unlimited
        **kwargs: Additional args passed to client.chat.completions.create()

    Returns:
        List of result dicts, one per prompt per iteration.
    """
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if max_output_tokens is not None:
        kwargs["max_completion_tokens"] = max_output_tokens

    # Warm-up: run throwaway requests to avoid cold-start skew
    if warmup > 0:
        warmup_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompts[0]},
        ]
        for _ in range(warmup):
            measure_latency(client, warmup_messages, model, **kwargs)

    results = []
    for iteration in range(num_iterations):
        for i, prompt in enumerate(prompts):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            result = measure_latency(client, messages, model, **kwargs)
            result["model"] = model
            result["prompt_index"] = i
            result["prompt"] = prompt[:100]
            result["iteration"] = iteration
            result["reasoning_effort"] = reasoning_effort
            results.append(result)
    return results
