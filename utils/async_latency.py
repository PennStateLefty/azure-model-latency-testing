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


async def async_measure_latency(client, messages: list, model: str, **kwargs) -> dict:
    """Measure latency metrics for a single chat completion request using async streaming.

    Async counterpart to measure_latency(). Returns the same dict shape.
    """
    run_timestamp = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    ttft = None
    chunks = []
    completion_tokens = 0
    reasoning_tokens = 0

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )

        async for chunk in response:
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
