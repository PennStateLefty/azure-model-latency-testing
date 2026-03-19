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

    base_url = str(getattr(client, "base_url", "")).rstrip("/")
    if base_url:
        return f"{base_url}{fallback_path}"
    return f"<unknown>{fallback_path}"


def _serialize_annotation(annotation) -> dict:
    """Convert an SDK annotation object into a plain dict for notebook analysis."""
    if hasattr(annotation, "model_dump"):
        return annotation.model_dump()
    if isinstance(annotation, dict):
        return annotation
    return {"value": str(annotation)}


def _record_final_response(response) -> dict:
    """Extract final response status details from completed or incomplete events."""
    usage = getattr(response, "usage", None)
    incomplete_details = getattr(response, "incomplete_details", None)

    return {
        "final_status": getattr(response, "status", None),
        "incomplete_reason": getattr(incomplete_details, "reason", None),
        "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        "reasoning_tokens": _safe_reasoning_tokens(usage),
    }


def _safe_reasoning_tokens(usage) -> int:
    """Extract reasoning_tokens from usage.output_tokens_details if available."""
    if not usage:
        return 0
    details = getattr(usage, "output_tokens_details", None)
    if not details:
        return 0
    return getattr(details, "reasoning_tokens", 0) or 0


def measure_responses_latency(client, messages: list, model: str, **kwargs) -> dict:
    """Measure latency metrics for a single Responses API request using streaming.

    Args:
        client: OpenAI client configured for Azure Foundry
        messages: List of message dicts (role/content) — passed as 'input'
        model: Model deployment name
        **kwargs: Additional args passed to client.responses.create()

    Returns:
        dict with latency metrics, generated text, and normalized safety metadata
    """
    run_timestamp = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    ttft = None
    chunks = []
    completion_tokens = 0
    input_tokens = 0
    total_tokens = 0
    reasoning_tokens = 0
    final_status = None
    incomplete_reason = None
    annotation_events = []
    event_counts = {}

    try:
        stream = client.responses.create(
            model=model,
            input=messages,
            stream=True,
            **kwargs,
        )

        for event in stream:
            event_counts[event.type] = event_counts.get(event.type, 0) + 1

            if event.type == "response.output_text.delta":
                if ttft is None:
                    ttft = time.perf_counter() - start
                chunks.append(event.delta)

            if event.type == "response.output_text.annotation.added":
                annotation = _serialize_annotation(event.annotation)
                annotation_events.append(
                    {
                        "sequence_number": event.sequence_number,
                        "seconds_since_start": round(time.perf_counter() - start, 4),
                        "annotation_index": event.annotation_index,
                        "content_index": event.content_index,
                        "output_index": event.output_index,
                        "annotation_type": annotation.get("type", "unknown"),
                        "annotation": annotation,
                    }
                )

            if event.type in {"response.completed", "response.incomplete"}:
                final_response = _record_final_response(event.response)
                final_status = final_response["final_status"]
                incomplete_reason = final_response["incomplete_reason"]
                completion_tokens = final_response["output_tokens"]
                input_tokens = final_response["input_tokens"]
                total_tokens = final_response["total_tokens"]
                reasoning_tokens = final_response["reasoning_tokens"]
    except Exception as exc:
        request_uri = _extract_request_uri(exc, client, "/responses")
        raise RuntimeError(
            "Responses API request failed. "
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
    first_annotation_time = annotation_events[0]["seconds_since_start"] if annotation_events else None
    last_annotation_time = annotation_events[-1]["seconds_since_start"] if annotation_events else None

    return {
        "ttft": round(ttft, 4) if ttft else None,
        "total_time": round(total_time, 4),
        "tokens_per_second": round(tokens_per_second, 2),
        "decode_tokens_per_second": round(decode_tokens_per_second, 2),
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "input_tokens": input_tokens,
        "total_tokens": total_tokens,
        "response_text": response_text,
        "run_timestamp": run_timestamp,
        "final_status": final_status,
        "incomplete_reason": incomplete_reason,
        "content_filter_triggered": incomplete_reason == "content_filter",
        "annotation_count": len(annotation_events),
        "first_annotation_time": first_annotation_time,
        "last_annotation_time": last_annotation_time,
        "annotation_events": annotation_events,
        "event_counts": event_counts,
    }


def run_responses_latency_test(
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
    """Run Responses API latency measurements across multiple prompts.

    Args:
        client: OpenAI client configured for Azure Foundry
        model: Model deployment name
        prompts: List of user prompt strings to test
        system_prompt: System message for all requests
        num_iterations: Number of times to repeat the full prompt set
        warmup: Number of throwaway requests before measured runs
        reasoning_effort: Reasoning effort level ("low", "medium", "high") or None
        max_output_tokens: Cap on output tokens per request, or None for unlimited
        **kwargs: Additional args passed to client.responses.create()

    Returns:
        List of result dicts, one per prompt per iteration.
    """
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens

    # Warm-up: run throwaway requests to avoid cold-start skew
    if warmup > 0:
        warmup_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompts[0]},
        ]
        for _ in range(warmup):
            measure_responses_latency(client, warmup_messages, model, **kwargs)

    results = []
    for iteration in range(num_iterations):
        for i, prompt in enumerate(prompts):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            result = measure_responses_latency(client, messages, model, **kwargs)
            result["model"] = model
            result["prompt_index"] = i
            result["prompt"] = prompt[:100]
            result["iteration"] = iteration
            result["reasoning_effort"] = reasoning_effort
            results.append(result)
    return results
