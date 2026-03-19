import json
from azure.ai.inference.models import UserMessage, SystemMessage


JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating two AI model responses to the same prompt.

Evaluate both responses on the following criteria, scoring each from 1-5:
{criteria}

For each criterion, provide:
- A score for Response A (1-5)
- A score for Response B (1-5)
- Brief reasoning

Then declare an overall winner (A, B, or Tie).

Respond ONLY with valid JSON in this exact format:
{{
  "evaluations": [
    {{
      "criterion": "<criterion name>",
      "score_a": <int>,
      "score_b": <int>,
      "reasoning": "<brief explanation>"
    }}
  ],
  "overall_winner": "<A|B|Tie>",
  "overall_reasoning": "<brief summary>"
}}"""

DEFAULT_CRITERIA = ["Accuracy", "Completeness", "Clarity"]


def judge_response(
    client,
    judge_model: str,
    prompt: str,
    response_a: str,
    response_b: str,
    model_a_name: str = "A",
    model_b_name: str = "B",
    criteria: list[str] | None = None,
) -> dict:
    """Use an LLM judge to evaluate two model responses to the same prompt.

    Args:
        client: Authenticated ChatCompletionsClient
        judge_model: Model deployment name for the judge
        prompt: The original user prompt both models answered
        response_a: Response from model A
        response_b: Response from model B
        model_a_name: Display name for model A
        model_b_name: Display name for model B
        criteria: Evaluation criteria (defaults to Accuracy, Completeness, Clarity)

    Returns:
        dict with evaluations, overall_winner, overall_reasoning, and model names
    """
    criteria = criteria or DEFAULT_CRITERIA
    criteria_text = "\n".join(f"- {c}" for c in criteria)

    system_msg = JUDGE_SYSTEM_PROMPT.format(criteria=criteria_text)
    user_msg = (
        f"**Original Prompt:**\n{prompt}\n\n"
        f"**Response A ({model_a_name}):**\n{response_a}\n\n"
        f"**Response B ({model_b_name}):**\n{response_b}"
    )

    response = client.complete(
        model=judge_model,
        messages=[
            SystemMessage(content=system_msg),
            UserMessage(content=user_msg),
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON from response, handling possible markdown code fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    result = json.loads(raw)
    result["model_a"] = model_a_name
    result["model_b"] = model_b_name
    result["prompt"] = prompt[:100]
    return result


def judge_comparison(
    client,
    judge_model: str,
    prompts: list[str],
    results_a: list[dict],
    results_b: list[dict],
    model_a_name: str = "A",
    model_b_name: str = "B",
    criteria: list[str] | None = None,
) -> list[dict]:
    """Run LLM-as-judge across all prompt pairs from a side-by-side comparison.

    Args:
        client: Authenticated ChatCompletionsClient
        judge_model: Model deployment name for the judge
        prompts: The original prompts
        results_a: Results from run_latency_test for model A
        results_b: Results from run_latency_test for model B
        model_a_name: Display name for model A
        model_b_name: Display name for model B
        criteria: Evaluation criteria list

    Returns:
        List of judge result dicts, one per prompt
    """
    judgments = []
    for i, prompt in enumerate(prompts):
        judgment = judge_response(
            client=client,
            judge_model=judge_model,
            prompt=prompt,
            response_a=results_a[i]["response_text"],
            response_b=results_b[i]["response_text"],
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            criteria=criteria,
        )
        judgment["prompt_index"] = i
        judgments.append(judgment)
    return judgments
