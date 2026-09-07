"""Chat prompt serialization for reasoning tasks."""

from __future__ import annotations

from typing import Any


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Follow the user's instructions carefully."


def answer_instruction(answer_type: str, dataset: str) -> str:
    if answer_type == "choice":
        return "Conclude with `Final Answer: X`, where X is only the uppercase option letter."
    if dataset.lower().startswith("aime"):
        return "Conclude with `Final Answer: \\boxed{N}`, where N is one integer from 0 to 999."
    if answer_type == "math":
        return "Put the final answer in \\boxed{<final answer>} on the last line."
    raise ValueError(f"unsupported answer type: {answer_type!r}")


def build_chat_messages(
    reasoning_instruction: str,
    question: str,
    final_answer_instruction: str,
) -> list[dict[str, str]]:
    user_content = (
        f"{reasoning_instruction.strip()}\n"
        f"{final_answer_instruction.strip()}\n\n"
        f"Problem:\n{question.strip()}"
    )
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def serialize_reasoning_prefill(
    tokenizer: Any,
    reasoning_instruction: str,
    question: str,
    final_answer_instruction: str,
) -> Any:
    messages = build_chat_messages(
        reasoning_instruction,
        question,
        final_answer_instruction,
    )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
