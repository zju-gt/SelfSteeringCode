"""Qwen assistant-prefill prompt serialization."""

from __future__ import annotations

from typing import Any


def answer_instruction(answer_type: str, dataset: str) -> str:
    if answer_type == "choice":
        return "Conclude with `Final Answer: X`, where X is only the uppercase option letter."
    if dataset.lower().startswith("aime"):
        return (
            "Conclude with `Final Answer: \\boxed{N}`, where N is one integer from 0 to 999."
        )
    if answer_type == "math":
        return "Conclude with `Final Answer: \\boxed{...}` containing only the final answer."
    raise ValueError(f"unsupported answer type: {answer_type!r}")


def build_chat_messages(
    reasoning_instruction: str,
    question: str,
    final_answer_instruction: str,
) -> list[dict[str, str]]:
    user_content = (
        f"{reasoning_instruction.strip()}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"{final_answer_instruction.strip()}"
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "Reasoning:"},
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
        add_generation_prompt=False,
        continue_final_message=True,
    )

