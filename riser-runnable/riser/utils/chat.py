"""Helpers for consistently formatting user prompts with a model chat template."""

from __future__ import annotations

from typing import Any


def has_chat_template(tokenizer: Any) -> bool:
    """Return whether a tokenizer exposes a non-empty Hugging Face chat template."""

    return bool(
        callable(getattr(tokenizer, "apply_chat_template", None))
        and getattr(tokenizer, "chat_template", None)
    )


def format_chat_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    add_generation_prompt: bool = True,
    require_chat_template: bool = False,
) -> str:
    """Format a raw user prompt with the tokenizer's chat template.

    Evaluation JSONL and contrastive prompt-pair files intentionally store the
    readable user content. This helper is the single runtime boundary where
    that content becomes a model-ready conversation.
    """

    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")

    if not has_chat_template(tokenizer):
        if require_chat_template:
            raise ValueError(
                "The tokenizer does not provide a chat_template. "
                "Use an instruct/chat model or set a tokenizer chat template."
            )
        return prompt

    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(formatted, str):
        raise TypeError(
            "tokenizer.apply_chat_template(..., tokenize=False) must return a string"
        )
    return formatted


__all__ = ["format_chat_prompt", "has_chat_template"]
