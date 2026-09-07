from self_steering.prompts.serialization import (
    answer_instruction,
    build_chat_messages,
    serialize_reasoning_prefill,
)
from self_steering.prompts.templates import CAPABILITY_PROMPTS, GENERIC_PROMPT


class RecordingTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return [10, 20, 30]


def test_exactly_four_capability_prompts_plus_generic() -> None:
    assert set(CAPABILITY_PROMPTS) == {"QLl", "QLq", "CL", "MCr"}
    assert GENERIC_PROMPT not in CAPABILITY_PROMPTS.values()


def test_math_prompt_uses_stir_style_chat_template() -> None:
    tokenizer = RecordingTokenizer()
    ids = serialize_reasoning_prefill(
        tokenizer,
        "instruction",
        "question",
        "Return boxed math.",
    )
    assert ids == [10, 20, 30]
    assert tokenizer.messages == [
        {
            "role": "system",
            "content": "You are a helpful assistant. Follow the user's instructions carefully.",
        },
        {
            "role": "user",
            "content": "instruction\nReturn boxed math.\n\nProblem:\nquestion",
        },
    ]
    assert tokenizer.kwargs["add_generation_prompt"] is True
    assert "continue_final_message" not in tokenizer.kwargs
    assert tokenizer.kwargs["tokenize"] is True


def test_generic_and_capability_change_only_reasoning_instruction() -> None:
    generic = build_chat_messages(GENERIC_PROMPT, "Q", "format")
    capability = build_chat_messages(CAPABILITY_PROMPTS["QLq"], "Q", "format")
    assert generic[0] == capability[0]
    assert (
        generic[-1]["content"].split("\n\nProblem:", 1)[1]
        == capability[-1]["content"].split("\n\nProblem:", 1)[1]
    )


def test_answer_instructions_are_dataset_specific() -> None:
    assert "option letter" in answer_instruction("choice", "arc_c")
    assert "boxed" in answer_instruction("math", "math500")
    assert "0 to 999" in answer_instruction("math", "aime2026")
