from self_steering.evaluation.answers import extract_answer, is_correct, normalize_math_answer


def test_extracts_last_boxed_answer_with_nested_braces() -> None:
    text = r"first \boxed{12}, correction \boxed{\frac{1}{2}}"
    assert extract_answer(text, "math") == r"\frac{1}{2}"


def test_extracts_final_answer_fallback() -> None:
    assert extract_answer("work\nFinal Answer: 42", "math") == "42"


def test_extracts_choice_letter() -> None:
    assert extract_answer("Reasoning. Final Answer: c", "choice") == "C"


def test_aime_leading_zeros_compare_as_integers() -> None:
    assert is_correct("007", "7", dataset="aime2026", answer_type="math")


def test_math_normalization_removes_common_wrappers() -> None:
    assert normalize_math_answer(" $ 1,234. $ ") == "1234"


def test_missing_answer_is_incorrect() -> None:
    assert extract_answer("unfinished reasoning", "math") is None
    assert not is_correct(None, "2", dataset="math500", answer_type="math")

