"""Primitive utilities with lazy optional-dependency imports."""

from importlib import import_module


_EXPORTS = {
    "ActivationExtractor": ("extractor", "ActivationExtractor"),
    "ContrastivePromptGenerator": ("extractor", "ContrastivePromptGenerator"),
    "LLMJudgeFilter": ("filtering", "LLMJudgeFilter"),
    "PrimitiveClustering": ("clustering", "PrimitiveClustering"),
    "PrimitiveLibrary": ("library", "PrimitiveLibrary"),
}

__all__ = [
    "ActivationExtractor",
    "ContrastivePromptGenerator",
    "LLMJudgeFilter",
    "PrimitiveClustering",
    "PrimitiveLibrary",
]


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attribute_name = _EXPORTS[name]
        value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
