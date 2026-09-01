"""The one generic and four capability-specific reasoning instructions."""

GENERIC_PROMPT = """Solve the problem carefully.
Reason systematically and verify the final answer."""

CAPABILITY_PROMPTS = {
    "QLl": """Identify the relevant premises and rules.
Track logical dependencies and derive conclusions only through warranted deductions.""",
    "QLq": """Identify the relevant quantities and numerical relationships.
Represent them explicitly and verify intermediate calculations.""",
    "CL": """Infer the underlying pattern or rule.
Form an abstraction that explains the examples and apply it to the new case.""",
    "MCr": """Identify which information is necessary.
Separate relevant evidence from distractors before solving the problem.""",
}
