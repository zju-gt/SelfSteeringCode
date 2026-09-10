"""The one generic and four capability-specific reasoning instructions."""

GENERIC_PROMPT = """Solve the following problem."""

CAPABILITY_PROMPTS = {
    "QLl": """Prioritize logical deduction.
Identify the relevant premises and rules, and use their logical dependencies to derive only warranted conclusions.""",
    "QLq": """Prioritize quantitative reasoning.
Identify the relevant quantities and relationships, and use explicit calculations to derive and verify the result.""",
    "CL": """Prioritize abstraction and rule induction.
Identify the underlying pattern or rule, and use that abstraction to explain the examples and solve the new case.""",
    "MCr": """Prioritize relevance filtering.
Identify the information necessary for the solution, and use relevant evidence while excluding irrelevant or distracting information.""",
}
