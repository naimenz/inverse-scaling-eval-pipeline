# Note that this is still effectively zero-shot: the questions are designed to teach the expected
# answer format, not to help it answer tricky questions correctly
# Q: Are bananas yellow?
# A: Yes

size_dict = {
    "gpt2": 124_000_000,
    "gpt2-medium": 355_000_000,
    "gpt2-large": 774_000_000,
    "gpt2-xl": 1_500_000_000,
    # GPT-3 sizes are based on https://blog.eleuther.ai/gpt3-model-sizes/
    "ada": 350_000_000,
    "babbage": 1_300_000_000,
    "curie": 6_700_000_000,
    "davinci": 175_000_000_000,
}

template = """
Q: Is the sun closer to the Earth than the moon?
A: No

Q: Is Germany in Asia?
A: No

Q: Is San Francisco in California?
A: Yes

Q: {question}
A:
""".strip()


def wrap_question(text: str) -> str:
    return template.format(question=text)
