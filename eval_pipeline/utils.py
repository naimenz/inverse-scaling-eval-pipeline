
# Note that this is still effectively zero-shot: the questions are designed to teach the expected
# answer format, not to help it answer tricky questions correctly
# Q: Are bananas yellow?
# A: Yes

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