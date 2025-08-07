



import re

def extract_code(text):
    if text.startswith("```python"):
        text = "hey\n" + text
    blocks = [block.split("```", 1)[0].strip() for block in text.split("```python") if '```' in block]
    blocks = [block for block in blocks if block]
    if not blocks:
        return ""
    code = []
    for block in blocks[:-1]:
        for line in block.split("\n"):
            if line.startswith("    ") or line.startswith("import") or line.startswith("def "):
                code.append(line)
            elif 'print(' not in line:
                code.append(line)
    code = "\n".join(code) + "\n" + blocks[-1]
    return code.strip()

# def extract_code(completion: str) -> str:
#     pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
#     matches = pattern.findall(completion)
#     extracted_answer = matches[-1] if len(matches) >= 1 else ""
#     return extracted_answer

import re
code_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
output_pattern = re.compile(r"```output\n(.*?)```", re.DOTALL)