



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