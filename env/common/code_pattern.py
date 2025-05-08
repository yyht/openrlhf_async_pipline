


# from pygments.lexers import guess_lexer
# from pygments.util import ClassNotFound

# def has_code(text):
#     try:
#         # 尝试猜测文本的语言类型
#         lexer = guess_lexer(text)
#         return True
#     except ClassNotFound:
#         return False


import re

def extract_code_deepseek(text):
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

def has_code(text):
    try:
        # 尝试猜测文本的语言类型
        lexer = extract_code_deepseek(text)
        if lexer:
            return True
        else:
            return False
    except:
        return False



    