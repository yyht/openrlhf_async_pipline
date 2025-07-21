

def multiturn_code_merge(code_snippets_dict_list):
    code_snippets = ""
    for code_snippets_dict in code_snippets_dict_list:
        if code_snippets_dict['error'] > 0:
            continue

        code = code_snippets_dict['query']
        execout = code_snippets_dict['exec_result']
        code_snippets += code
    return code_snippets