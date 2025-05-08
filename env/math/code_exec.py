

import os
import io
from contextlib import redirect_stdout
import pickle
import regex
import copy
from typing import Any, Dict, Optional
import multiprocess
from concurrent.futures import TimeoutError
from functools import partial
import traceback
from timeout_decorator import timeout

class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []
    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        if regex.search(r'(\s|^)?input\(', code_piece) or regex.search(r'(\s|^)?os.system\(', code_piece):
            raise RuntimeError()
        exec(code_piece, self._global_vars)
        
    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)
    
    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v
    
    @property
    def answer(self):
        return self._global_vars['answer']
    
runtime = GenericRuntime()

@timeout(10, use_signals=False)
def run_code(code):
    code = code.split('\n')
    if "print(" in code[-1]:
        program_io = io.StringIO()
        with redirect_stdout(program_io):
            timeout(10)(runtime.exec_code)('\n'.join(code))
        program_io.seek(0)
        result = program_io.read()
    else:
        print(code)
        timeout(10)(runtime.exec_code)('\n'.join(code[:-1]))
        result = timeout(10)(runtime.eval_code)(code[-1])
    return result