

KEYWORDS = {
    'reflection': [
        "rethink",
        "re-think"
        "recheck",
        "re-check",
        "double check",
        "try again",
        "re-evaluate",
        "check again",
        "let's correct it",
        "verify this step",
        "let's think again",
        "let's check",
        "核实",
        "验证",
        "检查",
        "稍等",
    ],
    'new_idea': [
        "alternatively",
        "try another",
        "use different",
        "find a new way",
        "reframe",
        "reimagine",
        "reconsider",
        "re-envision",
        "改进",
        "重新思考",
        "重新构想",
        "新思路",
        " wait",
    ] 
}

from flashtext import KeywordProcessor
from collections import Counter

keyword_processor = KeywordProcessor()

for key in KEYWORDS:
    for word in KEYWORDS[key]:
        keyword_processor.add_keyword(word, key)

from functools import lru_cache
@lru_cache
def pattern_count(response):
    results = keyword_processor.extract_keywords(response)
    my_counter = {
        'reflection': 0.0,
        'new_idea': 0.0
    }
    for item in results:
        my_counter[item] += 1.0
        
    final_counter = {
        'reflection': 0.0,
        'new_idea': 0.0
    }
    for key in my_counter:
        if my_counter[key] > 0:
            final_counter[key] = 1.0
        else:
            final_counter[key] = 0.0
    return dict(final_counter)