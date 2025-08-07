


import re
from typing import Dict, Tuple, Optional

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from functools import lru_cache

@lru_cache
def extract_solution(solution_str):
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if not matches:
        logger.info("[Error] No valid answer tags found")
        return None, solution_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, solution_str

@lru_cache
def validate_response_structure(processed_str, tokenizer):
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    logger.info("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        # 'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        logger.info(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            logger.info(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # # Verify tag order
    # if (positions['think_start'] > positions['think_end'] or
    #     positions['think_end'] > positions['answer_start'] or
    #     positions['answer_start'] > positions['answer_end']):
    #     logger.info("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
    #     validation_passed = False
    # else:
    #     logger.info("  Tag sequence validation passed")
    
    if validation_passed:
        left_str = ''.join(processed_str.split('</answer>')[1:])
        left_str_len = len(tokenizer(left_str)['input_ids'])
        if left_str_len > 50:
            validation_passed = False

    return validation_passed