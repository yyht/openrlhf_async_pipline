
import re
from typing import Dict, Tuple, Optional

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import sys, os
sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_082'))

from env.logic.base import extract_solution, validate_response_structure
from functools import lru_cache

@lru_cache
def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary.
    
    Args:
        solution_text: Formatted solution text from dataset
        
    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    logger.info("\n[Ground Truth Parsing]")
    
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            logger.info(f"  Found: {name} → {role}")
        else:
            logger.info(f"  [Warning] Unparseable line: '{line}'")
    
    return status_dict

def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.
    
    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification
        
    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    logger.info("\n[Model Answer Parsing]")
    logger.info(f"  Expected characters: {expected_names}")

    knight_count = answer_text.lower().count('knight')
    knave_count = answer_text.lower().count('knave')

    logger.info(f"  Number of predicted roles: {knight_count + knave_count}")
    if knight_count + knave_count != len(expected_names):
        logger.info(f"  [Error] Number of characters mismatch: {knight_count + knave_count} != {len(expected_names)}")
        return None

    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
            re.IGNORECASE
        )
        match = pattern.search(answer_text)
        
        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            logger.info(f"  Found: {name} → {role}")
        else:
            logger.info(f"  [Error] Missing identification for {name}")
            return None
    
    return status_dict

def compute_score(solution_str, 
                 ground_truth,
                 format_reward=1,
                 answer_reward=1.0,
                 tokenizer=None) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    logger.info("\n" + "="*80)
    logger.info(" Processing New Sample ".center(80, '='))
    
    # Parse ground truth data
    solution_text = ground_truth
    gt_status = parse_solution_text_format(solution_text)
    expected_names = list(gt_status.keys())
    logger.info(f"[Ground Truth] Final identities: {gt_status}")

    # Extract model answer
    logger.info("## Extracting Model Answer ##")
    answer_text, processed_str = extract_solution(solution_str)
    logger.info(f"\n[Model Response]\n{processed_str}")

    # Validate response structure
    format_correct = validate_response_structure(processed_str, tokenizer)
    format_score = format_reward if format_correct else -abs(format_reward)
    logger.info(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    logger.info(f"  Format score: {format_score}")

    # Validate answer content
    answer_score = 0
    gold_score = 0
    if format_correct and answer_text:
        pred_status = parse_model_answer(answer_text, expected_names)
        if pred_status:
            logger.info(f"\n[Content Validation]")
            logger.info(f"  Expected: {gt_status}")
            logger.info(f"  Predicted: {pred_status}")
            
            correct_ratio = 0.0
            for key in gt_status:
                if key in pred_status:
                    if gt_status[key] == pred_status[key]:
                        correct_ratio += 1.0
            soft_score = correct_ratio / len(gt_status)
            
            if pred_status == gt_status:
                answer_score = 1.0
                gold_score = 1.0
                logger.info("  Content validation: FULL MATCH")
            elif correct_ratio > 0:
                answer_score = soft_score
                gold_score = soft_score
            else:
                answer_score = 0.0
                logger.info("  Content validation: MISMATCH")
        else:
            answer_score = 0.0
            logger.info( "Fail to parse answer")
    else:
        answer_score = 0.0
        logger.info("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    logger.info("\n" + "-"*80)
    logger.info(f" Final Score ".center(80, '-'))
    logger.info(f"  Format: {format_score}")
    logger.info(f"  Answer: {answer_score}")
    logger.info(f"  Total: {total_score}")
    logger.info("="*80 + "\n")

    return total_score, gold_score, answer_score