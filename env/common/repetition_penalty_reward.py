from transformers import AutoTokenizer
import os, sys
ROBERTA_PATH = os.getenv('ROBERTA_PATH', '/newcpfs/user/chenhao/pretrained_models/FacebookAI/xlm-roberta-base/')
distinct_ngram_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)

import logging, json
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info({
    'INFO': '##REPETITION-PENALTY##',
    'VALUE': ROBERTA_PATH
})

import re

def merge_subwords(tokens):
    result = []
    current_word = []
    
    for token in tokens:
        # 处理前导的"▁"
        if token.startswith('▁'):
            if current_word:
                result.append(''.join(current_word))
                current_word = []
            token = token[1:]
        
        # 跳过空token
        if not token:
            continue
        
        # 检查token是否为中文字符
        if is_chinese_char(token):
            if current_word:
                result.append(''.join(current_word))
                current_word = []
            result.append(token)
        else:
            current_word.append(token)
    
    # 添加最后一个可能的单词
    if current_word:
        result.append(''.join(current_word))
    
    return result

def is_chinese_char(text):
    # 检查字符是否为中文字符
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False



def multilingual_tokenization(text):
    tokens = distinct_ngram_tokenizer.tokenize(text)
    merged = merge_subwords(tokens)
    return merged

def zipngram(words, ngram_size):
    return zip(*[words[i:] for i in range(ngram_size)])

def repetition_penalty_reward(completions, ngram_size, max_penalty) -> float:
    """
    reward function the penalizes repetitions
    ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
        completions: List of model completions
    """

    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    contents = [completion for completion in completions]
    rewards = []
    for completion in contents:

        tokens = multilingual_tokenization(completion)
        if len(tokens) < ngram_size:
            rewards.append(0.0)
            logger.info({
                'INFO': '##REPETITION-PENALTY##',
                'VALUE': f"completion {tokens} has less than {ngram_size} tokens"
            })
            continue

        ngrams = set()
        total = 0
        for ng in zipngram(tokens, ngram_size):
            ngrams.add(ng)
            total += 1

        scaling = 1 - len(ngrams) / total
        reward = scaling * max_penalty
        rewards.append(reward)
    return rewards