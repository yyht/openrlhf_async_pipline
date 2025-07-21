

import sys, os
os.environ['COMPILE_SERVER'] = 'http://10.39.2.54:10003'
os.environ['REMOTE_RM_URL'] = 'http://10.39.2.54:10007'
os.environ['MATH_VERIFY_SERVER'] = 'http://10.39.2.54:10008'
os.environ['XVERIFY_MATH_MODEL_SERVER'] = 'http://10.39.2.54:10005'
os.environ['EMBEDDING_MODEL_SERVER'] = 'http://10.39.2.54:10009'
os.environ['DEBUG_FLAG'] = 'no'
os.environ['PRETRAIN'] = '/cpfs/user/chenhao/pretrained_models/Qwen/Qwen2.5-7B-local/'
os.environ['OPENRLHF_PATH'] = '/cpfs/user/chenhao/debug/OpenRLHF_082/'
os.environ['ROBERTA_PATH'] = '/cpfs/user/chenhao/pretrained_models/FacebookAI/xlm-roberta-base/'
os.environ['NGINX_IP_FILE']='/cpfs/user/chenhao/hf_datasets/qwen25_qwq/nginx_conf/nginx_ip.txt'
os.environ['COMPILE_SERVER_PORT']='10003'
os.environ['MATH_VERIFY_SERVER_PORT']='10008'
os.environ['XVERIFY_MATH_MODEL_SERVER_PORT']='10005'
os.environ['REMOTE_RM_URL']='http://10.39.2.54:10007'
import sys, os
sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_082'))
import sys, os

from env.math.math_reward import math_score
from env.math.math_verify_server import math_verify_server
from env.math.xverify_server import xverify_server
from env.common.embedding import embedding_server
from env.logic.kk_reward import kk_score
from env.logic.zebralogic_reward import zebralogic_score
from env.synlogic.synlogic_reward import synlogic_score
from env.synlogic.synlogic_utils import synlogic_score_fn
# from env.math.math_tir_process_single_request import remote_compile
from env.math.sandbox_utils import remote_compile

prompt = """
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:Given \\( x_{1}, x_{2}, x_{3} \\in [0, 12] \\),\n\\[ x_{1} x_{2} x_{3} = \\left(\\left(12 - x_{1}\\right)\\left(12 - x_{2}\\right)\\left(12 - x_{3}\\right)\\right)^{2}. \\]\n\nFind the maximum value of \\( f = x_{1} x_{2} x_{3} \\).\nAssistant: <think>
"""
response = "To solve the problem, we need to determine how many of Farmer Brown's 20 animals are chickens and how many are cows, given that chickens have 2 legs and cows have 4 legs, and the total number of legs is 70.\n\nLet's define:\n- \\( c \\) as the number of chickens.\n- \\( k \\) as the number of cows.\n\nFrom the problem, we have two key pieces of information:\n1. The total number of animals is 20.\n2. The total number of legs is 70.\n\nWe can express these pieces of information as two equations:\n1. \\( c + k = 20 \\)\n2. \\( 2c + 4k = 70 \\)\n\nFirst, we can simplify the second equation by dividing every term by 2:\n\\[ c + 2k = 35 \\]\n\nNow we have the system of linear equations:\n1. \\( c + k = 20 \\)\n2. \\( c + 2k = 35 \\)\n\nTo eliminate \\( c \\), we can subtract the first equation from the second:\n\\[ (c + 2k) - (c + k) = 35 - 20 \\]\n\\[ c + 2k - c - k = 15 \\]\n\\[ k = 15 \\]\n\nSo, there are 15 cows. Now we substitute \\( k = 15 \\) back into the first equation to find \\( c \\):\n\\[ c + 15 = 20 \\]\n\\[ c = 20 - 15 \\]\n\\[ c = 5 \\]\n\nTherefore, there are 5 chickens. To verify, we can check the total number of legs:\n- Chickens: \\( 5 \\times 2 = 10 \\) legs\n- Cows: \\( 15 \\times 4 = 60 \\) legs\n\nTotal legs:\n\\[ 10 + 60 = 70 \\]\n\nThe calculations are correct, so the number of chickens is \\( \\boxed{5} \\).\n</think>\n\n<answer> \\boxed{5} </answer><|endoftext|>"

gold_ans = '5'

label = {
    'gold_ans': gold_ans,
    'uuid': '1.0'
}

prompt = 'A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 3 inhabitants: Michael, Zoey, and Ethan. Michael was heard saying, "Ethan is a knight if and only if Michael is a knight". "Zoey is a knight or Ethan is a knight," Zoey mentioned. Ethan asserted: "Michael is a knave if and only if Zoey is a knave". So who is a knight and who is a knave?'
response = 'I have a solution </think> <answer>(1) Michael is a knight\n(2) Zoey is a knight\n(3) Ethan is a knight</answer>'

label = {
    'gold_ans': '(1) Michael is a knight\n(2) Zoey is a knight\n(3) Ethan is a knight',
    'uuid': '01e0f0f0-01e0f0f0-01e0f0f0-01e0f0f0',
}

prompt = 'There are 5 houses, numbered 1 to 5 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:\n - Each person has a unique name: `Arnold`, `Bob`, `Alice`, `Eric`, `Peter`\n - The people are of nationalities: `dane`, `swede`, `german`, `brit`, `norwegian`\n - Each person has a unique hobby: `gardening`, `painting`, `knitting`, `photography`, `cooking`\n - Everyone has a favorite smoothie: `dragonfruit`, `lime`, `watermelon`, `desert`, `cherry`\n - People have unique hair colors: `brown`, `gray`, `black`, `blonde`, `red`\n - People have unique favorite sports: `baseball`, `soccer`, `swimming`, `tennis`, `basketball`\n\n## Clues:\n1. The person who loves cooking is not in the second house.\n2. The Watermelon smoothie lover is not in the fourth house.\n3. The person who loves baseball is somewhere to the right of the Desert smoothie lover.\n4. The person who drinks Lime smoothies is the person who has brown hair.\n5. The person who has gray hair is somewhere to the right of the Swedish person.\n6. The photography enthusiast is not in the fifth house.\n7. The Swedish person is not in the first house.\n8. The German is the person who has blonde hair.\n9. There is one house between the Watermelon smoothie lover and the photography enthusiast.\n10. Alice and the person who drinks Lime smoothies are next to each other.\n11. The British person and the person who enjoys knitting are next to each other.\n12. The person who has black hair is the Swedish person.\n13. The British person is the person who enjoys gardening.\n14. The person who loves swimming is the person who loves cooking.\n15. The Dragonfruit smoothie lover is not in the third house.\n16. The Watermelon smoothie lover is not in the second house.\n17. Eric is somewhere to the left of the person who drinks Lime smoothies.\n18. The person who loves soccer is Alice.\n19. Peter is the photography enthusiast.\n20. The person who enjoys gardening is directly left of the person who loves tennis.\n21. The Norwegian and the photography enthusiast are next to each other.\n22. The British person is Alice.\n23. Bob is the person who has blonde hair.\n\nWhat is Smoothie of the person who lives in House 3?'
response = '</think> <answer> \\boxed{111}  </answer>'

label = {
    'gold_ans': 'cherry',
    'uuid': '123'
}

import asyncio
import math_verify
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from functools import lru_cache
@lru_cache
def math_verify_reward_function(solution_str, ground_truth):
    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    
    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return 0.0
    
    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return 0.0
    
    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return 1.0
    
    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                gt,
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return 1.0
        except Exception:
            continue
    
    # Very unlikely to be correct after the above matches
    return 0.0

async def math_verify_async():
    score = await asyncio.wait_for(
        asyncio.to_thread(
            math_verify_reward_function,
            f"\\boxed{{{128}}}",
            '128'
        ),
        timeout=10.0
    )
    return score

# prompt = '123'
# label = {
#     'answer':[[1, 1, 1, 1, 1, 1], [1, 9, 9, 9, 9, 1], [1, 9, 0, 0, 9, 1], [1, 9, 0, 0, 9, 1], [1, 9, 9, 9, 9, 1], [1, 1, 1, 1, 1, 1]],
#     'data_source': 'arc_agi'
# }
# response = '</think><answer>```json\n[[1, 1, 1, 1, 1, 1], [1, 9, 9, 9, 9, 1], [1, 9, 0, 0, 9, 1], [1, 9, 0, 0, 9, 1], [1, 9, 9, 9, 9, 1], [1, 1, 1, 1, 1, 1]]\n```</answer>'

def blocking_function():
    print("开始执行阻塞函数")
    import time
    time.sleep(2)  # 模拟耗时操作
    print("阻塞函数执行完毕")
    return "结果"

async def main():
    print("开始异步函数")
    
    # 将阻塞函数放到线程中执行，避免阻塞事件循环
    result = await math_verify_server(prompt, response, label, 'score')
    # result = await math_score(prompt, response, label, 'stop', '<|text_pad|>')
    # result = await zebralogic_score(prompt, response, label, 'stop', '<|text_pad|>')
    # result = await embedding_server(prompt, response, label, max_retries=3)
    # result = await synlogic_score_fn(prompt, response, label)
    # result = await synlogic_score(prompt, response, label, 'stop', '<|text_pad|>')
    code = """# Function to convert a base 12 number (as a string) to base 10, handling \'A\' and \'B\'\ndef base12_to_base10(num_str):\n    base10_num = 0\n    for power, digit in enumerate(reversed(num_str)):\n        if digit == \'A\':\n            digit_value = 10\n        elif digit == \'B\':\n            digit_value = 11\n        else:\n            digit_value = int(digit)\n        base10_num += digit_value * (12 ** power)\n    return base10_num\n\n# Given base 12 numbers\nnum1_base12 = "8A2"\nnum2_base12 = "3B7"\n\n# Convert to base 10\nnum1_base10 = base12_to_base10(num1_base12)\nnum2_base10 = base12_to_base10(num2_base12)\n\n# Add the base 10 numbers\nsum_base10 = num1_base10 + num2_base10\n\n# Convert the sum back to base 12\nsum_base12 = base10_to_base12(sum_base10)\nprint(sum_base12)\n"""
    # result = await remote_compile(code, '1')
    # result = await math_verify_async()
    # result = await xverify_server(prompt, response, label, max_retries=10, temperature=0.0, max_tokens=8192)
    print(f"获取结果: {result}")
    
    # 继续执行其他异步操作
    # await asyncio.sleep(1)
    print("异步函数执行完毕")

# 普通脚本环境
import time
start = time.time()
asyncio.run(main())
print(time.time() - start)


# url = 'http://10.39.7.131:1000/math_verify'
# # url = 'http://10.39.0.143:2031/compile_python'
# # url = 'http://10.39.11.118:1000/compile_python'
# headers = {
#         "Content-Type": "application/json",
#     }
# # code = "#defining the equation\nfrom sympy import symbols, Eq, solve\n\nP = symbols('P')\nequation = Eq(1.62*P - 96, 1.3*P)\n\n#solving the equation\nsolution = solve(equation, P)\npurchase_price = solution[0]\npurchase_price"
# total = []
# import time
# start = time.time()
# from tqdm import tqdm
# # for key in DATA_DICT:
# import requests
# s = requests.post(url, json=({
#     'response': 'The answer is 523',
#     'gold_ans': '\\boxed{523}',
#     'prompt': 'xxxx'
# }), headers=headers)
# print(s)
# # break
# print(s.json(), url)
    
# time.time()-start