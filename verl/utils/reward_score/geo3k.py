# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re

from mathruler.grader import extract_boxed_content, grade_answer

import requests
import pickle
import json
from typing import List, Dict, Any, Union

import logging
logger_p = logging.getLogger('my_logger')
logger_p.setLevel(logging.INFO)
file_handler = logging.FileHandler('./log/log_test.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger_p.addHandler(file_handler)


def get_multimodal_predictions(
    prompts: List[str],
    image_paths_list: List[List[str]],
    n: int,
    server_url: str = "http://127.0.0.1:5000/generate_multimodal"
) -> Dict[str, Any]:
    """
    Sends requests to the multimodal VLLM server to get 'n' predictions for each prompt.

    Args:
        prompts (List[str]): A list of prompt strings containing the <image> placeholder.
        image_paths_list (List[List[str]]): A nested list where each inner list contains
                                            local file paths of images corresponding to the prompt.
        n (int): The number of desired answers to generate for each prompt.
        server_url (str): The URL of the running server.

    Returns:
        Dict[str, Any]: The JSON response returned from the server, typically containing prediction results.
    """
    if len(prompts) != len(image_paths_list):
        raise ValueError("The number of prompts must equal the number of image_paths_list.")

    headers = {"Content-Type": "application/json"}
    payload = {
        "prompts": prompts,
        "image_paths": image_paths_list,
        "n": n
    }

    try:
        # Increase timeout because model generation may take some time
        response = requests.post(server_url, headers=headers, data=json.dumps(payload), timeout=300)
        response.raise_for_status()  # Raise HTTPError if the response status code is 4xx or 5xx
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the server: {e}")
        return {"error": str(e)}


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)

    if not match_result:
        return 0.0

    think_start_count = predict_str.count("<think>")
    think_end_count = predict_str.count("</think>")

    if think_start_count == 1 and think_end_count == 1:
        return 1.0
    else:
        return 0.0

def format_reward_woboxed(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0
    

def extract_after_think(predict_str: str) -> str:
    cleaned = re.sub(r"</?think>", "", predict_str, flags=re.DOTALL).strip()

    parts = re.split(r"</think>", predict_str, maxsplit=1, flags=re.DOTALL)
    if len(parts) == 2:
        suffix = parts[1]
        if suffix.strip():
            return suffix.strip()
        return cleaned

    return cleaned

def replace_think_tags_simple(text: str) -> str:
    return text.replace('<think>', '[BEGIN THINK]').replace('</think>', '[END THINK]')

def replace_image_tags_simple(text: str) -> str:
    return text.replace('<image>', '[BEGIN IMAGE]').replace('</image>', '[END IMAGE]')


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def format_deepthoughts_input(question:str, deepthought:str):
    deepthought = replace_image_tags_simple(deepthought)
    format_prompt = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}. If the question is multiple-choice (single- or multi-select), put the final answer inside \\boxed{}, and format your answer as a Python list of uppercase letters in single quotes, separated by commas (e.g., \\boxed{['D']} or \\boxed{['A','B']}); otherwise, do not use a list."
    dt_prompt = f"To solve the problem above, you may refer to the expert analysis of the the given information and the problem scenario.\n### Expert Analysis:\n[Analysis Start]\n{deepthought}\n[Analysis End]\n\n"
    return question+dt_prompt+format_prompt


#def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
#    return (1.0 - format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(
#        predict_str
#    )

def compute_score(solution_str: Union[str, List[str]], ground_truth: Union[str, List[str]], **kwargs):
    use_deepthought = kwargs.get("deepthought", False)
    val_flag = kwargs.get("val_flag", False)
    if use_deepthought:
        format_score = 0.1

        extra_info_list = kwargs.get("extra_info")
        image_paths_list = [i['origin_images'] for i in extra_info_list]

        origin_questions_list = [i['question'] for i in extra_info_list]
        deepthoughts_list = [replace_think_tags_simple(i) for i in solution_str]

        prompt_w_deepthoughts_list = [
            format_deepthoughts_input(origin_questions_list[i], deepthoughts_list[i])
            for i in range(len(origin_questions_list))
        ]

        if val_flag:
            freeze_policy_results = get_multimodal_predictions(
                prompts = prompt_w_deepthoughts_list,
                image_paths_list = image_paths_list,
                n = 1, # tradeoff of speed an accuary
            )
        else:
            freeze_policy_results = get_multimodal_predictions(
                prompts = prompt_w_deepthoughts_list,
                image_paths_list = image_paths_list,
                n = 4, # tradeoff of speed an accuary
            )

        if 'predictions' in freeze_policy_results:
            logger_p.info("Get one Results in Geo3k")


        # [[str, str,..., str],...] --> freeze_policy_results
        freeze_policy_results = freeze_policy_results['predictions']
        scores_list = []
        for i in range(len(freeze_policy_results)):
            temp = []
            for j in freeze_policy_results[i]:
                temp.append(acc_reward(j,ground_truth[i]))
            scores_list.append(temp)
        scores_list = [sum(sublist) / len(sublist) for sublist in scores_list]

        return_results = [
            {
                "score": (1.0 - format_score) * scores_list[i] + format_score * format_reward_woboxed(solution_str[i]),
                "acc": (1.0 - format_score) * scores_list[i] + format_score * format_reward_woboxed(solution_str[i]) > 0.5
            } for i in range(len(scores_list))
        ]

        return return_results
    else:
        data_source = kwargs.get("data_source", None) # Through this to identify how to form reward
        format_score = 0.1
        use_boxed = True

        reward = (1.0 - format_score) * acc_reward(solution_str, ground_truth, use_boxed) + format_score * format_reward(solution_str)
        acc = acc_reward(solution_str, ground_truth, use_boxed) > 0.5

        return {
            "score": reward,
            "acc": acc
        }