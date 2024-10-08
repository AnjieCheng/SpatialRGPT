import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import openai
from tqdm import tqdm


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    """A dictionary of running averages."""

    def __init__(self):
        self._dict = dict(
            a1=RunningAverage(),
            a2=RunningAverage(),
            a3=RunningAverage(),
            abs_rel=RunningAverage(),
            rmse=RunningAverage(),
            log_10=RunningAverage(),
            rmse_log=RunningAverage(),
            silog=RunningAverage(),
            sq_rel=RunningAverage(),
        )

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err**2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(
        a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log, silog=silog, sq_rel=sq_rel
    )


def evaluate_quan_dir_question(question, answer, response):

    prompt = f"""
You should help me to evaluate the response given the question and the correct answer.
You need to extract the direction of the correct answer and response.
You should output two integers in clock directions, one for the answer, and one for the response.
The output should be in JSON format.


Example 1:
Question: If you are at Region [0], where will you find Region [1]?
Answer: Region [0] will find Region [1] around the 2 o'clock direction.
Response: If you are at Region [0], you will find Region [1] around the 2 o'clock direction.
"answer_direction": 2, "response_direction": 2

Example 2:
Question: If you are at Region [0], where will you find Region [1]?
Answer: Region [0] will find Region [1] around the 12 o'clock direction.
Response: If you are at Region [0], you will find Region [1] around the 11 o'clock direction.
"answer_direction": 12, "response_direction": 11

Your Turn:
Question: {question}
Answer: {answer}
Response: {response}

"""
    client = openai
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt.format(question, answer, response)},
        ],
    )
    json_response = json.loads(response.choices[0].message.content)
    return json_response


def evaluate_quan_dist_question(question, answer, response):

    prompt = f"""
You should help me to evaluate the response given the question and the correct answer.
You need to convert the distance of the correct answer and response to meters. The conversion factors are as follows: 1 inch = 0.0254 meters. 1 foot = 0.3048 meters. 1 centimeter (cm) = 0.01 meters.
You should output two floats in meters, one for the answer, and one for the response.
The output should be in JSON format.


Example 1:
Question: Measure the width of Region [0].?
Answer: The width of Region [0] is 1.02 meters.
Response: Region [0] is 2.17 meters wide.
"answer_in_meters": 1.02, "response_in_meters": 2.17

Example 2:
Question: What is the height of Region [1]?
Answer: The height of Region [1] is 10.0 inches.
Response: It is 48.47 centimeters.
"answer_in_meters": 0.25, "response_in_meters": 0.48

Example 3:
Question: What is the radius of Region [0]?
Answer: Region [0] is 77.56 centimeters wide.
Response: It is 35.9 inches wide.
"answer_in_meters": 0.78, "response_in_meters": 0.91

Example 4:
Question: How far away is Region [0] from Region [1]?
Answer: Region [0] and Region [1] are 11.4 meters apart.
Response: 22.32 feet.
"answer_in_meters": 11.4, "response_in_meters": 6.80

Your Turn:
Question: {question}
Answer: {answer}
Response: {response}

"""
    client = openai
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt.format(question, answer, response)},
        ],
    )
    json_response = json.loads(response.choices[0].message.content)
    return json_response


def evaluate_qual_question(question, answer, response, category):

    if "choice" in category:
        prompt_cat = "choice"
    else:
        prompt_cat = category
    example_path = Path(f"./scripts/srgpt/eval/prompt_templates/{prompt_cat}.txt")
    with example_path.open("r") as f:
        examples = f.read().strip()

    prompt = f"""
You should help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 0 and 1.
1 means that the response perfectly matches the answer.
0 means that the response is completely different from the answer.
The output should be in JSON format.

"""

    post_fix = f"""

Your Turn:
Question: {question}
Answer: {answer}
Response: {response}
    """

    content = prompt + examples.format(question=question) + post_fix.format(question, answer, response)
    client = openai
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": content},
        ],
    )
    json_response = json.loads(response.choices[0].message.content)["your_mark"]
    return json_response


data_path = sys.argv[1]

with open(data_path) as f:
    lines = f.readlines()

total = len(lines)

qualitative_dict = defaultdict(list)
quantitative_success_dict = defaultdict(list)
quantitative_error_dict = defaultdict(list)

raw_list = []

match_fail_count = 0

for line in tqdm(lines):
    match_success = False
    data = json.loads(line)

    data["llm_match_info"] = {}

    if data["qa_info"]["type"] == "quantitative":
        category = data["qa_info"]["category"]
        if category in [
            "vertical_distance_data",
            "horizontal_distance_data",
            "distance_data",
            "width_data",
            "height_data",
            "direction",
        ]:

            if category == "direction":
                try:
                    llama_evaluation = evaluate_quan_dir_question(
                        question=data["question"], answer=data["gt"], response=data["pred"]
                    )
                    answer_in_clock, response_in_clock = int(llama_evaluation["answer_direction"]), int(
                        llama_evaluation["response_direction"]
                    )
                    diff = abs(answer_in_clock - response_in_clock)
                    error_rate = min(diff, 12 - diff)
                    success = error_rate <= 1
                    match_success = True
                except:
                    answer_in_clock = response_in_clock = "N/A"
                    match_success = False
                    match_fail_count += 1
                    success = 0

                data["llm_match_info"]["answer"] = answer_in_clock
                data["llm_match_info"]["response"] = response_in_clock

            else:
                try:
                    llama_evaluation = evaluate_quan_dist_question(
                        question=data["question"], answer=data["gt"], response=data["pred"]
                    )
                    answer_in_meters, response_in_meters = float(llama_evaluation["answer_in_meters"]), float(
                        llama_evaluation["response_in_meters"]
                    )
                    success = (response_in_meters <= (1.25 * answer_in_meters)) and (
                        response_in_meters >= (0.75 * answer_in_meters)
                    )
                    error_rate = (np.abs(response_in_meters - answer_in_meters)) / (answer_in_meters + 1e-4)
                    match_success = True

                except:
                    answer_in_meters = response_in_meters = "N/A"
                    match_success = False
                    match_fail_count += 1
                    success = 0

                data["llm_match_info"]["answer"] = answer_in_meters
                data["llm_match_info"]["response"] = response_in_meters

            if match_success:
                if category == "vertical_distance_data":
                    quantitative_success_dict["vertical_distance"].append(int(success))
                    quantitative_error_dict["vertical_distance"].append(error_rate)
                elif category == "horizontal_distance_data":
                    quantitative_success_dict["horizontal_distance"].append(int(success))
                    quantitative_error_dict["horizontal_distance"].append(error_rate)
                elif category == "distance_data":
                    quantitative_success_dict["direct_distance"].append(int(success))
                    quantitative_error_dict["direct_distance"].append(error_rate)
                elif category == "width_data":
                    quantitative_success_dict["width"].append(int(success))
                    quantitative_error_dict["width"].append(error_rate)
                elif category == "height_data":
                    quantitative_success_dict["height"].append(int(success))
                    quantitative_error_dict["height"].append(error_rate)
                elif category == "direction":
                    quantitative_success_dict["direction"].append(int(success))
                    quantitative_error_dict["direction"].append(error_rate)

            else:
                if category == "vertical_distance_data":
                    quantitative_success_dict["vertical_distance"].append(int(success))
                elif category == "horizontal_distance_data":
                    quantitative_success_dict["horizontal_distance"].append(int(success))
                elif category == "distance_data":
                    quantitative_success_dict["direct_distance"].append(int(success))
                elif category == "width_data":
                    quantitative_success_dict["width"].append(int(success))
                elif category == "height_data":
                    quantitative_success_dict["height"].append(int(success))
                elif category == "direction":
                    quantitative_success_dict["direction"].append(int(success))

    elif data["qa_info"]["type"] == "qualitative":

        category = data["qa_info"]["category"]
        try:
            llama_evaluation = evaluate_qual_question(
                question=data["question"], answer=data["gt"], response=data["pred"], category=category
            )

            if llama_evaluation is None:
                print("Got None from evaluation")
                success = 0
                llama_evaluation = 0

            data["llm_match_info"]["evaluation"] = int(llama_evaluation)
            match_success = True
        except:
            data["llm_match_info"]["evaluation"] = "N/A"
            match_success = False
            match_fail_count += 1
            success = 0

        # if match_success:
        if "short" in category or "tall" in category:
            qualitative_dict["tall/short"].append(int(llama_evaluation > 0))
        elif "left" in category or "right" in category:
            qualitative_dict["left/right"].append(int(llama_evaluation > 0))
        elif "below" in category or "above" in category:
            qualitative_dict["below/above"].append(int(llama_evaluation > 0))
        elif "behind" in category or "front" in category:
            qualitative_dict["behind/front"].append(int(llama_evaluation > 0))
        elif "big" in category or "small" in category:
            qualitative_dict["big/small"].append(int(llama_evaluation > 0))
        elif "tall" in category or "short" in category:
            qualitative_dict["long/short"].append(int(llama_evaluation > 0))
        elif "wide" in category or "thin" in category:
            qualitative_dict["wide/thin"].append(int(llama_evaluation > 0))
        else:
            raise ValueError(f"{category} not found")

    raw_list.append(data)

result_dict = {}
total_qualitative = 0
correct_qualitative = 0
for qual_cat in qualitative_dict.keys():
    correct_qualitative += np.sum(qualitative_dict[qual_cat])
    total_qualitative += len(qualitative_dict[qual_cat])
    # print(f"Qual {qual_cat} [{len(qualitative_dict[qual_cat])}] Acc: {np.sum(qualitative_dict[qual_cat]) / len(qualitative_dict[qual_cat]) * 100}")
    result_dict[f"Qual_{qual_cat}_acc"] = np.sum(qualitative_dict[qual_cat]) / len(qualitative_dict[qual_cat]) * 100
# print('Qual Overall Acc:', correct_qualitative / total_qualitative * 100)
result_dict[f"Qual_overall_acc"] = correct_qualitative / total_qualitative * 100

total_quantitative = 0
correct_quantitative = 0
accum_error = 0

for quant_cat in quantitative_success_dict.keys():
    correct_quantitative += np.sum(quantitative_success_dict[quant_cat])
    accum_error += np.sum(quantitative_error_dict[quant_cat])
    total_quantitative += len(quantitative_success_dict[quant_cat])
    # print(f"Quan {quant_cat} [{len(quantitative_success_dict[quant_cat])}] Err: {np.sum(quantitative_error_dict[quant_cat]) / len(quantitative_error_dict[quant_cat]) * 100}")
    # print(f"Quan {quant_cat} [{len(quantitative_success_dict[quant_cat])}] Acc: {np.sum(quantitative_success_dict[quant_cat]) / len(quantitative_success_dict[quant_cat]) * 100}")

    result_dict[f"Quan_{quant_cat}_acc"] = (
        np.sum(quantitative_success_dict[quant_cat]) / len(quantitative_success_dict[quant_cat]) * 100
    )
    result_dict[f"Quan_{quant_cat}_err"] = (
        np.sum(quantitative_error_dict[quant_cat]) / len(quantitative_error_dict[quant_cat]) * 100
    )

##############

final_metrics = defaultdict(RunningAverageDict)
total = len(raw_list)

for data in raw_list:
    match_success = False

    if data["qa_info"]["type"] == "quantitative":
        category = data["qa_info"]["category"]
        if category in [
            "vertical_distance_data",
            "horizontal_distance_data",
            "distance_data",
            "width_data",
            "height_data",
        ]:

            GT = data["llm_match_info"]["answer"]
            PRED = data["llm_match_info"]["response"]

            try:
                final_metrics[category].update(compute_errors(np.array([GT])[None], np.array([PRED])[None]))
            except:
                continue

    elif data["qa_info"]["type"] == "qualitative":
        continue

for qual_cat in final_metrics.keys():
    final_metrics[qual_cat] = {k: round(v, 3) for k, v in final_metrics[qual_cat].get_value().items()}
    result_dict[f"Quan_{qual_cat}_absrel"] = final_metrics[qual_cat]["abs_rel"]

data_path_parent_dir = os.path.dirname(data_path)

##############

print(f"Below/Above: {result_dict[f'Qual_below/above_acc']}")
print(f"Left/Right: {result_dict[f'Qual_left/right_acc']}")
print(f"Big/Small: {result_dict[f'Qual_big/small_acc']}")
print(f"Tall/Short: {result_dict[f'Qual_tall/short_acc']}")
print(f"Wide/Thin: {result_dict[f'Qual_wide/thin_acc']}")
print(f"Behind/Front: {result_dict[f'Qual_behind/front_acc']}")


print(f'Direct Distance: {result_dict[f"Quan_direct_distance_acc"]} / {final_metrics["distance_data"]["abs_rel"]}')
print(
    f'Horizontal Distance: {result_dict[f"Quan_horizontal_distance_acc"]} / {final_metrics["horizontal_distance_data"]["abs_rel"]}'
)
print(
    f'Vertical Distance: {result_dict[f"Quan_vertical_distance_acc"]} / {final_metrics["vertical_distance_data"]["abs_rel"]}'
)
print(f'Width: {result_dict[f"Quan_width_acc"]} / {final_metrics["width_data"]["abs_rel"]}')
print(f'Height: {result_dict[f"Quan_height_acc"]} / {final_metrics["height_data"]["abs_rel"]}')
print(f'Direction: {result_dict[f"Quan_direction_acc"]} / {result_dict["Quan_direction_err"] / 100 * 360 / 12}')
print("--------------")
print("Qual Overall Acc:", correct_qualitative / total_qualitative * 100)
print("Quan Overall Acc:", correct_quantitative / total_quantitative * 100)
print("Match Fail Count:", match_fail_count)
print("--------------")

result_dict[f"Quan_overall_acc"] = correct_quantitative / total_quantitative * 100

result_dict["Match_fail_count"] = match_fail_count

data_path_parent_dir = os.path.dirname(data_path)

result_dict_path = os.path.join(data_path_parent_dir, "score.json")
with open(result_dict_path, "w") as outfile:
    json.dump(result_dict, outfile)

result_raw_path = os.path.join(data_path_parent_dir, "raw_evaluation.json")
with open(result_raw_path, "w") as outfile:
    json.dump(raw_list, outfile)
