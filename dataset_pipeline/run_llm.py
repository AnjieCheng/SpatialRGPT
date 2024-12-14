import argparse
import time
import warnings
import re
import json

from sglang import function, system, gen, set_default_backend, RuntimeEndpoint
from sglang.utils import (
    execute_shell_command,
    wait_for_server,
)

# Suppressing all warnings
warnings.filterwarnings("ignore")


response_regex = r"\{" + r'    "Question": "[\w\d\s<>?,.!]{1,512}",' + r'    "Answer": "[\w\d\s<>?,.!]{1,512}"' + r"\}"


@function
def rephrase_qa(s, question_1):
    s += system(
        r"""
                You are a helpful assistant tasked with generating spatial reasoning-based questions and answers from provided descriptions of scenes. 
                Always craft a question without directly revealing specific details from the description. 
                Always generate questions related to the description using <regionX>. 
                The description should always be used to answer and not leak into the question. 
                When mentioning the objects or regions, use <regionX> instead of the objects or regions. 
                Speak like you are the observer's perspective. 
                Always make sure all the description objects or regions are mentioned with <regionX> in the question. 
                Only mention each <regionX> once.

                Here's several examples:

                [Objects]: <region4> sofa, <region1> chair. [Description]: The path between the <region4> and <region1> is 1.5 meters.
                "Question": You are a cleaning robot that is 1 meter wide. Now you are standing in a living room and see the image; you want to move from here to the door that leads to the backyard. Do you think I can go through the path between the <region4> and <region1>? 
                "Answer": The path between <region4> and <region1> is 1.5 meters, so yes, the robot can go through the path between <region4> and <region1> since it is wider than the robot's width.

                [Objects]: <region2> apple, <region3> orange. [Description]:  <region2> is positioned on the left side of <region3>.
                "Question": You see two fruits, an apple in <region2> and an orange in <region3>. Which one is more on the left side? 
                "Answer": The apple in <region2> is more on the left.

                [Objects]: <region3> desk, <region6> bed. [Description]:  <region3> is further to the viewer than <region6>.
                "Question": You are exploring a bedroom and walking towards <region3> and <region6>. Which one will you reach first? 
                "Answer": You will reach the bed first because it is closer to you than the desk, which is further away.

                [Objects]: <region0> book. [Description]: <region0> is 50 cm in width.
                "Question": You are a librarian currently standing in front of a 40 cm width bookshelf, and you see <region0> that you want to place on the shelf. Can you determine if <region0> will fit on the shelf?
                "Answer":  Answer: <region0> is 50 cm in width, so the shelf is not wide enough to hold a book of that size. Please find a larger shelf.

                Now its your turn!

"""
    )
    s += question_1
    s += gen("json_output", max_tokens=1024, regex=response_regex)


def process_prompt(prompt, rephrase_qa, max_retries=5):
    for attempt in range(max_retries):
        try:
            llama_response = rephrase_qa.run(prompt, temperature=0.2)
            response_string = llama_response["json_output"]

            # Clean and parse the response
            cleaned_string = response_string.strip()
            cleaned_string = "".join(char for char in cleaned_string if ord(char) >= 32 or char == "\n")
            cleaned_string = re.sub(r"\s+", " ", cleaned_string)
            cleaned_string = cleaned_string.replace("'", '"')
            json_response = json.loads(cleaned_string)

            question, answer = json_response["Question"], json_response["Answer"]

            # Cleanup question/answer
            question = question[2:] if question and question[:2] == ". " else question
            answer = answer[2:] if answer and answer[:2] == ". " else answer

            # Validate region tags
            prompt_tags = {x for x in prompt.split() if x.startswith("<region") and x.endswith(">")}
            question_tags = {x for x in question.split() if x.startswith("<region") and x.endswith(">")}
            answer_tags = {x for x in answer.split() if x.startswith("<region") and x.endswith(">")}

            # Check if all validations pass
            if prompt_tags == question_tags and prompt_tags == answer_tags:
                if all(question.count(tag) == 1 for tag in prompt_tags):
                    print(f"Prompt: {prompt}")
                    print(f"Question: {question}")
                    print(f"Answer: {answer}")
                    print("---------------")
                    return True, question, answer
                else:
                    print(f"Attempt {attempt + 1}: skipping because <regionX> appeared >1 times in question")
            else:
                print(f"Attempt {attempt + 1}: skipping because <regionX> miss-matched in question/answer")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {str(e)}")

    print(f"Failed to get valid output after {max_retries} attempts")
    return False, None, None


def main(args):
    """Main function to control the flow of the program."""

    # Launch sglang backend
    server_process = execute_shell_command(
        f"python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-70B-Instruct --port {args.port} --host 0.0.0.0 --tp {args.gpus}"
    )
    wait_for_server(f"http://localhost:{args.port}")
    set_default_backend(RuntimeEndpoint(f"http://localhost:{args.port}"))

    # Read llm_prompts json
    with open(args.llm_prompts_path, "r") as f:
        llm_prompts = json.load(f)

    conversations = []
    for prompt in llm_prompts:
        success, question, answer = process_prompt(prompt, rephrase_qa)
        if success:
            conversations.append((question, answer))

    for sample in conversations:
        print(f"Q: {sample[0]}")
        print(f"A: {sample[1]}")
        print("-----------------------")


def parse_args():
    """Command-line argument parser."""
    parser = argparse.ArgumentParser(description="Generate 3D SceneGraph for an image.")
    parser.add_argument("--config", default="configs/v2.py", help="Annotation config file path.")
    parser.add_argument("--port", default=3000, help="Port for Sglang")
    parser.add_argument("--gpus", default=8, help="Number of gpus")
    parser.add_argument(
        "--llm-prompts-path",
        default="./demo_out/20241125_175649/json/indoor_llm_prompts.json",
        help="Path to llm prompt json.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.timestamp = timestamp
    main(args)
