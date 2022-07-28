import argparse
import os

import openai
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from tqdm import tqdm

from data import SharedTaskData, SharedTaskConstants

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def gpt3(prompts, model="text-davinci-002"):
    """ functions to call GPT3 predictions """
    response = openai.Completion.create(
        model=model,
        prompt=prompts,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=1
    )
    return [
        {
            'prompt': prompt,
            'text': response['text'],
            'tokens': response['logprobs']['tokens'],
            'logprob': response['logprobs']['token_logprobs']
        }
        for response, prompt in zip(response.choices, prompts)
    ]

def get_prompt(topic: str, premise: str, conclusion: str, prompt_style: int = 0, goal: str = 'novelty') -> str:
    if goal == "novelty":
        task_prompt = "Is there a presence of novel premise-related content?"
    elif goal == "validity":
        task_prompt = "Is there a logical inference that links the premise to the conclusion?"
    else:
        raise ValueError(f"Unknown prompt task {goal}")


    if prompt_style == 0:
        prompt = f"""topic {topic}
premise {premise}
conclusion {conclusion}
{task_prompt}
"""
    else:
        raise ValueError(f"Unknown prompt style: {prompt_style}")

    return prompt


def parse_response(api_response, label):
    first_5_tokens = api_response['tokens'][:5]
    if any([x.lower() == 'yes' for x in first_5_tokens]):
        return label
    else:
        return f"not-{label}"


def main(args):
    train_data = SharedTaskData("TaskA_train.csv")
    dev_data = SharedTaskData("TaskA_dev.csv")

    y_true = {'novelty': [], 'validity': []}
    y_pred = {'novelty': [], 'validity': []}

    for sample, y_val, y_nov in tqdm(dev_data):
        y_true['validity'].append(SharedTaskConstants.local_str_mapping[y_val])
        y_true['novelty'].append(SharedTaskConstants.local_str_mapping[y_nov])
        x_val = get_prompt(sample['topic'], sample['premise'], sample['conclusion'], prompt_style=0, goal='novelty')
        x_nov = get_prompt(sample['topic'], sample['premise'], sample['conclusion'], prompt_style=0, goal='validity')

        response = gpt3([x_val, x_nov])
        assert len(response) == 2
        response_val, response_nov = response

        pred_val = parse_response(response_val, 'valid')
        y_pred['validity'].append(SharedTaskConstants.local_str_mapping[pred_val])

        pred_nov = parse_response(response_nov, 'novel')
        y_pred['novelty'].append(SharedTaskConstants.local_str_mapping[pred_nov])

    print("Validity")
    results_validity = classification_report(
        y_true['validity'],
        y_pred['validity'],
        target_names=['not-valid', 'valid'],
        zero_division=0
    )
    print(results_validity)

    print("Novelty")
    results_novelty = classification_report(
        y_true['novelty'],
        y_pred['novelty'],
        target_names=['not-novel', 'novel'],
        zero_division=0
    )
    print(results_novelty)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_style', default=0, type=int, choices=[0],
                        help="Which prompt style to use")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
