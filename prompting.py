import argparse
import json
import os

import openai
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from tqdm import tqdm

from data import SharedTaskConstants, SharedTaskData
from baseline import print_results

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
        task_prompt = "Is there a strict logical inference that links the premise to the conclusion?"
    else:
        raise ValueError(f"Unknown prompt task {goal}")


    if prompt_style == 0:
        prompt = f"""topic {topic}
premise {premise}
conclusion {conclusion}
{task_prompt}
"""
    elif prompt_style == 1:
        prompt = f"""TOPIC: {topic}
PREMISE: {premise}
CONCLUSION: {conclusion}
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

    dump_data = []
    for sample, y_val, y_nov in tqdm(dev_data):
        y_true['validity'].append(SharedTaskConstants.local_str_mapping[y_val])
        y_true['novelty'].append(SharedTaskConstants.local_str_mapping[y_nov])
        x_val = get_prompt(sample['topic'], sample['premise'], sample['conclusion'], prompt_style=args.prompt_style, goal='novelty')
        x_nov = get_prompt(sample['topic'], sample['premise'], sample['conclusion'], prompt_style=args.prompt_style, goal='validity')

        response = gpt3([x_val, x_nov])
        assert len(response) == 2
        response_val, response_nov = response

        pred_val = parse_response(response_val, 'valid')
        y_pred['validity'].append(SharedTaskConstants.local_str_mapping[pred_val])

        pred_nov = parse_response(response_nov, 'novel')
        y_pred['novelty'].append(SharedTaskConstants.local_str_mapping[pred_nov])

        dump_data.append({
            'y_nov': y_nov,
            'y_val': y_val,
            'x_nov': x_nov,
            'x_val': x_val,
            'response_nov': response_nov,
            'response_val': response_val,
            'pred_nov': pred_nov,
            'pred_val': pred_val,
        })

    # Write results to file as backup
    with open(f'dump_prompt_{args.prompt_style}_results.json', 'w') as f:
        json.dump(dump_data, f)

    print_results(f"Prompting (style {args.prompt_style})", y_true, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_style', default=0, type=int, choices=[0, 1],
                        help="Which prompt style to use")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
