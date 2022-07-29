import argparse
import json
import os
import time

import numpy as np
import openai
from dotenv import load_dotenv
from tqdm import tqdm

from baseline import print_results
from data import SharedTaskConstants, SharedTaskData

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def gpt3(prompts, model="text-davinci-002"):
    """ functions to call GPT3 predictions """
    response = openai.Completion.create(
        model=model,
        prompt=prompts,
        temperature=0,
        max_tokens=50,
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


def get_examples(n: int, example_data, prompt_style: int, goal: str) -> str:
    """
    Get n example prompts from example_data dataset, using `prompt_style` and `goal`
    """
    shown_example_idx = np.random.randint(0, len(example_data), size=n)
    example_prompts = []
    for idx in shown_example_idx:
        sample, label_val, label_nov = example_data[idx]
        if goal == 'validity':
            label = label_val
        elif goal == 'novelty':
            label = label_nov
        example_prompts.append(get_prompt_text(
            sample['topic'],
            sample['premise'],
            sample['conclusion'],
            prompt_style=prompt_style,
            goal=goal,
            label=label))
    return "\n\n".join(example_prompts)


def get_prompt(sample, prompt_style: int, goal: str):
    """
    Helper function for getting a prompt given a dataset instance.
    """
    return get_prompt_text(sample['topic'], sample['premise'], sample['conclusion'], prompt_style=prompt_style,
                           goal=goal)


def get_prompt_text(topic: str,
                    premise: str,
                    conclusion: str,
                    prompt_style: int = 0,
                    goal: str = 'novelty',
                    label: str = None) -> str:
    """ Function to create a prompt """

    if goal != "novelty" and goal != "validity":
        raise ValueError(f"Unknown prompt task {goal}")

    task_prompt = {}
    # Baseline
    if prompt_style == 0:
        task_prompt['novelty'] = "Is there a presence of novel premise-related content? "
        task_prompt['validity'] = "Is there a strict logical inference that links the premise to the conclusion? "
        prompt = f"""topic {topic}
premise {premise}
conclusion {conclusion}
{task_prompt[goal]}
"""

    # Capitalized and colon
    elif prompt_style == 1:
        task_prompt['novelty'] = "Is there a presence of novel premise-related content? "
        task_prompt['validity'] = "Is there a strict logical inference that links the premise to the conclusion? "
        prompt = f"""TOPIC: {topic}
PREMISE: {premise}
CONCLUSION: {conclusion}
{task_prompt[goal]}
"""

    # Dynamic prompts
    elif prompt_style == 2:
        task_prompt['novelty'] = "NOVELTY:"
        task_prompt['validity'] = "VALIDITY:"
        prompt = f"""TOPIC: {topic}
PREMISE: {premise}
CONCLUSION: {conclusion}
{task_prompt[goal]}"""

    # Static merged prompts
    elif prompt_style == 3:
        prompt = f"""TOPIC: Torture
PREMISE: A terrorist will experience pain for a short period while being tortured. Yet, the millions of lives that could be lost if that pain is not inflicted will be gone forever. The ethical trade-off is overwhelmingly in favor of performing torture.
CONCLUSION: Pain of torture is worth suffering
VALIDITY:not-valid, NOVELTY:novel

TOPIC: United Nations Standing Army
PREMISE:especially if it were to include purchase of air and sea transport to reach theatres of operation, added to the high costs of permanent establishment and training, and equipping the force for every possible type of terrain. At present the UN can draw upon different kind of troops for different kinds of missions from whatever member states feel best equipped to deal with a particular situation.
CONCLUSION: The cost of such an army would be very high,
VALIDITY:valid, NOVELTY:not-novel

TOPIC: {topic}
PREMISE: {premise}
CONCLUSION: {conclusion}"""

    else:
        raise ValueError(f"Unknown prompt style: {prompt_style}")

    if label is not None:
        if prompt_style == 0 or prompt_style == 1:
            label_str = 'no' if 'not' in label else 'yes'
        elif prompt_style == 2:
            label_str = label

        prompt += label_str

    return prompt


def parse_response(api_response, label: str, prompt_style: int):
    """
    Parse a response from OpenAI GPT for a particular task (defined by `label`) and `prompt_style`
    """
    first_5_tokens = api_response['tokens'][:5]
    if prompt_style == 0 or prompt_style == 1:
        if any([x.lower() == 'yes' for x in first_5_tokens]):
            return label
        else:
            return f"not-{label}"

    elif prompt_style in [2, 3]:
        if f"not-{label}" in api_response['text']:
            return f"not-{label}"
        else:
            return label

    else:
        raise ValueError(f"Can't parse GPT response for prompt_style: {prompt_style}")


def main(args):
    train_data = SharedTaskData("TaskA_train.csv")
    dev_data = SharedTaskData("TaskA_dev.csv")

    # Prepare storage of labels and dumpables
    y_true = {'novelty': [], 'validity': []}
    y_pred = {'novelty': [], 'validity': []}
    dump_data = []
    for sample, y_val, y_nov in tqdm(dev_data):
        # Get true labels
        y_true['validity'].append(SharedTaskConstants.local_str_mapping[y_val])
        y_true['novelty'].append(SharedTaskConstants.local_str_mapping[y_nov])

        # Get the prompt for asking about the sample
        x_val = get_prompt(sample, prompt_style=args.prompt_style, goal='validity')
        x_nov = get_prompt(sample, prompt_style=args.prompt_style, goal='novelty')
        if args.n_shot > 0 and args.prompt_style != 3:
            # Get examples
            example_prompt_val = get_examples(args.n_shot, train_data, prompt_style=args.prompt_style, goal='validity')
            example_prompt_nov = get_examples(args.n_shot, train_data, prompt_style=args.prompt_style, goal='novelty')
            # Prepend examples
            x_val = f"{example_prompt_val}\n\n{x_val}"
            x_nov = f"{example_prompt_nov}\n\n{x_nov}"

        if args.prompt_style != 3:
            # Pass through model
            response = gpt3([x_val, x_nov])
            assert len(response) == 2
            response_val, response_nov = response

        else:
            # Pass through model
            response = gpt3([x_val])
            [response_val], [response_nov] = response, response

        # Parse responses
        pred_val = parse_response(response_val, 'valid', prompt_style=args.prompt_style)
        y_pred['validity'].append(SharedTaskConstants.local_str_mapping[pred_val])
        pred_nov = parse_response(response_nov, 'novel', prompt_style=args.prompt_style)
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

        # Ugly hack to adhere to rate limit (1 / s)
        if args.prompt_style in [2, 3]:
            time.sleep(1)

    # Write results to file as backup
    with open(f'predictions/dump_prompt_{args.prompt_style}_results.json', 'w') as f:
        json.dump(dump_data, f)

    print_results(f"Prompting (style {args.prompt_style})", y_true, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_style', default=0, type=int, choices=[0, 1, 2, 3],
                        help="Which prompt style to use: "
                             "0 baseline, 1 capitalized, 2 random dynamic examles, 3 static examples, merged prompts")
    parser.add_argument('--n_shot', default=0, type=int,
                        help="How many examples to give in the prompt")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
