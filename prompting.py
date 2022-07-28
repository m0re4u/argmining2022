import argparse
import os
import openai
from dotenv import load_dotenv
from data import SharedTaskData

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

def get_prompt(topic: str, premise: str, conclusion: str, prompt_style: int = 0) -> str:
    if prompt_style == 0:
        prompt = f"""topic {topic}
premise {premise}
conclusion {conclusion}
Is there a logical inference that links the premise to the conclusion?
"""

    return prompt



def main(args):
    train_data = SharedTaskData("TaskA_train.csv")
    dev_data = SharedTaskData("TaskA_dev.csv")
    sample = dev_data.df.iloc[0]
    # TODO: split novelty and validity
    x = get_prompt(sample['topic'], sample['Premise'], sample['Conclusion'])
    print(x)

    prediction = gpt3(x)
    print(prediction)

    # TODO: parse prediction into yes/no classes
    # TODO: compare with label




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
