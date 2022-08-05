import argparse
import json

import pandas as pd
from pathlib import Path
from data import SharedTaskData

PREDICTION_SOURCES = [
    "prompt-only",
    "supervised-mtl-only",
    "supervised-mtl-novelty-prompt-validity"
]


def load_prompt_preds(test_data, predictions_filename):
    label_mapping = {
        'novel': 1,
        'valid': 1,
        'not-novel': -1,
        'not-valid': -1,
    }
    with open(predictions_filename, 'r') as f:
        pred_data = json.load(f)

    records = []
    for test_sample, prediction in zip(test_data, pred_data):
        records.append({
            'topic': test_sample['topic'],
            'Premise': test_sample['premise'],
            'Conclusion': test_sample['conclusion'],
            'predicted validity': label_mapping[prediction['pred_val']],
            'predicted novelty': label_mapping[prediction['pred_nov']],
        })
    df = pd.DataFrame.from_records(records)
    return df


def load_mtl_preds(test_data, predictions_filename):
    label_mapping = {0: -1, 1: 1}
    with open(predictions_filename, 'r') as f:
        pred_data = json.load(f)
    records = []
    for i, test_sample in enumerate(test_data):
        records.append({
            'topic': test_sample['topic'],
            'Premise': test_sample['premise'],
            'Conclusion': test_sample['conclusion'],
            'predicted validity': label_mapping[pred_data['validity'][i]],
            'predicted novelty': label_mapping[pred_data['novelty'][i]],
        })
    df = pd.DataFrame.from_records(records)
    return df


def load_mixed_preds(test_data, prediction_source, predictions_filename1, predictions_filename2):
    if prediction_source == 'supervised-mtl-novelty-prompt-validity':
        preds_mtl = load_mtl_preds(test_data, predictions_filename1)
        preds_prompting = load_prompt_preds(test_data, predictions_filename2)
        # Keep predicted novelty from preds_mtl and predicted validity from preds_prompting
        preds_mtl['predicted validity'] = preds_prompting['predicted validity']

    return preds_mtl


def get_approach_title(prediction_source):
    if prediction_source == "prompt-only":
        return "GPT-3 few-shot prompt engineering"
    elif prediction_source == "supervised-mtl-only":
        return "Supervised Multi-Task Learning using pretrained Transformers and contrastive learning"
    elif prediction_source == "supervised-mtl-novelty-prompt-validity":
        return "Mixed GPT-3 and Multi-Task Learning"

def get_approach_abstract(prediction_source):
    if prediction_source == "prompt-only":
        return """
In our prompt-engineering approach, we used OpenAI's GPT-3 (https://beta.openai.com/playground) for few-shot classification of novelty and validity labels. We construct a prompt by concatenating topic, premise and conclusions in a structured format, and request either a novelty or validity label at the time in separate prompts. In addition, we show four static examples before asking a label from the model, selected from short, hard examples (=have only majority annotation agreement) in the training dataset.
        """
    elif prediction_source == "supervised-mtl-only":
        return """
Our supervised approach uses Multi-Task Learning (MTL) for predicting novelty and validity labels. The model consists of a shared encoder with task-specific classification heads (single layer). As input, we feed topic, premise and conclusion, and switch uniformly at random during training between the novelty and validity task. In this particular version, we use a pretrained RoBERTa on NLI datasets as starting point, run an intermediate training procedure, followed by finetuning using MTL on the training data. As an intermediate step we implemented contrastive learning. Our model, roberta-large-mnli (RoBERTa large fine-tuned on the Multi-Genre Natural Language Inference (MNLI) corpus), is optimized with triplet loss according to the contrastive learning framework from [SimCSE](https://github.com/princeton-nlp/SimCSE). As data, we grouped the training data by premise and novelty, and created premise, positive-novelty-conclusion and negative-novelty-conclusion triples.

        """
    elif prediction_source == "supervised-mtl-novelty-prompt-validity":
        return """
For our mixed approach, we combine two models for predictions of the tasks separately: we use OpenAI's GPT-3 for the classification of validity labels, which is trained in the same way as the model in our GPT-3 few-shot prompt engineering approach, and roberta-large-mnli in a contrastive setting for the classification of the novelty labels. We choose these two methods as we observed them to achieve high results on their respective task.
        """

def get_extra_data_description(prediction_source):
    if prediction_source == "prompt-only":
        return "No extra data was used"
    elif prediction_source == "supervised-mtl-only":
        return "No extra data was used"

def prepare_email(prediction_source):
    subject_task = "[ArgMining22-SharedTask-SubtaskA]"
    team_name = "CLTeamL"
    approach_number = PREDICTION_SOURCES.index(prediction_source) + 1
    email = f"""
SUBJECT: "{subject_task} {team_name} ({approach_number})"


Team name: CLTeamL
Team members:
    - Name: Michiel van der Meer (Main contact)
        Email: m.t.van.der.meer@liacs.leidenuniv.nl
        Affiliation: Leiden Institute of Advanced Computer Science, Hybrid Intelligence Consortium
        Webpage: https://liacs.leidenuniv.nl/~meermtvander/
    - Name: Myrthe Reuver
        Email: myrthe.reuver@vu.nl
        Affiliation: Vrije Universiteit Amsterdam
        Webpage: https://myrthereuver.github.io/
    - Name: Urja Khurana
        Email: u.khurana@vu.nl
        Affiliation: Vrije Universiteit Amsterdam, Hybrid Intelligence Consortium
        Webpage: https://urjakh.github.io/
    - Name: Lea Krause
        Email: l.krause@vu.nl
        Affiliation: Vrije Universiteit Amsterdam, Hybrid Intelligence Consortium
        Webpage: https://lkra.github.io/
    - Name: Selene Báez Santamaría
        Email: s.baezsantamaria@vu.nl
        Affiliation: Vrije Universiteit Amsterdam, Hybrid Intelligence Consortium
        Webpage: https://selbaez.github.io/
Approach title: {get_approach_title(prediction_source)}
Abstract: {get_approach_abstract(prediction_source)}
Extra training data: {get_extra_data_description(prediction_source)}


predictions are attached.

Kind regards,

Michiel van der Meer
    """
    return email


def write_submission_files(prediction_df, email, prediction_source):
    out_folder = Path(f"submissions/{prediction_source}/")
    out_folder.mkdir(parents=True, exist_ok=True)
    csv_out_file = out_folder / "predictions.csv"
    prediction_df.to_csv(csv_out_file, index=False)

    email_out_file = out_folder / "email.txt"
    with open(email_out_file, 'w') as f:
        f.write(email)


def main(args):
    test_data = SharedTaskData('TaskA_test-without-labels.csv', test_set=True)

    email = prepare_email(args.prediction_source)
    if args.prediction_source == 'prompt-only':
        preds = load_prompt_preds(test_data, args.prediction_file)
    elif args.prediction_source == 'supervised-mtl-only':
        preds = load_mtl_preds(test_data, args.prediction_file)
    elif args.prediction_source == 'supervised-mtl-novelty-prompt-validity':
        preds = load_mixed_preds(test_data, args.prediction_source, args.prediction_file, args.second_prediction_file)

    write_submission_files(preds, email, args.prediction_source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_source', default='prompt-only', type=str, choices=PREDICTION_SOURCES,
                        help="Which method was used for creating the predictions")
    parser.add_argument('prediction_file', default=None, type=str,
                        help="File to load predictions from")
    parser.add_argument('--second_prediction_file', default=None, type=str,
                        help="File to load second set of predictions from for MIXED models")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
