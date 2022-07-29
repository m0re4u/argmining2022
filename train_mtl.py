import argparse

import torch
import transformers
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, TrainingArguments, set_seed

from data import SharedTaskData
from models import MultitaskModel
from trainers import MultitaskTrainer, NLPDataCollator


class Tokenize:
    def __init__(self, model_name, tensorflows):
        self.model_name: str = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                       from_tf=tensorflows)

    def _tokenize_fn(self, examples):
        batch_size = len(examples['Premise'])
        batched_inputs = [
            examples['topic'][i] + self.tokenizer.sep_token + \
            examples['Premise'][i] + self.tokenizer.sep_token + \
            examples['Conclusion'][i] for i in range(batch_size)
        ]
        return self.tokenizer(batched_inputs, truncation=True, padding=True)

    def tokenize_function_val(self, examples):
        samples = self._tokenize_fn(examples)
        samples['labels'] = examples['validity_str']
        return samples

    def tokenize_function_nov(self, examples):
        samples = self._tokenize_fn(examples)
        samples['labels'] = examples['novelty_str']
        return samples


def single_label_metrics(predictions, labels):
    softmax = torch.nn.Softmax(dim=1)
    preds = torch.Tensor(predictions)
    probs = softmax(preds)
    y_pred = torch.argmax(probs, dim=1)
    y_true = labels
    report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, zero_division=0)
    output_metrics = {
        'macro f1': report['macro avg']['f1-score'],
        'accuracy': report['accuracy'],
        'weighted f1': report['weighted avg']['f1-score'],
    }
    return output_metrics


def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return single_label_metrics(
        predictions=preds,
        labels=p.label_ids
    )


def prepare_data(model_name, train_dataset, dev_dataset, target_task, tokenizer_fn):
    """
    Prepare a training and development dataset for HF training. Includes tokenizations and casting to torch tensors.
    """
    train_dataset = train_dataset.convert_to_hf_dataset(target_task)
    dev_dataset = dev_dataset.convert_to_hf_dataset(target_task, features=train_dataset.features)
    assert train_dataset.features[f'{target_task}_str']._str2int == dev_dataset.features[f'{target_task}_str']._str2int
    tokenized_train_dataset = train_dataset.map(tokenizer_fn, batched=True)
    tokenized_dev_dataset = dev_dataset.map(tokenizer_fn, batched=True)
    if 'roberta' in model_name or 'ArgumentRelation' in model_name:
        column_names = ['input_ids', 'attention_mask', 'labels']
    else:
        column_names = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    tokenized_train_dataset.set_format(type='torch', columns=column_names)
    tokenized_dev_dataset.set_format(type='torch', columns=column_names)
    return tokenized_train_dataset, tokenized_dev_dataset


def main(
        use_model: str = "bert-base-uncased",
        seed: int = 0,
        tensorflows: bool = False,
        eval_only: bool = False,
        learning_rate: float = 5e-05,
        epochs: int = 10
    ):
    set_seed(seed)
    tokenize = Tokenize(use_model, tensorflows)

    # torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(config["pipeline"]["seed"])
    torch.backends.cudnn.deterministic = True

    train_data = SharedTaskData("TaskA_train.csv")
    dev_data = SharedTaskData("TaskA_dev.csv")

    tokenized_train_dataset_novelty, tokenized_dev_dataset_novelty = prepare_data(
        use_model,
        train_data,
        dev_data,
        "novelty",
        tokenize.tokenize_function_nov
    )
    tokenized_train_dataset_validity, tokenized_dev_dataset_validity = prepare_data(
        use_model,
        train_data,
        dev_data,
        "validity",
        tokenize.tokenize_function_val
    )

    if "ArgumentRelation" in use_model:
        kwargs = {'pad_token_id': 1}
    else:
        kwargs = {}

    # Create model
    multitask_model = MultitaskModel.create(
        model_name=use_model,
        model_type_dict={
            "novelty": transformers.AutoModelForSequenceClassification,
            "validity": transformers.AutoModelForSequenceClassification,
        },
        model_config_dict={
            "novelty": transformers.AutoConfig.from_pretrained(use_model, num_labels=2, **kwargs),
            "validity": transformers.AutoConfig.from_pretrained(use_model, num_labels=2, **kwargs),
        },
        tensorflows=tensorflows
    )

    # Combine datasets
    train_dataset = {
        "novelty": tokenized_train_dataset_novelty,
        "validity": tokenized_train_dataset_validity,
    }

    val_dataset = {
        "novelty": tokenized_dev_dataset_novelty,
        "validity": tokenized_dev_dataset_validity,
    }

    # Arguments for training loop
    training_args = TrainingArguments(
        f"hftrainer_am_mtl_{use_model}_{epochs}_{learning_rate}_{seed}",
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        report_to="wandb",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        label_names=['labels']
    )

    # Go!
    trainer = MultitaskTrainer(
        model=multitask_model,
        args=training_args,
        data_collator=NLPDataCollator(tokenizer=tokenize.tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    if eval_only:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_model', default=None, type=str,
                        help="Which model to load")
    parser.add_argument('--tensorflows', default=False, type=bool,
                        help="enable loading from Tensorflow models")
    parser.add_argument('--eval_only', default=False, action='store_true',
                        help="Whether to do training or evaluation only")
    parser.add_argument('--seed', '-s', default=0, type=int, help="Set seed for reproducibility.")
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs to train for.")
    parser.add_argument('--learning_rate', '-l', default=5e-05, type=float, help="Learning rate hyperparameter.")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(
        config["use_model"],
        config["seed"],
        config["tensorflows"],
        config["eval_only"],
        config["learning_rate"],
        config["epochs"]
    )
