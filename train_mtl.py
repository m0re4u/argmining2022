import argparse

import torch
import transformers
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, TrainingArguments, set_seed

from data import SharedTaskData
from models import MultitaskModel
from trainers import MultitaskTrainer, NLPDataCollator


class Tokenize:
    def __init__(self, model_name, tensorflows, tokenizer_structure, predict_only):
        self.model_name: str = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                       from_tf=tensorflows)
        self.structure = tokenizer_structure
        self.predict_only = predict_only

    def _tokenize_fn(self, examples):
        batch_size = len(examples['Premise'])
        if self.structure == 'concatenation':
            batched_inputs = [
                examples['topic'][i] + self.tokenizer.sep_token + \
                examples['Premise'][i] + self.tokenizer.sep_token + \
                examples['Conclusion'][i] for i in range(batch_size)
            ]
        elif self.structure == 'segment+concatenation':
            batched_inputs = [
                f"TOPIC: {examples['topic'][i]}" + self.tokenizer.sep_token + \
                f"PREMISE: {examples['Premise'][i]}" + self.tokenizer.sep_token + \
                f"CONCLUSION: {examples['Conclusion'][i]}" for i in range(batch_size)
            ]
        return self.tokenizer(batched_inputs, truncation=True, padding=True)

    def tokenize_function_val(self, examples):
        samples = self._tokenize_fn(examples)
        if not self.predict_only:
            samples['labels'] = examples['validity_str']
        return samples

    def tokenize_function_nov(self, examples):
        samples = self._tokenize_fn(examples)
        if not self.predict_only:
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


def prepare_data(model_name, train_dataset, dev_dataset, target_task, tokenizer_fn, predict_only=False):
    """
    Prepare a training and development dataset for HF training. Includes tokenizations and casting
    to torch tensors. In case `predict_only` is set, ignore all labels in the dev set (will be used
    as the test set).
    """
    train_dataset = train_dataset.convert_to_hf_dataset(target_task)
    dev_dataset = dev_dataset.convert_to_hf_dataset(target_task, features=train_dataset.features)

    tokenized_train_dataset = train_dataset.map(tokenizer_fn, batched=True)
    tokenized_dev_dataset = dev_dataset.map(tokenizer_fn, batched=True)
    if 'roberta' in model_name or 'ArgumentRelation' in model_name:
        column_names = ['input_ids', 'attention_mask']
    else:
        column_names = ['input_ids', 'token_type_ids', 'attention_mask']

    if not predict_only:
        column_names += ['labels']
    tokenized_train_dataset.set_format(type='torch', columns=column_names)
    tokenized_dev_dataset.set_format(type='torch', columns=column_names)
    return tokenized_train_dataset, tokenized_dev_dataset


def main(
        use_model: str = "bert-base-uncased",
        seed: int = 0,
        tensorflows: bool = False,
        eval_only: bool = False,
        predict_only: bool = False,
        learning_rate: float = 5e-05,
        epochs: int = 10,
        tokenizer_structure: str = 'concatenation',
        checkpoint: str = None
    ):
    set_seed(seed)
    tokenize = Tokenize(use_model, tensorflows, tokenizer_structure, predict_only)

    # torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(config["pipeline"]["seed"])
    torch.backends.cudnn.deterministic = True

    train_data = SharedTaskData("TaskA_train.csv")
    dev_data = SharedTaskData("TaskA_dev.csv")
    if predict_only:
        dev_data = SharedTaskData("TaskA_test-without-labels.csv", test_set=True)

    tokenized_train_dataset_novelty, tokenized_dev_dataset_novelty = prepare_data(
        use_model,
        train_data,
        dev_data,
        "novelty",
        tokenize.tokenize_function_nov,
        predict_only=predict_only
    )
    tokenized_train_dataset_validity, tokenized_dev_dataset_validity = prepare_data(
        use_model,
        train_data,
        dev_data,
        "validity",
        tokenize.tokenize_function_val,
        predict_only=predict_only
    )

    if "ArgumentRelation" in use_model:
        kwargs = {'pad_token_id': 1}
    else:
        kwargs = {}

    print(tokenized_dev_dataset_validity)
    print(tokenized_dev_dataset_novelty)
    print(tokenized_dev_dataset_validity[0])
    print(tokenized_dev_dataset_novelty[0])

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

    if checkpoint is not None:
        multitask_model.load_trainer_checkpoint(checkpoint)

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
        report_to="none" if eval_only else "wandb",
        logging_strategy="epoch",
        metric_for_best_model="eval_org_f1_macro",
        greater_is_better=True,
        load_best_model_at_end=True,
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
        tokenizer=tokenize.tokenizer,
        compute_metrics=compute_metrics
    )

    if predict_only:
        print("results novelty")
        results = trainer.predict(val_dataset)
        for result_key, result in results.items():
            print(f"  {result_key:>52} : {result:2.4f}")
    if eval_only:
        results = trainer.evaluate()
        for result_key, result in results.items():
            print(f"  {result_key:>52} : {result:2.4f}")
    else:
        trainer.train()
        trainer.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_model', default=None, type=str,
                        help="Which model to load as model definition")
    parser.add_argument('--tensorflows', default=False, type=bool,
                        help="enable loading from Tensorflow models")
    parser.add_argument('--eval_only', default=False, action='store_true',
                        help="Whether to do training or evaluation only")
    parser.add_argument('--predict_only', default=False, action='store_true',
                        help="Whether to do predict on test set. Takes prevalence over eval_only")
    parser.add_argument('--seed', '-s', default=0, type=int, help="Set seed for reproducibility.")
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs to train for.")
    parser.add_argument('--learning_rate', '-l', default=5e-05, type=float, help="Learning rate hyperparameter.")
    parser.add_argument('--tokenizer_structure', default='concatenation', type=str, choices=['concatenation', 'segment+concatenation'],
                        help="Which input string structure to use in tokenization (mixes Topic/Premise/Conclusion segments)")
    parser.add_argument('--checkpoint', default=None, type=str,
                        help="Which trained checkpoint to load (weights only)")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(**config)
