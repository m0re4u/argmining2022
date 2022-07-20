import argparse
import torch
import transformers
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, TrainingArguments

from data import SharedTaskData
from models import MultitaskModel
from trainers import MultitaskTrainer, NLPDataCollator

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def _tokenize_fn(examples):
    batch_size = len(examples['Premise'])
    batched_inputs = [
        examples['topic'][i] + tokenizer.sep_token + \
        examples['Premise'][i] + tokenizer.sep_token + \
        examples['Conclusion'][i] for i in range(batch_size)
    ]
    return tokenizer(batched_inputs, truncation=True, padding=True)


def tokenize_function_val(examples):
    samples = _tokenize_fn(examples)
    samples['labels'] = examples['validity_str']
    return samples


def tokenize_function_nov(examples):
    samples = _tokenize_fn(examples)
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
    print(p.predictions)
    print(p.label_ids)
    print(p.inputs)
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return single_label_metrics(
        predictions=preds,
        labels=p.label_ids
    )

def prepare_data(train_dataset, dev_dataset, target_task, tokenizer_fn):
    """
    Prepare a training and development dataset for HF training. Includes tokenizations and casting to torch tensors.
    """
    train_dataset = train_dataset.convert_to_hf_dataset(target_task)
    dev_dataset = dev_dataset.convert_to_hf_dataset(target_task, features=train_dataset.features)
    assert train_dataset.features[f'{target_task}_str']._str2int == dev_dataset.features[f'{target_task}_str']._str2int
    tokenized_train_dataset = train_dataset.map(tokenizer_fn, batched=True)
    tokenized_dev_dataset = dev_dataset.map(tokenizer_fn, batched=True)
    tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    tokenized_dev_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    return tokenized_train_dataset, tokenized_dev_dataset

def main():
    train_data = SharedTaskData("TaskA_train.csv")
    dev_data = SharedTaskData("TaskA_dev.csv")

    tokenized_train_dataset_novelty, tokenized_dev_dataset_novelty = prepare_data(train_data, dev_data, "novelty", tokenize_function_nov)
    tokenized_train_dataset_validity, tokenized_dev_dataset_validity = prepare_data(train_data, dev_data, "validity", tokenize_function_val)

    # Create model
    multitask_model = MultitaskModel.create(
        model_name=checkpoint,
        model_type_dict={
            "novelty": transformers.AutoModelForSequenceClassification,
            "validity": transformers.AutoModelForSequenceClassification,
        },
        model_config_dict={
            "novelty": transformers.AutoConfig.from_pretrained(checkpoint, num_labels=2),
            "validity": transformers.AutoConfig.from_pretrained(checkpoint, num_labels=2),
        },
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
        "argmining2022_trainer_mtl",
        num_train_epochs=10,
        # report_to="wandb",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        label_names=['labels']
    )

    # Go!
    trainer = MultitaskTrainer(
        model=multitask_model,
        args=training_args,
        data_collator=NLPDataCollator(tokenizer=tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_model', default=None, type=str,
                        help="Which model to load")
    parser.add_argument('--eval_only', default=False, action='store_true',
                        help="Whether to do training or evaluation only")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main()
