import os

import numpy as np
import seaborn as sns
import torch
import transformers
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from data import SharedTaskData
from models import MultitaskModel
from trainers import MultitaskTrainer, NLPDataCollator

checkpoint = "/home/mreuver/SSSC_test/1_model_bert-base-uncased/10_19JUNE_TRAIN-size:20763"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, from_tf=True)
path_config = "/home/mreuver/SSSC_test/1_model_bert-base-uncased/10_19JUNE_TRAIN-size:20763/config.json"

def tokenize_function_val(examples):
    batch_size = len(examples['Premise'])
    batched_inputs = [
        examples['topic'][i] + tokenizer.sep_token + \
        examples['Premise'][i] + tokenizer.sep_token + \
        examples['Conclusion'][i] for i in range(batch_size)
    ]
    samples = tokenizer(batched_inputs, truncation=True, padding=True)
    samples['labels'] = examples['validity_str']
    return samples


def tokenize_function_nov(examples):
    batch_size = len(examples['Premise'])
    batched_inputs = [
        examples['topic'][i] + tokenizer.sep_token + \
        examples['Premise'][i] + tokenizer.sep_token + \
        examples['Conclusion'][i] for i in range(batch_size)
    ]
    samples = tokenizer(batched_inputs, truncation=True, padding=True)
    samples['labels'] = examples['novelty_str']
    return samples


def single_label_metrics(predictions, labels):
    softmax = torch.nn.Softmax(dim=1)
    preds = torch.Tensor(predictions)
    probs = softmax(preds)
    y_pred = torch.argmax(probs, dim=1)
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    return {'f1': f1_micro_average, 'accuracy': accuracy}


def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return single_label_metrics(
        predictions=preds,
        labels=p.label_ids
    )


def main():
    train_data = SharedTaskData("TaskA_train.csv")
    dev_data = SharedTaskData("TaskA_dev.csv")

    train_dataset_novelty = train_data.convert_to_hf_dataset("novelty")
    train_dataset_validity = train_data.convert_to_hf_dataset("validity")
    # Use feature mapping from training dataset to ensure features are mapped correctly
    dev_dataset_novelty = dev_data.convert_to_hf_dataset("novelty", features=train_dataset_novelty.features)
    dev_dataset_validity = dev_data.convert_to_hf_dataset("validity", features=train_dataset_validity.features)

    # Make sure internal label mapping is identical across datasets
    assert train_dataset_validity.features['validity_str']._str2int == dev_dataset_validity.features['validity_str']._str2int
    assert train_dataset_novelty.features['novelty_str']._str2int == dev_dataset_novelty.features['novelty_str']._str2int

    tokenized_train_dataset_validity = train_dataset_validity.map(tokenize_function_val, batched=True)
    tokenized_dev_dataset_validity = dev_dataset_validity.map(tokenize_function_val, batched=True)
    tokenized_train_dataset_validity.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    tokenized_dev_dataset_validity.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    tokenized_train_dataset_novelty = train_dataset_novelty.map(tokenize_function_nov, batched=True)
    tokenized_dev_dataset_novelty = dev_dataset_novelty.map(tokenize_function_nov, batched=True)
    tokenized_train_dataset_novelty.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    tokenized_dev_dataset_novelty.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    # intermed_model = MultitaskModel.make_intermed_model(
    #     model_name=checkpoint,
    #     model_type=transformers.AutoModelForSequenceClassification,
    #     model_config=checkpoint
    # )

    #print(intermed_model)
    
    model = MultitaskModel.create_intermed(
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

    #print(intermed_model["taskmodels_dict"])

    train_dataset = {
        "validity": tokenized_train_dataset_validity,
        "novelty": tokenized_train_dataset_novelty
    }

    val_dataset = {
        "validity": tokenized_dev_dataset_validity,
        "novelty": tokenized_dev_dataset_novelty
    }


    training_args = TrainingArguments(
        "argmining2022_trainer_mtl",
        num_train_epochs=10,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = MultitaskTrainer(
        model=model,
        args=training_args,
        data_collator=NLPDataCollator(tokenizer=tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        #batch_size=64
    )
    trainer.train()


if __name__ == "__main__":
    main()