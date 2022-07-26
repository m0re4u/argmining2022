import argparse
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

def get_dataset_config(dataset):
    if dataset == 'anli':
        num_labels = 3
    else:
        raise ValueError(f"No config found for dataset: {dataset} ")
    return num_labels


def compute_metrics(p):
    f1_metric = evaluate.load('f1')
    acc_metric = evaluate.load('accuracy')
    return {
        'accuracy': acc_metric.compute(predictions=p.predictions.argmax(axis=1), references=p.label_ids),
        'f1': f1_metric.compute(predictions=p.predictions.argmax(axis=1), references=p.label_ids, average='macro')
    }


def main(args):
    cfg_num_labels = get_dataset_config(args.dataset)
    checkpoint = args.use_model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=cfg_num_labels)

    def encode(examples):
        samples = tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length')
        samples['labels'] = examples['label']
        return samples

    dataset = load_dataset(args.dataset)
    dataset = dataset.map(encode, batched=True)

    # Arguments for training loop
    training_args = TrainingArguments(
        f"intermediate_{args.use_model}_{args.dataset}",
        num_train_epochs=10,
        report_to="wandb",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        label_names=['labels']
    )

    # Go!
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train_r1'],
        eval_dataset=dataset['dev_r1'],
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_model', default='bert-base-uncased', type=str,
                        help="Which model to load")
    parser.add_argument('--dataset', default='anli', type=str, choices=['anli'],
                        help="Which dataset to train on for intermediate training to load")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
