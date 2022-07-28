import argparse
import evaluate
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

def get_dataset_config(dataset):
    dataset_config = {}
    if dataset == 'anli':
        dataset_config['num_labels'] = 3
        dataset_config['train_set_names'] = ['train_r1', 'train_r2', 'train_r3']
        dataset_config['dev_set_names'] = ['dev_r1', 'dev_r2', 'dev_r3']
    else:
        raise ValueError(f"No config found for dataset: {dataset} ")
    return dataset_config


def compute_metrics(p):
    f1_metric = evaluate.load('f1')
    acc_metric = evaluate.load('accuracy')
    return {
        'accuracy': acc_metric.compute(predictions=p.predictions.argmax(axis=1), references=p.label_ids),
        'f1': f1_metric.compute(predictions=p.predictions.argmax(axis=1), references=p.label_ids, average='macro')
    }


def main(args):
    # Load config
    dataset_cfg = get_dataset_config(args.dataset)
    checkpoint = args.use_model

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=dataset_cfg['num_labels'])

    def encode(examples):
        samples = tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length')
        samples['labels'] = examples['label']
        return samples

    # Load dataset in memory and tokenize
    dataset = load_dataset(args.dataset)
    dataset = dataset.map(encode, batched=True)

    # Combine dataset partitions
    train_set = concatenate_datasets([dataset[x] for x in dataset_cfg['train_set_names']])
    dev_set = concatenate_datasets([dataset[x] for x in dataset_cfg['dev_set_names']])

    # Arguments for training loop
    training_args = TrainingArguments(
        f"hftrainer_intermediate_{args.use_model}_{args.dataset}",
        num_train_epochs=args.num_epochs,
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
        tokenizer=tokenizer,
        train_dataset=train_set,
        eval_dataset=dev_set,
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_model', default='bert-base-uncased', type=str,
                        help="Which model to load")
    parser.add_argument('--dataset', default='anli', type=str, choices=['anli'],
                        help="Which dataset to train on for intermediate training to load")
    parser.add_argument('--num_epochs', default=10, type=int,
                        help="Number of epochs to train with")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
