import pandas as pd
from datasets import Dataset


def map_label(example):
    """
    Map labels from int to str according to a predefined mapping (see [1]).

    [1]: https://github.com/phhei/ArgsValidNovel/blob/gh-pages/data-description.md
    """
    example["validity_str"] = SharedTaskConstants.validity_label_mapping[example["Validity"]]
    example["novelty_str"] = SharedTaskConstants.novelty_label_mapping[example["Novelty"]]
    return example


class SharedTaskData():
    def __init__(self, filename, ):
        self.df = pd.read_csv(filename)

    def convert_to_hf_dataset(self, label_target, features=None):
        """
        Convert the dataset to Huggingface Dataset with the `label_target` column as target labels.

        If `features` is set, override the label by copying features from existing dataset.
        """
        if label_target not in SharedTaskConstants.targets:
            raise ValueError("Not a valid target label")
        self.target = label_target
        ds = Dataset.from_pandas(self.df, split="train")
        ds = ds.map(map_label)

        if features is not None:
            ds = ds.cast(features)
        else:
            if label_target == "validity":
                ds = ds.class_encode_column("validity_str")
            elif label_target == "novelty":
                ds = ds.class_encode_column("novelty_str")
            else:
                raise ValueError("Unknown target")

        return ds

class SharedTaskConstants():
    """
    Use these constants to interface with the data, not with the id2label used
    inside the Huggingface models!!
    """
    targets = ['validity', 'novelty']
    validity_label_mapping = {
        -1: "not-valid",
        0: "deafisible",  # can be excluded since test set does not contain these
        1: "valid",
    }

    novelty_label_mapping = {
        -1: "not-novel",
        0: "borderline novel",  # can be excluded since test set does not contain these
        1: "novel",
    }

    validity_id2label = {v: k for k, v in validity_label_mapping.items()}
    novelty_id2label = {v: k for k, v in novelty_label_mapping.items()}



if __name__ == "__main__":
    print("Novelty - Train")
    train_data = SharedTaskData("TaskA_train.csv")
    train_dataset_novelty = train_data.convert_to_hf_dataset(label_target='novelty')
    print(train_dataset_novelty.features['novelty_str']._str2int)
    print(len(train_dataset_novelty))

    print("Validity - Train")
    train_dataset_validity = train_data.convert_to_hf_dataset(label_target='validity')
    print(train_dataset_validity.features['validity_str']._str2int)
    print(len(train_dataset_validity))

    print("Novelty - Val")
    val_data = SharedTaskData("TaskA_dev.csv")
    print(train_dataset_novelty.features)
    val_dataset_novelty = val_data.convert_to_hf_dataset(label_target='novelty', features=train_dataset_novelty.features)
    print(val_dataset_novelty.features['novelty_str']._str2int)
    print(len(val_dataset_novelty))

    print("Validity - Val")
    val_dataset_validity = val_data.convert_to_hf_dataset(label_target='validity', features=train_dataset_validity.features)
    print(val_dataset_validity.features['validity_str']._str2int)
    print(len(val_dataset_validity))