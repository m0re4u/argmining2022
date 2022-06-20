import pandas as pd
from datasets import Dataset


def map_label(example):
    """
    Map labels from int to str according to a predefined mapping (see [1]).

    [1]: https://github.com/phhei/ArgsValidNovel/blob/gh-pages/data-description.md
    """
    example["validity_str"] = SharedTaskConstants.validity_label_mapping[example["Validity"]]
    return example


class SharedTaskData():
    def __init__(self, filename):
        self.df = pd.read_csv(filename)

    def convert_to_hf_dataset(self, features=None):
        ds = Dataset.from_pandas(self.df, split="train")
        ds = ds.map(map_label)
        if features is not None:
            ds = ds.cast(features)
        else:
            ds = ds.class_encode_column("validity_str")

        return ds

class SharedTaskConstants():
    """
    Use these constants to interface with the data, not with the id2label used
    inside the Huggingface models!!
    """
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
    train_data = SharedTaskData("TaskA_train.csv")
    train_dataset = train_data.convert_to_hf_dataset()
    print(train_dataset.features['validity_str'].num_classes)
    print(train_dataset.features['validity_str']._int2str)
    print(train_dataset.features['validity_str']._str2int)

