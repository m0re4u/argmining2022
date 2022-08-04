from typing import Dict

import numpy as np
import pandas as pd
from datasets import Dataset, Features


def map_label(example):
    """
    Map labels from int to str according to a predefined mapping (see [1]).

    [1]: https://github.com/phhei/ArgsValidNovel/blob/gh-pages/data-description.md
    """
    example["validity_str"] = SharedTaskConstants.validity_label_mapping[example["Validity"]]
    example["novelty_str"] = SharedTaskConstants.novelty_label_mapping[example["Novelty"]]
    return example


class SharedTaskData:
    def __init__(self, filename, test_set=False):
        self.df = pd.read_csv(filename)
        self.test_set = test_set

    def convert_to_hf_dataset(self, label_target=None, features=None):
        """
        Convert the dataset to Huggingface Dataset with the `label_target` column as target labels.

        If `features` is set, override the label by copying features from existing dataset.
        """
        if self.test_set:
            ds = Dataset.from_pandas(self.df, split='test')
            new_features = Features({
                'topic': features['topic'],
                'Premise':features['Premise'],
                'Conclusion': features['Conclusion']
            })
            ds = ds.cast(new_features)
            return ds

        if label_target not in SharedTaskConstants.targets:
            raise ValueError("Not a valid target label")
        self.target = label_target
        ds_type = "train"
        if features is not None:
            ds_type = "test"
        ds = Dataset.from_pandas(self.df, split=ds_type)
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

    def count_statistics(self):
        """
        Print how many samples there are in the dataset according to label combination
        """
        if self.test_set:
            raise NotImplementedError("Cannot count statistics for test set")
        dataset = self.df
        dataset['Validity'] = dataset['Validity'].apply(lambda x: -1 if x == 0 else x)
        dataset['Novelty'] = dataset['Novelty'].apply(lambda x: -1 if x == 0 else x)
        label_combinations = dataset.groupby(by=['Validity', 'Novelty'])

        for index, group in label_combinations:
            print(f'Validity: {index[0]}, Novelty: {index[1]}')
            print(f'\t{len(group)}')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.test_set:
            return {
                'premise': row['Premise'],
                'topic': row['topic'],
                'conclusion': row['Conclusion'],
            }
        else:
            return (
                {
                    'premise': row['Premise'],
                    'topic': row['topic'],
                    'conclusion': row['Conclusion'],
                },
                SharedTaskConstants.validity_label_mapping[row['Validity']],
                SharedTaskConstants.novelty_label_mapping[row['Novelty']]
            )

    def __len__(self):
        return len(self.df)


class SharedTaskConstants:
    """
    Use these constants to interface with the data, not with the id2label used
    inside the Huggingface models!!
    """
    targets = ['validity', 'novelty']
    validity_label_mapping = {
        -1: "not-valid",
        0: "not-valid",  # can be excluded since test set does not contain these
        1: "valid",
    }

    novelty_label_mapping = {
        -1: "not-novel",
        0: "not-novel",  # can be excluded since test set does not contain these
        1: "novel",
    }

    validity_id2label = {v: k for k, v in validity_label_mapping.items()}
    novelty_id2label = {v: k for k, v in novelty_label_mapping.items()}

    local_str_mapping = {
        'novel': 1,
        'not-novel': 0,
        'valid': 1,
        'not-valid': 0
    }

    @staticmethod
    def val_nov_metric(is_validity: np.ndarray, should_validity: np.ndarray, is_novelty: np.ndarray,
                       should_novelty: np.ndarray) -> Dict[str, float]:
        ret = dict()

        ret_base_help = {
            "true_positive_validity": np.sum(np.where(
                np.all(np.stack([is_validity >= .5, should_validity >= .5]), axis=0),
                1, 0)),
            "true_positive_novelty": np.sum(np.where(
                np.all(np.stack([is_novelty >= .5, should_novelty >= .5]), axis=0),
                1, 0)),
            "true_positive_valid_novel": np.sum(np.where(
                np.all(np.stack([is_validity >= .5, is_novelty >= .5,
                                 should_validity >= .5, should_novelty >= .5]), axis=0),
                1, 0)),
            "true_positive_nonvalid_novel": np.sum(np.where(
                np.all(np.stack([is_validity < .5, is_novelty >= .5,
                                 should_validity < .5, should_novelty >= .5]), axis=0),
                1, 0)),
            "true_positive_valid_nonnovel": np.sum(np.where(
                np.all(np.stack([is_validity >= .5, is_novelty < .5,
                                 should_validity >= .5, should_novelty < .5]), axis=0),
                1, 0)),
            "true_positive_nonvalid_nonnovel": np.sum(np.where(
                np.all(np.stack([is_validity < .5, is_novelty < .5,
                                 should_validity < .5, should_novelty < .5]), axis=0),
                1, 0)),
            "classified_positive_validity": np.sum(np.where(is_validity >= .5, 1, 0)),
            "classified_positive_novelty": np.sum(np.where(is_novelty >= .5, 1, 0)),
            "classified_positive_valid_novel": np.sum(np.where(
                np.all(np.stack([is_validity >= .5, is_novelty >= .5]), axis=0),
                1, 0)),
            "classified_positive_nonvalid_novel": np.sum(np.where(
                np.all(np.stack([is_validity < .5, is_novelty >= .5]), axis=0),
                1, 0)),
            "classified_positive_valid_nonnovel": np.sum(np.where(
                np.all(np.stack([is_validity >= .5, is_novelty < .5]), axis=0),
                1, 0)),
            "classified_positive_nonvalid_nonnovel": np.sum(np.where(
                np.all(np.stack([is_validity < .5, is_novelty < .5]), axis=0),
                1, 0)),
            "indeed_positive_validity": np.sum(np.where(should_validity >= .5, 1, 0)),
            "indeed_positive_novelty": np.sum(np.where(should_novelty >= .5, 1, 0)),
            "indeed_positive_valid_novel": np.sum(np.where(
                np.all(np.stack([should_validity >= .5, should_novelty >= .5]), axis=0),
                1, 0)),
            "indeed_positive_nonvalid_novel": np.sum(np.where(
                np.all(np.stack([should_validity < .5, should_novelty >= .5]), axis=0),
                1, 0)),
            "indeed_positive_valid_nonnovel": np.sum(np.where(
                np.all(np.stack([should_validity >= .5, should_novelty < .5]), axis=0),
                1, 0)),
            "indeed_positive_nonvalid_nonnovel": np.sum(np.where(
                np.all(np.stack([should_validity < .5, should_novelty < .5]), axis=0),
                1, 0)),
        }

        ret_help = {
            "precision_validity": ret_base_help["true_positive_validity"] /
                                  max(1, ret_base_help["classified_positive_validity"]),
            "precision_novelty": ret_base_help["true_positive_novelty"] /
                                 max(1, ret_base_help["classified_positive_novelty"]),
            "recall_validity": ret_base_help["true_positive_validity"] /
                               max(1, ret_base_help["indeed_positive_validity"]),
            "recall_novelty": ret_base_help["true_positive_novelty"] /
                              max(1, ret_base_help["indeed_positive_novelty"]),
            "precision_valid_novel": ret_base_help["true_positive_valid_novel"] /
                                     max(1, ret_base_help["classified_positive_valid_novel"]),
            "precision_valid_nonnovel": ret_base_help["true_positive_valid_nonnovel"] /
                                        max(1, ret_base_help["classified_positive_valid_nonnovel"]),
            "precision_nonvalid_novel": ret_base_help["true_positive_nonvalid_novel"] /
                                        max(1, ret_base_help["classified_positive_nonvalid_novel"]),
            "precision_nonvalid_nonnovel": ret_base_help["true_positive_nonvalid_nonnovel"] /
                                           max(1, ret_base_help["classified_positive_nonvalid_nonnovel"]),
            "recall_valid_novel": ret_base_help["true_positive_valid_novel"] /
                                  max(1, ret_base_help["indeed_positive_valid_novel"]),
            "recall_valid_nonnovel": ret_base_help["true_positive_valid_nonnovel"] /
                                     max(1, ret_base_help["indeed_positive_valid_nonnovel"]),
            "recall_nonvalid_novel": ret_base_help["true_positive_nonvalid_novel"] /
                                     max(1, ret_base_help["indeed_positive_nonvalid_novel"]),
            "recall_nonvalid_nonnovel": ret_base_help["true_positive_nonvalid_nonnovel"] /
                                        max(1, ret_base_help["indeed_positive_nonvalid_nonnovel"])
        }

        ret.update({
            "f1_validity": 2 * ret_help["precision_validity"] * ret_help["recall_validity"] / max(1e-4, ret_help[
                "precision_validity"] + ret_help["recall_validity"]),
            "f1_novelty": 2 * ret_help["precision_novelty"] * ret_help["recall_novelty"] / max(1e-4, ret_help[
                "precision_novelty"] + ret_help["recall_novelty"]),
            "f1_valid_novel": 2 * ret_help["precision_valid_novel"] * ret_help["recall_valid_novel"] / max(1e-4,
                                                                                                           ret_help[
                                                                                                               "precision_valid_novel"] +
                                                                                                           ret_help[
                                                                                                               "recall_valid_novel"]),
            "f1_valid_nonnovel": 2 * ret_help["precision_valid_nonnovel"] * ret_help["recall_valid_nonnovel"] / max(
                1e-4, ret_help["precision_valid_nonnovel"] + ret_help["recall_valid_nonnovel"]),
            "f1_nonvalid_novel": 2 * ret_help["precision_nonvalid_novel"] * ret_help["recall_nonvalid_novel"] / max(
                1e-4, ret_help["precision_nonvalid_novel"] + ret_help["recall_nonvalid_novel"]),
            "f1_nonvalid_nonnovel": 2 * ret_help["precision_nonvalid_nonnovel"] * ret_help[
                "recall_nonvalid_nonnovel"] / max(1e-4, ret_help["precision_nonvalid_nonnovel"] + ret_help[
                "recall_nonvalid_nonnovel"])
        })

        ret.update({
            "f1_macro": (ret["f1_valid_novel"] + ret["f1_valid_nonnovel"] + ret["f1_nonvalid_novel"] + ret[
                "f1_nonvalid_nonnovel"]) / 4
        })

        return ret


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
    val_dataset_novelty = val_data.convert_to_hf_dataset(label_target='novelty',
                                                         features=train_dataset_novelty.features)
    print(val_dataset_novelty.features['novelty_str']._str2int)
    print(len(val_dataset_novelty))

    print("Validity - Val")
    val_dataset_validity = val_data.convert_to_hf_dataset(label_target='validity',
                                                          features=train_dataset_validity.features)
    print(val_dataset_validity.features['validity_str']._str2int)
    print(len(val_dataset_validity))
