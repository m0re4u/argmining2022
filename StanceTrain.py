import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import torch
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import logging
import numpy as np
import time
import os
import random
from argparse import ArgumentParser
from typing import Sequence
import datetime

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

random_seed=77990

random.seed(random_seed)
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

# device = "cuda:0"
#if torch.cuda.is_available() else "cpu"

def import_df(path, fraction=None):
    data = pd.read_csv(path)
    df = data.dropna()
    df = df.sample(frac=1, random_state=random_seed)
    if fraction is not None:
        df = df.sample(fraction,random_state=random_seed)
    return df

def make_lists(df):
    text1 = df["text1"].tolist()
    text2 = df["text2"].tolist()
    labels = df["label"].tolist()

    ints = map(int, labels)
    labels = list(ints)
    return text1, text2, labels

def encode_data(text1, text2, labels, model_name, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = tokenizer(text1,
                          text2,
                          padding="max_length",
                          truncation=True,
                          return_tensors="tf")
    encodings = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    tokenizer.save_pretrained(model_path)
    return encodings

def train_model(train_data, model_name, batch_size, learning_rate, epochs, num_labels, model_path):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-06)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name,
                                                                 num_labels=num_labels)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(x=train_data.shuffle(1000).batch(batch_size),
              validation_data=None,
              epochs=epochs)

    model.save_pretrained(model_path)

    model.config.__class__.from_pretrained(model_name).save_pretrained(model_path)
    return model, model_path


def predict(texts1, texts2, model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    encodings = tokenizer(texts1,
                          texts2,
                          padding="max_length",
                          truncation=True,
                          return_tensors="tf")
    print(encodings[0])

    logging.info(f'running inference using model {model_name} on {len(texts1)} text pairs')

    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

    predictions = model(encodings)
    logits = predictions.logits.numpy().tolist()
    preds = np.argmax(logits, axis=1)

    print(encodings[0])
    return preds

def evaluate(true, pred, model_name, TRAIN, DEV, timing):
    results = []
    labels = [int(x) for x in true]

    print("LABELS")
    print(labels)
    print("prediction")
    print(pred)

    model = f"model is {model_name}"
    train_data = f"train data is {TRAIN}"
    test_data = f"test data is {DEV}"

    acc = accuracy_score(true, pred)
    accuracy = f"accuracy is {acc}"

    f1 = precision_recall_fscore_support(labels, pred, average='macro')
    prec = f"precision is {f1[0]}"
    rec = f"recall is {f1[1]}"
    f1 = f"F1 is {f1[2]}"

    report = classification_report(labels, pred)

    time = (f"this took: {timing}")

    results.extend([train_data, test_data, model, accuracy, prec, rec, f1, report, time])
    results_string = "\n".join(results)
    print(results_string)
    return results_string

def write_results(results, filename):
    with open(filename, "w+") as file:
        file.write(results)

################

# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO,
#                         format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
#
#     parser = ArgumentParser()
#     parser.add_argument('--train_file', type=str, required=True)
#     parser.add_argument('--eval_file', type=str, required=True)
#     parser.add_argument('--random_seed', type=int, default=0)
#     # parser.add_argument('--inter_training_epochs', type=int, default=1)
#     # parser.add_argument('--finetuning_epochs', type=int, default=10)
#     # parser.add_argument('--num_clusters', type=int, default=50)
#
#     args = parser.parse_args()
#     logging.info(args)
#
#     # set random seed
#     random.seed(args.random_seed)
#     tf.random.set_seed(args.random_seed)
#     np.random.seed(args.random_seed)

start = time.time()

BATCH=64
RATE=0.00005
EPOCHS=15
NUM_LABELS=2
###################
TRAIN = 'dataframe_allTraining_SameSide.csv'
DEV = 'dataframe_allDev_SameSide.csv'
model_name='bert-base-uncased'

###################
train_text1, train_text2, train_labels = make_lists(import_df(TRAIN))
dev_text1, dev_text2, dev_labels = make_lists(import_df(TRAIN))

df_subset = import_df(TRAIN)
train1_sub, train2_sub, train_labels_sub = make_lists(df_subset)

df_subset_dev = import_df(DEV, 500)
dev1_sub, dev2_sub, dev_labels_sub = make_lists(df_subset_dev)

########
model_dir = f"1_model_{model_name}"
data_dir = f"{TRAIN}"
output_dir = os.path.join(model_dir, "output/")
model_save=os.path.join(model_dir,f"19_4july_TRAIN-size:{len(train1_sub)}/")
######################

train_encodings = encode_data(train1_sub,
                              train2_sub,
                              train_labels_sub,
                              model_name,
                              model_save)

model, model_path = train_model(train_encodings,
                               model_name=model_name,
                               batch_size=BATCH,
                               learning_rate=RATE,
                               epochs=EPOCHS,
                               num_labels=NUM_LABELS,
                               model_path=model_save)

predictions = predict(dev1_sub,
                      dev2_sub,
                      model_name=model_save,
                      num_labels=NUM_LABELS
                      )

# write_results(results, f"model_{model_name}/TRAIN-{TRAIN}/results_{model_name}_{datetime.datetime.day(tz=None)}.txt")

stop = time.time()
timing = stop-start
time_processed = datetime.timedelta(seconds=timing)
print(f"this took: {time_processed}")

results = evaluate(df_subset_dev["label"], predictions, model_name, TRAIN, DEV, timing)
write_results(results, os.path.join(model_save,f"14_4july_results_{model_name}_train-{len(train1_sub)}_seed-{random_seed}_batch-{BATCH}_LR-{RATE}.txt"))

# print(df_subset["label"].value_counts())
# print(df_subset.head())
#
# print(df_subset_dev["label"].value_counts())
# print(df_subset_dev.head())
