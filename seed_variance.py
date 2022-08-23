import argparse
import pandas as pd
import numpy as np

def main(args):
    df = pd.read_csv(args.wandb_csv)
    print(df)
    print(df.columns)
    runs = [
        'hftrainer_am_mtl_../SimCSE/result/novelty-sup-simcse-sup-roberta-large-mnli/_10_1e-05_0 - eval/org_f1_macro',
        'hftrainer_am_mtl_../SimCSE/result/novelty-sup-simcse-sup-roberta-large-mnli/_10_1e-05_1 - eval/org_f1_macro',
        'hftrainer_am_mtl_../SimCSE/result/novelty-sup-simcse-sup-roberta-large-mnli/_10_1e-05_7 - eval/org_f1_macro',
        'hftrainer_am_mtl_../SimCSE/result/novelty-sup-simcse-sup-roberta-large-mnli/_10_1e-05_42 - eval/org_f1_macro',
        'hftrainer_am_mtl_../SimCSE/result/novelty-sup-simcse-sup-roberta-large-mnli/_10_1e-05_1337 - eval/org_f1_macro',
    ]
    scores = df[runs]
    scores['mean'] = df[runs].apply(lambda y: np.average([np.array(x) for x in y]), axis=1)
    scores['std'] = df[runs].apply(lambda y: np.std([np.array(x) for x in y]), axis=1)
    print(scores)
    for i, row in scores.iterrows():
        print(f"{i}\t{row['mean']}\t{row['std']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('wandb_csv', default=None, type=str,
                        help="CSV export from wandb")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(args)
