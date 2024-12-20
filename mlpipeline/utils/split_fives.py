import os
import pickle
import random

import click
import pandas as pd

from mlpipeline.data.splitter import FoldSplit


def split_train_val(splitter, fold_ind=0):
    return splitter.fold(fold_ind)


@click.command()
@click.option("--metadata_fullname")
@click.option("--output_dir")
@click.option("--stage", default="train")
@click.option("--seed", default=-1)
def main(metadata_fullname, output_dir, stage, seed):
    # Generate seed
    if seed == -1:
        seed = random.randint(1, 99999)

    # Data
    n_folds = 5
    df = pd.read_csv(metadata_fullname)

    # Get dataframes for train
    if stage == "train":
        output_fullname = os.path.join(
            output_dir, f"cv_split_5folds_fives_{seed:05d}.pkl")
        df = df[df["stage"] == "train"]
        print(len(df))

        data = []
        splitter = FoldSplit(ds=df, n_folds=5, target_col=None,
                             group_col=None, random_state=seed)

        for fold_ind in range(n_folds):
            df_train, df_val = split_train_val(splitter, fold_ind=fold_ind)
            data.append((df_train, df_val))
            print(
                f"Fold {fold_ind}, Dataset: FIVES, Size: {len(df)} to {len(df_train)}/{len(df_val)}")

        with open(output_fullname, "wb") as f:
            print(f"Saving metadata to {output_fullname}")
            pickle.dump(data, f, protocol=4)

    # Get dataframe for test
    elif stage == "test":
        output_fullname = os.path.join(
            output_dir, f"cv_split_5folds_fives_test.pkl")
        df = df[df["stage"] == "test"]
        print(len(df))
        df = df.reset_index(drop=True)

        with open(output_fullname, "wb") as f:
            print(f"Saving metadata to {output_fullname}")
            pickle.dump(df, f, protocol=4)

    else:
        output_fullname = os.path.join(
            output_dir, f"cv_split_1folds_dummy_test.pkl")
        df = df[df["stage"] == "train"]
        print(len(df))
        df = df.reset_index(drop=True)
        data = [(df, df.copy())]

        with open(output_fullname, "wb") as f:
            print(f"Saving metadata to {output_fullname}")
            pickle.dump(data, f, protocol=4)

    return


if __name__ == "__main__":
    main()
