import pandas as pd
import os

import click
from tqdm import tqdm
from natsort import natsorted


def retrieve_data_from_directory(path, stage, img_dir="Original", mask_dir="GroundTruth"):
    mask = mask_dir
    ori = img_dir
    filenames = natsorted(os.listdir(os.path.join(path, ori)))
    print("Length:", len(filenames))
    data = []

    for filename in tqdm(filenames, total=len(filenames), desc="Processing"):
        mask_filename = os.path.join(path, mask, filename)
        if filename.endswith(".png") and os.path.isfile(mask_filename):
            row = dict()
            row["input"] = stage + "/" + ori + "/" + filename
            row["gt"] = stage + "/" + mask + "/" + filename
            row["gt_b"] = stage + "/" + mask + \
                "_b" + "/" + filename[:-4] + ".npy"
            row["gt_t"] = stage + "/" + mask + \
                "_t" + "/" + filename[:-4] + ".npy"
            row["gt_c"] = stage + "/" + mask + \
                "_c" + "/" + filename[:-4] + ".npy"
            row["stage"] = stage
            data.append(row)

    data = pd.DataFrame(data)
    return data


@click.command()
@click.option("--img_root")
@click.option("--img_dir", default="Original")
def main(img_root: str, img_dir: str):
    df_fullname = os.path.join(img_root, "fives.csv")

    all_data = []
    for stage in ["train", "test"]:
        root = os.path.join(img_root, stage)
        data = retrieve_data_from_directory(root, stage, img_dir=img_dir)
        all_data.append(data)

    df = pd.concat(all_data, axis=0)
    df.to_csv(df_fullname, index=False)


if __name__ == '__main__':
    main()
