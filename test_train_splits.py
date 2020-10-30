"""
Task 6.5 If training with test/train splits, this script generates splits.
"""

import argparse
import pandas as pd
import config
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical
from utils.file import directory_check, get_experiment_tag, generate_out_file


def arg_parse(arg_list=None):
    parser = argparse.ArgumentParser(description="Generate Train/Test Splits")

    #### Data Characteristics ####

    parser.add_argument(
        "--negative-examples",
        "-neg",
        dest="neg",
        action="store_true",
        default=False,
        help="Use Negative Examples as 6th class",
    )

    parser.add_argument(
        "--unbalanced",
        "-ub",
        dest="unbalanced",
        action="store_true",
        default=False,
        help="Do not balance dataset. Default False.",
    )

    parser.add_argument(
        "--test-size",
        "-ts",
        dest="test_size",
        type=float,
        default=0.2,
        help="The portion of data to allocate to the test set, default 0.2",
    )

    parser.add_argument(
        "--in-dir",
        "-in",
        dest="in_dir",
        type=str,
        default=config.SP_DIR,
        help=f"Set directory that has shortest path dataframe, default {config.SP_DIR}",
    )

    parser.add_argument(
        "--out-dir",
        "-out",
        dest="out_dir",
        type=str,
        default=config.SP_SPLITS_DIR,
        help=f"Set filepath for splits export, default {config.SP_SPLITS_DIR}",
    )

    if arg_list:
        return parser.parse_args(args=arg_list)
    else:
        return parser.parse_args()


def add_negative_samples(df):
    df_no = df.loc[df["Maj_Vote"] == "no"]
    df_no["Relation"] = "none"
    df_no["Maj_Vote"] = "yes"
    df = df.append(df_no)
    return df


if __name__ == "__main__":
    args = arg_parse()

    now = datetime.datetime.now().strftime("%y%m%d")

    directory_check(args.in_dir)
    directory_check(args.out_dir)

    # Can manually write in path to shortest path pickles, if deviating too far from standard execution
    sp_files = [(args.in_dir + '/' + x) for x in os.listdir(args.in_dir) if x.endswith('.pkl')]
    assert len(sp_files) <= 2, "Only a maximum of two dataframes can be processed"


    tag = ""
    if args.neg:
        tag += "neg-"
    if args.unbalanced:
        tag += "unbal-"
    tag += get_experiment_tag(sp_files[0]) + f"-{now}"

    df = None

    if len(sp_files) > 1:
        # merge two dataframes
        df1 = pd.read_pickle(sp_files[0])
        df2 = pd.read_pickle(sp_files[1])
        df = pd.merge(df1,df2, on=['UID', 'Maj_Vote', 'Relation'], how='inner')
        del df1
        del df2
        df['Subject'] = df['Subject_x']
        df['Object'] = df['Object_x']
        df['Short_Path'] = df['Short_Path_x'].apply(lambda x: x.tolist()) + df['Short_Path_y'].apply(lambda x: x.tolist())
        df = df.drop(columns=['Subject_x', 'Object_x', 'Subject_y', 'Object_y', 'Short_Path_x', 'Short_Path_y'])
    else:
        df = pd.read_pickle(sp_files[0])

    # Remove NaN Rows
    df = df.dropna(how="any")

    # Add null class ('No' Votes)
    if args.neg:
        df = add_negative_samples(df)

    # Drop rows where the majority vote was no/skip
    df = df.loc[df["Maj_Vote"] == "yes"]
    
    # Balance Classes
    if not args.unbalanced:
        df_temp = df.groupby("Relation")
        df = df_temp.apply(lambda x: x.sample(df_temp.size().min())).reset_index(
            drop=True
        )
    
    # Prep Labels
    y = df[["Relation"]]
    encoder = LabelEncoder()
    encoder.fit(y)
    np.save(f"{config.SP_SPLITS_DIR}/class_encoder.npy", encoder.classes_)
    y = encoder.transform(y)
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=args.test_size, stratify=y, random_state=config.RANDOM_SEED
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=args.test_size,
        stratify=y_train,
        random_state=config.RANDOM_SEED,
    )

    out_dir = config.SP_SPLITS_DIR + '/train'
    out_file = "X_train.pkl"
    out_file = generate_out_file(out_file, out_dir, tag)
    X_train.to_pickle(out_file)
    out_file = "y_train.npy"
    out_file = generate_out_file(out_file, out_dir, tag)
    np.save(out_file, y_train)

    out_dir = config.SP_SPLITS_DIR + '/valid'
    out_file = "X_valid.pkl"
    out_file = generate_out_file(out_file, out_dir, tag)
    X_valid.to_pickle(out_file)
    out_file = "y_valid.npy"
    out_file = generate_out_file(out_file, out_dir, tag)
    np.save(out_file, y_valid)

    out_dir = config.SP_SPLITS_DIR + '/test'
    out_file = "X_test.pkl"
    out_file = generate_out_file(out_file, out_dir, tag)
    X_test.to_pickle(out_file)
    out_file = "y_test.npy"
    out_file = generate_out_file(out_file, out_dir, tag)
    np.save(out_file, y_test)