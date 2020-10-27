"""
Task 6 Train on Shortest Path Vectors
"""
import argparse
import config
import datetime
import numpy as np
import pandas as pd
from classify import models
from utils.file import directory_check, remove_tag_date


def arg_parse(arg_list=None):
    parser = argparse.ArgumentParser(description="Train Shortest Path Corpus")

    #### Model ####

    parser.add_argument(
        "--model-name",
        dest="model_name",
        type=str,
        help="Name of model (see classify/model_dict.py).",
    )

    parser.add_argument(
        "--cross-validation",
        "-cv",
        dest="cv",
        action="store_true",
        default=False,
        help="Train with cross-validation. Default False (train/test splits)",
    )

    #### Data Characteristics ####

    parser.add_argument(
        "--no-early-stopping",
        dest="no_early_stopping",
        action="store_true",
        default=False,
        help="Don't implement early stopping, default False",
    )

    parser.add_argument(
        "--epochs",
        "-ep",
        dest="epochs",
        type=int,
        default=100,
        help="Number of epochs, default 100",
    )

    parser.add_argument(
        "--batch-size",
        "-bs",
        dest="batch_size",
        type=int,
        default=1,
        help="Batch Size, default 1",
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
        default=config.MODEL_DIR,
        help=f"Set filepath for model export, default {config.MODEL_DIR}",
    )

    parser.add_argument(
        "--input-tag",
        "-itag",
        dest="in_tag",
        type=str,
        help="The experiment tag for the input shortest path dataframe, i.e. sp_df-<tag>.pkl",
    )

    if arg_list:
        return parser.parse_args(args=arg_list)
    else:
        return parser.parse_args()


def load_splits(dir: str, tag: str):
    train_dir = dir + "/train"
    valid_dir = dir + "/valid"
    test_dir = dir + "/test"

    X_train_file = train_dir + "/X_train-" + tag + ".pkl"
    X_valid_file = valid_dir + "/X_valid-" + tag + ".pkl"
    X_test_file = test_dir + "/X_test-" + tag + ".pkl"

    y_train_file = train_dir + "/y_train-" + tag + ".npy"
    y_valid_file = valid_dir + "/y_valid-" + tag + ".npy"
    y_test_file = test_dir + "/y_test-" + tag + ".npy"

    X_train = pd.read_pickle(X_train_file)
    X_valid = pd.read_pickle(X_valid_file)
    X_test = pd.read_pickle(X_test_file)

    y_train = np.load(y_train_file)
    y_valid = np.load(y_valid_file)
    y_test = np.load(y_test_file)

    X_train = X_train["Short_Path"].apply(pd.Series)
    X_valid = X_valid["Short_Path"].apply(pd.Series)
    X_test = X_test["Short_Path"].apply(pd.Series)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%y%m%d")

    args = arg_parse()
    assert args.model_name is not None, "Must provide name of Model"
    assert args.in_tag is not None, "Must provide tag for Training Data"

    directory_check(args.in_dir, create=False)
    directory_check(args.out_dir)
    directory_check(config.TRAIN_LOGS)

    tag = f"{args.model_name}-"
    if args.cv:
        tag += "cv-"
    if args.no_early_stopping:
        tag += "nes-"
    tag += f"{model_name}-"
    tag += f"{remove_tag_date(args.in_tag)}-{now}"

    print("train.py")
    print("-" * 30)
    print(f"Now: {now}")
    print(f"Model: {args.model_name}")
    print(f"Cross-Validation?: {args.cv}")
    print(f"Early Stopping?: {not args.no_early_stopping}")
    print(f"Epochs: {args.epochs}")
    print(f"Incoming Data: {args.in_tag}")
    print(f"Output Dir: {args.out_dir}")
    print(f"Experiment tag: {tag}")

    early_stopping = not args.no_early_stopping

    if args.cv:
        exit()
        # TODO setup cross-validation option
    else:

        train, valid, test = load_splits(args.in_dir + "/splits", args.in_tag)

        if args.model_name == "dnn_wide":

            training_params = {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "workers": 4,
            }

            encoder_file = args.in_dir + "/splits/class_encoder.npy"

            model = models.DNN_W(
                train=train,
                valid=valid,
                test=test,
                training_params=training_params,
                out_dir=args.out_dir,
                tag=tag,
                early_stopping=early_stopping,
                encoder_file=encoder_file,
            )
            model.fit()
            model.predict()
            model.report()

        else:

            # merge training and validation set for sklearn algos
            train_X = train[0].append(valid[0], ignore_index=True)
            train_y = np.concatenate((train[1], valid[1]), axis=None)
            train = (train_X, train_y)
            del valid

            model = models.model_names[model_name](
                train=train,
                test=test,
                out_dir=args.out_dir,
                tag=tag,
                encoder_file=encoder_file,
            )

            model.fit()
            model.predict()
            model.report()
