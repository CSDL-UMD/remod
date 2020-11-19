import pathlib
import csv
import os


def get_filename(file_path: str, drop_extention: bool = True) -> str:
    """
    file_path: A system file path
    drop_extention: if True, drops the file extention at the end of the string

    Returns a string that represents the filename
    """
    if drop_extention:
        return pathlib.Path(file_path).stem
    else:
        return pathlib.Path(file_path).name


def dict_to_csv(file_path: str, d: dict) -> None:
    """
    file_path: A path to write a dictionary to. Keys will be rows
    d: The dictionary to be written
    """

    with open(file_path, "w") as out:
        csvwriter = csv.writer(out)
        count = 0
        for entry in d:
            if count == 0:
                header = entry.keys()
                csvwriter.writerow(header)
                count += 1
            csvwriter.writerow(entry.values())

        print(f"Finished writing dictionary to {file_path}")


def absolute_paths(dir_path: str) -> list:
    """
    Returns a list of absolute filepaths for files in dir_path
    """
    return [str(dir_path + "/" + x) for x in os.listdir(dir_path)]


def directory_check(dir_path: str, create: bool = True) -> None:
    """
    Checks if directory exists. If create=True, creates it. Otherwise raise exception
    """
    if not os.path.exists(dir_path):
        if create:
            os.makedirs(dir_path)
        else:
            raise Exception(f"This directory, {dir_path}, does not exist. Set create=True to make it")


def generate_out_file(filename, dir, tag = None):
    """Generates a full output path for a given file

    Arguments:
        filename {str} -- Filename with extension i.e this_graph.txt
        dir {str} -- Directory to save file to
        tag {str} -- Unique Experiment tag to be appended

    Returns:
        str -- Full file path
    """
    directory_check(dir)

    name = get_filename(filename)
    extension = "." + filename.split(".")[1]

    full_path = (dir + "/" + name + "-" + tag + extension) if tag is not None else (dir + "/" + name + extension)

    return full_path

def get_experiment_tag(filename: str) -> str:
    """Return trailing experiment tag. i.e. model-full-d256-wl50-nw200-win15-p4.0-q4.0-200930.pkl returns full-d256-wl50-nw200-win15-p4.0-q4.0

    Args:
        filename (str): RESOGE filepath

    Returns:
        str: The experimental tag associated with filename
    """

    name = get_filename(filename)
    return ('-').join(name.split('-')[1:-1])

def remove_tag_date(tag: str) -> str:
    """Return trailing experiment tag without date. i.e. model-full-d256-wl50-nw200-win15-p4.0-q4.0-200930.pkl returns model-full-d256-wl50-nw200-win15-p4.0-q4.0

    Args:
        tag (str): RESOGE tag

    Returns:
        str: The experimental tag without the date
    """
    return ('-').join(tag.split('-')[:-1])