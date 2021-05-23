import os
import re
import random
import urllib.request
from zipfile import ZipFile
from shutil import copyfile

# file download
THE_LOCAL_DOWNLOAD_FOLDER = "train-data"
DOWNLOAD_FILENAME = "train-data/images-n-labels.zip"
EXTRACT_FOLDER = "train-data/images-n-labels"
# the custom train data
THE_CUSTOM_FOLDER = "../data/custom"
IMAGES_FOLDER = "../data/custom/images"
LABELS_FOLDER = "../data/custom/labels"
TRAIN_SET_TXT = "../data/custom/train.txt"
VALID_SET_TXT = "../data/custom/valid.txt"
CLASS_NAME_FILE = "../data/custom/classes.names"
DATA_SET_PATH_PREFIX = "data/custom/images/"
# the custom train config
THE_BACKUP_CONFIG_FOLDER = "config"
THE_CONFIG_FOLDER = "../config"
TRAIN_CONFIG_FILE = "../config/yolov3-custom.cfg"


def is_daytime(h, m):
    time = h*100 + m
    return time > 830 and time < 2130


def prepare_config_files(clear_current=False):
    # make sure all folders exists
    folders = [THE_CUSTOM_FOLDER, IMAGES_FOLDER,
               LABELS_FOLDER, THE_CONFIG_FOLDER, THE_LOCAL_DOWNLOAD_FOLDER]
    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)

    # delete current config files
    if clear_current:
        print("deleting current config files...")
        files = [TRAIN_SET_TXT, VALID_SET_TXT,
                 CLASS_NAME_FILE, TRAIN_CONFIG_FILE]
        for file in files:
            try:
                os.remove(file)
                print(f"deleted {file}.")
            except:
                pass

    # copy config files
    config_files = [CLASS_NAME_FILE, TRAIN_CONFIG_FILE]
    print("copying config files...")
    for file in config_files:
        try:
            src = os.path.join(THE_BACKUP_CONFIG_FOLDER,
                               re.search("[^/]+$", file)[0])
            copyfile(src, file)
            print(f"copied {src} to {file}.")
        except:
            pass


def run():
    #
    # collect some arguments
    #
    day_night_filter = int(
        input("Filter data? NoFilter(0), DayTime(1), NightTime(2), default 0: ") or "0")

    validation_ratio = float(
        input("Percent of data for validation, default 20:") or "20") / 100

    clear_current_train_data = input(
        "Clear current train data & config? Y/N, default Y:") or "Y"

    print(
        f"Review input args: {day_night_filter}, {validation_ratio}, {clear_current_train_data}.\n")

    #
    # clear current train data & config
    #
    prepare_config_files(clear_current_train_data == "Y")

    #
    # download data
    #
    if os.path.exists(DOWNLOAD_FILENAME):
        print(
            f"train data file {DOWNLOAD_FILENAME} exists, skipped the download.")
    else:
        dataUrl = input("data zip url: ")
        print("downloading train data...")
        urllib.request.urlretrieve(dataUrl, DOWNLOAD_FILENAME)

    #
    # extract data
    #
    if os.path.exists(EXTRACT_FOLDER):
        print(
            f"train data folder {EXTRACT_FOLDER} exists, skipped the extraction.")
    else:
        print("extacting...")
        with ZipFile(DOWNLOAD_FILENAME, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(EXTRACT_FOLDER)

    #
    # moving data files & creating train/validate list
    #
    print("moving data files & creating train.txt and valid.txt...")

    train_set = []
    valid_set = []

    for dirName, subdirList, fileList in os.walk(EXTRACT_FOLDER):
        if dirName == EXTRACT_FOLDER:
            for label_filename in fileList:
                # only loop the label files
                # (assuming a jpeg may not have a label, but a label always have a jpeg)
                if re.search(".jpg$", label_filename):
                    continue

                m = re.search("_(\d{2})(\d{2})(\d{2})\..+?$", label_filename)
                if not m:
                    continue
                hh = int(m[1])
                mm = int(m[2])
                ss = int(m[3])
                image_in_daytime = is_daytime(hh, mm)
                # print(hh, mm, ss, "is daytime", image_in_daytime)

                # filenames
                label_filepath = os.path.join(dirName, label_filename)
                jpeg_filename = label_filename.replace("txt", "jpg")
                jpeg_filepath = os.path.join(dirName, jpeg_filename)

                # move the jpg file to the data/custom/images folder
                target1 = os.path.join(IMAGES_FOLDER, jpeg_filename)
                if not os.path.exists(target1):
                    copyfile(jpeg_filepath, target1)

                # move the label file to the data/custom/labels folder
                target2 = os.path.join(LABELS_FOLDER, label_filename)
                if not os.path.exists(target2):
                    copyfile(label_filepath, target2)

                # skip adding the file to train/valid list based on filters
                if day_night_filter == 1 and not image_in_daytime:
                    continue

                if day_night_filter == 2 and image_in_daytime:
                    continue

                # add the file to train/valid list based on validation ratio
                if random.random() < validation_ratio:
                    valid_set.append(jpeg_filename)
                else:
                    train_set.append(jpeg_filename)

    # print(train_set, valid_set)
    print(
        f"train set size: {len(train_set)}, valid set size: {len(valid_set)}")

    # writing train.txt & valid.txt
    f = open(TRAIN_SET_TXT, "w+")
    f.write("\n".join([DATA_SET_PATH_PREFIX + item for item in train_set]))
    f.close()

    f = open(VALID_SET_TXT, "w+")
    f.write("\n".join([DATA_SET_PATH_PREFIX + item for item in valid_set]))


if __name__ == "__main__":
    run()
