import os
import sys

#
# this program entry file is to replace the original poetry entries
# updated 20210808
#
# usage:
# python run.py train ...argv (eq. to poetry run train ...argv)
# python run.py detect ...argv (eq. to poetry run detect ...argv)
# python run.py gui (eq. to poetry run gui)
#


def main():
    action = sys.argv[1]
    argv = sys.argv[2:]

    entry = None

    if action == "train":
        from pytorchyolo.train import run
        entry = run
        argv = ["train.py", *argv]

    elif action == "detect":
        from pytorchyolo.detect import run
        entry = run
        argv = ["detect.py", *argv]

    elif action == "gui":
        from ttruck.gui import run
        entry = run
        argv = ["gui.py", *argv]

    else:
        print("\n\ninvalid command:", " ".join(sys.argv))
        exit()

    # setup the environment
    sys.path.append(os.getcwd())
    sys.argv = argv

    # call the real program entry
    print("\n====================\nThe target command:\n{}\n====================\n".format(
        " ".join(sys.argv)))

    entry()


if __name__ == '__main__':
    main()
