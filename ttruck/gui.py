# the framework modules
# import time
import threading
# from multiprocessing import Queue, Process
import tkinter as tk
from tkinter import filedialog

# the image plot related modules
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import random
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from torch.utils import data

# the yolo v3 module
from pytorchyolo import detect2 as detect
# from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, to_cpu, print_environment_info
# from pytorchyolo.utils.datasets import ImageFolder
# from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

# the detection arguments
IMAGE_FOLDER_PATH = 'C:\\max\\truck\\testimages\\'
WEIGHTS_FILE_PATH = 'C:\\max\\truck\\training\\yolo3-256-16\\yolov3_ckpt_660.pth'
CLASSES_FILE_PATH = 'ttruck\\config\\classes.names'
DEFAULT_CONF_THRES = '0.1'
IMAGE_SIZE = 416


# to compose the detect.py command line arguments...
def getDetectpyCommandline(data_folder=IMAGE_FOLDER_PATH, weights_path=WEIGHTS_FILE_PATH, conf_thres=DEFAULT_CONF_THRES):
    return [
        # 'yolo-detect',
        '--images', data_folder, '--model', 'ttruck/config/yolov3-custom.cfg', '--classes',
        'ttruck/config/classes.names', '--weights', weights_path, '--conf_thres', conf_thres]


# to fast create a frame...
class NewFrame():
    def __init__(self, master, pady=5):
        self.master = master
        self.pady = pady

    def __enter__(self):
        self.frame = tk.Frame(self.master)
        return self.frame

    def __exit__(self, *argv):
        self.frame.pack(pady=5)


# the main window


class MainWindow:
    def __init__(self):
        self.m = tk.Tk()
        self.m.title('Tesla Truck Detector')
        self.m.geometry('640x640')

        # init the detection lib
        detect.reset_detection()

        # lib
        self.gmail_server = None

        # app states
        self.canvasFrame = None
        self.canvas = None
        self.plot = None
        self.figure = None
        self.detectionThreadCreated = False

        # detection & ui interaction
        self.data_folder = None
        self.image_count = 0
        self.truck_image_count = 0
        self.processed_image_count = 0
        self.show_image_index = -1
        self.show_truck_image_index = -1

        # tkinter variable bindings
        self.confThres = tk.DoubleVar(value=DEFAULT_CONF_THRES)
        self.detectionState = tk.StringVar()
        self.detectionProgress = tk.StringVar()
        self.imageIndexLabel = tk.StringVar(value="Image Index: 0/0")
        self.truckImageIndexLabel = tk.StringVar(
            value="Truck Image Index: 0/0")
        self.currentShowImagePath = tk.StringVar(value="Image Path: N/A")

        # detection classes
        self.classes = load_classes(CLASSES_FILE_PATH)  # List of class names

        #
        # create UI elements
        #

        with NewFrame(self.m) as frame:
            self.startButton = tk.Button(
                frame, text='Start Dection', command=self.start_detection)
            self.pauseButton = tk.Button(
                frame, text='Pause Dection', command=self.pause_detection)
            self.quitButton = tk.Button(
                frame, text='Quit', command=self.quit)
            self.startButton.pack(side=tk.LEFT, padx=5)
            self.pauseButton.pack(side=tk.LEFT, padx=5)
            self.pauseButton.config(state='disabled')
            # self.quitButton.pack(side=tk.LEFT)

        # with NewFrame(self.m) as frame:
        #     tk.Label(frame, text="Conf Thres:").pack(side=tk.LEFT)
        #     tk.Entry(frame, textvariable=self.confThres).pack(side=tk.LEFT)

        with NewFrame(self.m) as frame:
            tk.Label(frame, textvariable=self.detectionState,
                     fg='green').pack(side=tk.LEFT)
            tk.Label(frame, textvariable=self.detectionProgress,
                     fg='green').pack(side=tk.LEFT)

        with NewFrame(self.m) as frame:
            self.canvasFrame = frame
            self.update_canvas_area()

        with NewFrame(self.m) as frame:
            tk.Label(frame, textvariable=self.currentShowImagePath).pack(
                side=tk.LEFT)

        with NewFrame(self.m) as frame:
            tk.Label(frame, textvariable=self.imageIndexLabel).pack(
                side=tk.LEFT, padx=5)
            tk.Button(
                frame, text='Prev (←)', command=self.show_prev_image).pack(side=tk.LEFT, padx=2)
            tk.Button(
                frame, text='Next (→)', command=self.show_next_image).pack(side=tk.LEFT, padx=2)
            tk.Label(frame, text="    ").pack(side=tk.LEFT)
            tk.Label(frame, textvariable=self.truckImageIndexLabel).pack(
                side=tk.LEFT, padx=5)
            tk.Button(
                frame, text='Prev (↑)', command=self.show_prev_truck_image).pack(side=tk.LEFT, padx=2)
            tk.Button(
                frame, text='Next (↓)', command=self.show_next_truck_image).pack(side=tk.LEFT, padx=2)
            # bind key board events
            self.m.bind('<Left>', self.show_prev_image)
            self.m.bind('<Right>', self.show_next_image)
            self.m.bind('<Up>', self.show_prev_truck_image)
            self.m.bind('<Down>', self.show_next_truck_image)

        self.start_ui_refresh_timer()

        self.m.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.m.mainloop()

    def quit(self):
        self.on_closing()

    def on_closing(self):
        detect.terminate_detection()
        self.m.destroy()

    def start_ui_refresh_timer(self):

        def timer():
            detection_state = detect.get_running_state()
            image_count = detection_state["image_count"]
            current_index = detection_state["current_index"]
            truck_image_list = detection_state["target_image_list"]
            processed_image_list = detection_state["processed_image_list"]
            truck_image_count = len(truck_image_list)
            processed_image_count = len(processed_image_list)

            # update class member
            self.image_count = image_count
            self.truck_image_count = truck_image_count
            self.processed_image_count = processed_image_count

            # update app state texts
            if not detection_state["paused"]:
                self.detectionProgress.set(
                    '{}/{}/{}'.format(truck_image_count, current_index, image_count))

                if image_count == 0:
                    self.detectionState.set('Loading data...')
                elif current_index == 0 or current_index < image_count:
                    self.detectionState.set('Detecting...')
                else:
                    self.detectionState.set('Detection completed.')
            else:
                if image_count == 0:
                    self.detectionState.set(
                        'Press the "Start Dection" button to detect.')
                else:
                    self.detectionState.set('Detection paused.')

            # update image index label
            self.update_image_index_labels()

            # trigger the timer again
            self.m.after(1000, timer)

        # run first time
        timer()

    def get_image_path_by_index(self, index):
        detection_state = detect.get_running_state()
        image_list = detection_state["processed_image_list"]
        if index >= 0 and index < len(image_list):
            return image_list[index]
        return None

    def image_index_to_truck_image_index(self, image_index):
        detection_state = detect.get_running_state()
        processed_image_list = detection_state["processed_image_list"]
        detections = detection_state["detections"]
        if image_index < 0 or image_index >= len(processed_image_list):
            return None
        return detections[processed_image_list[image_index]]["detected_index"]

    def truck_image_index_to_image_index(self, truck_image_index):
        detection_state = detect.get_running_state()
        truck_image_list = detection_state["target_image_list"]
        if truck_image_index < 0 or truck_image_index >= len(truck_image_list):
            return None
        return truck_image_list[truck_image_index]["image_index"]

    def update_image_index_labels(self):
        self.imageIndexLabel.set(
            'Image Index: {}/{}'.format(self.show_image_index+1, self.processed_image_count))
        self.truckImageIndexLabel.set(
            'Truck Image Index: {}/{}'.format(self.show_truck_image_index+1, self.truck_image_count))

    def update_canvas_area(self):
        image_path = self.get_image_path_by_index(self.show_image_index)
        detections = None
        # update image path label
        if image_path:
            self.currentShowImagePath.set('{}'.format(image_path))
            detection_state = detect.get_running_state()
            all_detections = detection_state["detections"]
            if image_path in all_detections:
                detections = all_detections[image_path]["detections"]
        # draw the canvas
        self.plot_image_with_detections(image_path, detections)

    def plot_image_with_detections(self, image_path, detections):
        # clear the previous canvas
        frame = self.canvasFrame
        # for widget in frame.winfo_children():
        #     widget.destroy()

        # image_path = "C:\\max\\truck\\testimages\\1212_20201005_043846.jpg"

        # init the canvas if necessary
        if self.canvas is None:
            f = Figure()

            canvas = FigureCanvasTkAgg(f, master=frame)
            canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
            canvas._tkcanvas.pack(side="top", fill="both", expand=1)

            self.figure = f
            self.canvas = canvas

        # add image if available
        if image_path:
            self.figure.clf()
            ax = self.figure.add_subplot(111)
            ax.axis('off')
            img = np.array(Image.open(image_path))
            ax.imshow(img)

            # add predictions if available
            if detections is not None:
                img_size = IMAGE_SIZE
                classes = self.classes

                detections = detections.clone()

                detections = rescale_boxes(detections, img_size, img.shape[:2])

                # max note 20210809:
                # to fix the following bug:
                # Can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first
                detections = detections.cpu()

                unique_labels = detections[:, -1].unique()
                n_cls_preds = len(unique_labels)
                # Bounding-box colors
                cmap = plt.get_cmap("tab20b")
                colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
                bbox_colors = random.sample(colors, n_cls_preds)

                for x1, y1, x2, y2, conf, cls_pred in detections:

                    print(
                        f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(
                        np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle(
                        (x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0})

                # Save generated image with detections
                plt.axis("off")
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())

        # update the canvas
        self.canvas.draw()

    def start_detection(self):
        if self.data_folder is None:
            self.data_folder = filedialog.askdirectory()

        self.startButton.config(state='disabled')
        self.pauseButton.config(state='active')

        detect.start_detection(True)

        if not self.detectionThreadCreated:
            threading.Thread(target=self.detect_thread_function).start()
            self.detectionThreadCreated = True

    def pause_detection(self):
        self.startButton.config(state='active')
        self.pauseButton.config(state='disabled')

        detect.start_detection(False)

    def terminate_detection(self):
        detect.terminate_detection()

    def detect_thread_function(self):
        detect.run(getDetectpyCommandline(data_folder=self.data_folder))
        # print('detection thread started.')

    def show_prev_image(self, event=None):
        if self.show_image_index > 0:
            self.show_image_index -= 1
            truck_image_index = self.image_index_to_truck_image_index(
                self.show_image_index)
            if truck_image_index is not None:
                self.show_truck_image_index = truck_image_index
            self.update_image_index_labels()
            self.update_canvas_area()

    def show_next_image(self, event=None):
        if self.show_image_index < self.processed_image_count-1:
            self.show_image_index += 1
            truck_image_index = self.image_index_to_truck_image_index(
                self.show_image_index)
            if truck_image_index is not None:
                self.show_truck_image_index = truck_image_index
            self.update_image_index_labels()
            self.update_canvas_area()

    def show_prev_truck_image(self, event=None):
        if self.show_truck_image_index > 0:
            self.show_truck_image_index -= 1
            image_index = self.truck_image_index_to_image_index(
                self.show_truck_image_index)
            if image_index is not None:
                self.show_image_index = image_index
            self.update_image_index_labels()
            self.update_canvas_area()

    def show_next_truck_image(self, event=None):
        if self.show_truck_image_index < self.truck_image_count-1:
            self.show_truck_image_index += 1
            image_index = self.truck_image_index_to_image_index(
                self.show_truck_image_index)
            if image_index is not None:
                self.show_image_index = image_index
            self.update_image_index_labels()
            self.update_canvas_area()


def run():
    window = MainWindow()


if __name__ == '__main__':
    run()
