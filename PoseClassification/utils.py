from matplotlib import pyplot as plt
import numpy as np
import os


def show_image(img, figsize=(10, 10)):
    """Shows output PIL image."""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()


def create_images_folder_structure():
    cwd = os.getcwd()

    # Create the folder assets if it doesn't already exist
    if not os.path.exists("assets"):
        os.makedirs("assets")
        print(f"Created directory: {os.path.join(cwd, 'assets')}")

    # Create the folder assets/images if it doesn't already exist
    if not os.path.exists("assets/images"):
        os.makedirs("assets/images")
        print(f"Created directory: {os.path.join(cwd, 'assets/images')}")
    # Define the base directory relative to the script's location
    base_dir = os.path.join("assets", "images")
    # Folders to create inside the base directory
    folders = ["fitness_poses_images_in", "fitness_poses_images_out"]
    # Iterate over each folder and create it if it doesn't exist
    for folder in folders:
        path = os.path.join(base_dir, folder)
        if not os.path.exists(path):
            os.makedirs(path)
            # print(f"Created directory: {path}")
        else:
            pass
            # print(f"Directory already exists: {path}")

    subfolder = ["pushups_up", "pushups_down"]
    # Now we create the sub folder for classifications
    for folder in subfolder:
        for _folder in folders:
            path = os.path.join(base_dir, _folder, folder)
            if not os.path.exists(path):
                os.makedirs(path)
                # print(f"Created directory: {path}")
            else:
                pass
                # print(f"Directory already exists: {path}")


def create_docs_folder_structure():
    cwd = os.getcwd()
    if not os.path.exists("docs"):
        os.makedirs("docs")
        print(f"Created directory: {os.path.join(cwd, 'docs')}")
        print("Copy paste here the folder of images and video from kaggle")

    for folder in ["images", "videos"]:
        path = os.path.join("docs", folder)
        if not os.path.exists(path):
            os.makedirs(path)
            print(
                f"Created directory: {path} --> Copy/Paste here our images from kaggles"
            )
        else:
            print(
                f"Directory already exists: {path} --> Copy/Paste here our videos from kaggles"
            )

def assert_images_in_folder():
  # We check that the folder : "obj", "test" and the files cls.txt are in docs/images
  assert os.path.exists(os.path.join("docs", "images", "obj"))
  assert os.path.exists(os.path.join("docs", "images", "test"))
  assert os.path.exists(os.path.join("docs", "images", "cls.txt"))
  
  

def assert_videos_in_folder():
  pass

class EMADictSmoothing(object):
    """Smoothes pose classification. Exponential moving average (EMA)."""

    def __init__(self, window_size=10, alpha=0.2):
        self._window_size = window_size
        self._alpha = alpha

        self._data_in_window = []

    def __call__(self, data):
        """Smoothes given pose classification.

        Smoothing is done by computing Exponential Moving Average for every pose
        class observed in the given time window. Missed pose classes arre replaced
        with 0.

        Args:
          data: Dictionary with pose classification. Sample:
              {
                'pushups_down': 8,
                'pushups_up': 2,
              }

        Result:
          Dictionary in the same format but with smoothed and float instead of
          integer values. Sample:
            {
              'pushups_down': 8.3,
              'pushups_up': 1.7,
            }
        """
        # Add new data to the beginning of the window for simpler code.
        self._data_in_window.insert(0, data)
        self._data_in_window = self._data_in_window[: self._window_size]

        # Get all keys.
        keys = set([key for data in self._data_in_window for key, _ in data.items()])

        # Get smoothed values.
        smoothed_data = dict()
        for key in keys:
            factor = 1.0
            top_sum = 0.0
            bottom_sum = 0.0
            for data in self._data_in_window:
                value = data[key] if key in data else 0.0

                top_sum += factor * value
                bottom_sum += factor

                # Update factor.
                factor *= 1.0 - self._alpha

            smoothed_data[key] = top_sum / bottom_sum

        return smoothed_data


class RepetitionCounter(object):
    """Counts number of repetitions of given target pose class."""

    def __init__(self, class_name, enter_threshold=6, exit_threshold=4):
        self._class_name = class_name

        # If pose counter passes given threshold, then we enter the pose.
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold

        # Either we are in given pose or not.
        self._pose_entered = False

        # Number of times we exited the pose.
        self._n_repeats = 0

    @property
    def n_repeats(self):
        return self._n_repeats

    def __call__(self, pose_classification):
        """Counts number of repetitions happend until given frame.

        We use two thresholds. First you need to go above the higher one to enter
        the pose, and then you need to go below the lower one to exit it. Difference
        between the thresholds makes it stable to prediction jittering (which will
        cause wrong counts in case of having only one threshold).

        Args:
          pose_classification: Pose classification dictionary on current frame.
            Sample:
              {
                'pushups_down': 8.3,
                'pushups_up': 1.7,
              }

        Returns:
          Integer counter of repetitions.
        """
        # Get pose confidence.
        pose_confidence = 0.0
        if self._class_name in pose_classification:
            pose_confidence = pose_classification[self._class_name]

        # On the very first frame or if we were out of the pose, just check if we
        # entered it on this frame and update the state.
        if not self._pose_entered:
            self._pose_entered = pose_confidence > self._enter_threshold
            return self._n_repeats

        # If we were in the pose and are exiting it, then increase the counter and
        # update the state.
        if pose_confidence < self._exit_threshold:
            self._n_repeats += 1
            self._pose_entered = False

        return self._n_repeats
