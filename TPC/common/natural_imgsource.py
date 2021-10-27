# Copied from https://github.com/VinAIResearch/TPC-tensorflow

# This code provides the class that is used to generate backgrounds for the natural background setting
# the class is used inside an environment wrapper and will be called each time the env generates an observation
# the code is largely based on https://github.com/facebookresearch/deep_bisim4control

import random

import cv2
import numpy as np
import skvideo.io


class ImageSource(object):
    """
    Source of natural images to be added to a simulated environment.
    """

    def get_image(self):
        """
        Returns:
            an RGB image of [h, w, 3] with a fixed shape.
        """
        pass

    def reset(self):
        """ Called when an episode ends. """
        pass


class RandomVideoSource(ImageSource):
    def __init__(self, shape, filelist, random_bg=False, max_videos=100, grayscale=False):
        """
        Args:
            shape: [h, w]
            filelist: a list of video files
        """
        self.grayscale = grayscale
        self.shape = shape
        self.filelist = filelist
        random.shuffle(self.filelist)
        self.filelist = self.filelist[:max_videos]
        self.max_videos = max_videos
        self.random_bg = random_bg
        self.current_idx = 0
        self._current_vid = None
        self.reset()

    def load_video(self, vid_id):
        fname = self.filelist[vid_id]
        if self.grayscale:
            frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
        else:
            frames = skvideo.io.vread(fname, num_frames=1000)
        img_arr = np.zeros((frames.shape[0], self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
        for i in range(frames.shape[0]):
            img_arr[i] = cv2.resize(
                frames[i], (self.shape[1], self.shape[0])
            )  # THIS IS NOT A BUG! cv2 uses (width, height)
        return img_arr

    def reset(self):
        del self._current_vid
        while True:
            try:
                self._video_id = np.random.randint(0, len(self.filelist))
                self._current_vid = self.load_video(self._video_id)
                break
            except Exception:
                continue
        self._loc = np.random.randint(0, len(self._current_vid))

    def get_image(self):
        if self.random_bg:
            self._loc = np.random.randint(0, len(self._current_vid))
        else:
            self._loc += 1
        img = self._current_vid[self._loc % len(self._current_vid)]
        return img
