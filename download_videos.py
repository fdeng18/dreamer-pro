# Copied from https://github.com/VinAIResearch/TPC-tensorflow

# This code downloads videos from the kinetics400 dataset that is used in the natural background setting
# videos for training and testing will be saved in kinetics400/videos/train and kinetics400/videos/test respectively

import json
import os
import sys
import ssl
from tqdm import trange
from pytube import YouTube


ssl._create_default_https_context = ssl._create_stdlib_context

train_path = "./kinetics400/kinetics400/train.json"
test_path = "./kinetics400/kinetics400/test.json"


def get_url(path):
    with open(path) as f:
        data = json.load(f)
    urls = []
    for k in data.keys():
        if data[k]["annotations"]["label"] == "driving car":
            urls.append(data[k]["url"])
    return urls


def download(urls, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    with trange(len(urls)) as t:
        t.set_description("Downloading video")
        for i in t:
            try:
                url = urls[i]
                video = YouTube(url)
                streams = video.streams.filter(file_extension="mp4")
                for stream in streams:
                    if stream.resolution == "360p":
                        itag = stream.itag
                        break
                video.streams.get_by_itag(itag).download(dest_path)
            except Exception:
                print (sys.exc_info())
                continue


train_urls = get_url(train_path)
test_urls = get_url(test_path)
download(train_urls, "./kinetics400/videos/train/")
download(test_urls, "./kinetics400/videos/test/")
