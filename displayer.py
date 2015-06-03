"""
displayer.py is responsible for saving the rendered animation into file.
"""

import subprocess as sp
import os
import time

import numpy as np
import pygame as pg

import configs as cfg

FFMPEG_BIN = "ffmpeg"  # on Windows


def savevideo(animation):
    """ Saves the simulation as a video """
    command = [FFMPEG_BIN,
               "-threads", "0",
               "-y",  # (optional) overwrite output file if it exists
               "-f", "rawvideo",
               "-vcodec", "rawvideo",
               "-s", str(cfg.window_size[0]) + "x" + str(cfg.window_size[1]),  # size of one frame
               "-pix_fmt", "rgb24",  # "rgb" + str(bitsize),
               "-r", "24",  # frames per second
               "-i", "-",  # The imput comes from a pipe
               "-an",  # Tells FFMPEG not to expect any audio
               "-vcodec", "mpeg4",
               # "-b:v", "2000000",
               "-qscale:v", "1", # Takes as much time, but file is bigger (bitrate is bigger)
               os.path.join(os.getcwd(), cfg.movie_url)]
    print("Saving video in", os.path.join(os.getcwd(), cfg.movie_url))
    pipe = sp.Popen(command, stdin=sp.PIPE, stdout=None, stderr=sp.STDOUT, shell=True)

    # Reference to pixel values in a surface (pointers)
    i = 0
    for frame in animation:
        pxarray = pg.surfarray.pixels3d(frame).astype(np.uint8)
        pxarray_t = np.transpose(pxarray, [1, 0, 2])
        try:
            pipe.stdin.write(pxarray_t.tostring())
        except OSError as e:
            print(str(i) + ": ")
            print(e)
        i += 1
    pipe.stdin.close()
    if pipe.stderr is not None:
        pipe.stderr.close()
    pipe.wait()

    del pipe


def saveimages(animation):
    """ Saves simulation as a series of pictures """
    i = 0

    # Clear previous images
    for filename in os.listdir(os.getcwd()):
        if filename.startswith(cfg.images_filename) and filename.endswith(cfg.images_format):
            removed = False
            while not removed:
                try:
                    os.remove(filename)
                    removed = True
                except:
                    pass
    for frame in animation:
        print("Saving image in", os.path.join(os.getcwd(),
              cfg.images_filename +
              str(int(cfg.startSecond * cfg.framespersecond) + i) + cfg.images_format))
        # saved = False
        # while (not saved):
        attempt = 0
        while attempt < 10:
            try:
                pg.image.save(frame, os.path.join(os.getcwd(),
                              cfg.images_filename +
                              str(int(cfg.startSecond * cfg.framespersecond) + i) +
                              cfg.images_format))
                break
            except:
                attempt += 1
                time.sleep(0.1)
        if attempt == 10:
            print("Failed to save image.")
        i += 1
