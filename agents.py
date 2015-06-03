"""
agents.py contains all agent-related data structures and functions.
"""
import os
import pickle

import numpy as np

import configs as cfg
import maze


class Boid:
    """ This class defines the boid """
    arrayLen = cfg.ARRDIM * cfg.Dimensions + 1
    arraySize = np.dtype(np.float32).itemsize * arrayLen

    def __init__(self):
        self.orientation = np.float32(0)  # TODO: randomize
        if cfg.bounding_rects_show:
            self.collided = False

    @staticmethod
    def init_boid(amaze):
        pos = np.zeros(cfg.Dimensions, dtype=np.float32)
        # Start from a random cell
        pos[0] = np.float32(np.trunc(np.random.uniform(0, cfg.maze_width)))
        pos[1] = np.float32(np.trunc(np.random.uniform(0, cfg.maze_height)))

        # Ensure that we are not placing the boid into a wall ---------------------
        # Change position until hit a free cell
        while isinstance(
                amaze.matrix[int(pos[0])][int(pos[1])].shape,
                maze.Wall):
            # While the cell is filled with a wall (==1)
            pos[0] = np.float32(np.trunc(np.random.uniform(0, cfg.maze_width)))
            pos[1] = np.float32(np.trunc(np.random.uniform(0, cfg.maze_height)))
            # Check that we are not placing the boind into the wall

        # Move boid to the center of the cell
        pos[0] += 0.5
        pos[1] += 0.5
        vel = np.zeros(cfg.Dimensions, dtype=np.float32)

        # TODO: change pheromone_level type to uint16
        pheromone_level = np.zeros([1], dtype=np.float32)

        return np.concatenate((pos, vel, pheromone_level))


class Flock:
    """ Arrays contain data that will be processed by GPU.
        Objects contain all the other data.
        """
    size = 0

    def __init__(self, size):
        self.size = size
        self.np_arrays = None
        self.object_list = []
        for i in range(size):
            boid = Boid()
            self.object_list.append(boid)

    def init_array(self, amaze):
        array_list = []
        for i in range(self.size):
            array_list.append(Boid.init_boid(amaze))
        self.np_arrays = np.asarray(array_list)

    def init_empty_array(self):
        self.np_arrays = np.zeros((self.size, Boid.arrayLen), dtype=np.float32)


def load_from_pickle(amaze):
    """ Loads previously generated flock from a pickle file """
    print("Unpickling the flock.")
    try:
        pickle_f = open(os.path.join(os.getcwd(), cfg.flock_pickle_url), "rb")
        flock = pickle.load(pickle_f)
        pickle_f.close()
    except (IOError, EOFError):
        print("Failed to load the pickle", os.path.join(os.getcwd(), cfg.flock_pickle_url))
        return None

    extended_flock = False
    if len(flock.object_list) < cfg.NumberOfBoids:
        print("Not enough boids in the pickle. Generating the missing ones.")
        extended_flock = True
        additional_boids = generate_agents(
            cfg.NumberOfBoids - len(flock.object_list), amaze)
        flock.object_list.extend(additional_boids.object_list)
        flock.np_arrays = np.concatenate([flock.np_arrays,
                                          additional_boids.np_arrays])

    if len(flock.object_list) > cfg.NumberOfBoids:
        print("More boids than necessary. Deleting the unnecessary ones")
        flock.object_list = flock.object_list[:cfg.NumberOfBoids]
        flock.np_arrays = np.resize(flock.np_arrays,
                                    (cfg.NumberOfBoids, Boid.arrayLen))

    if extended_flock and cfg.save_agents:
        pickle_agents(flock)

    flocks = [flock]
    return flocks


def pickle_agents(flock):
    """ Pickles agents inside a file """
    print("Pickling the flock.")
    try:
        pickle_f = open(os.path.join(os.getcwd(), cfg.flock_pickle_url), "wb")
        pickle.dump(flock, pickle_f)
        pickle_f.close()
    except IOError:
        print("Failed to save pickle into", os.path.join(os.getcwd(), cfg.flock_pickle_url))


def generate_first_flock(amaze):
    """ Generates the first instances of agents """
    print("Generates the first flock.")
    flocks = [{}]
    flocks[0] = generate_agents(cfg.NumberOfBoids, amaze)

    if cfg.save_agents:
        pickle_agents(flocks[0])
    return flocks


def generate_agents(n, amaze):
    """ Generates agents """
    flock = Flock(n)
    flock.init_array(amaze)
    return flock
