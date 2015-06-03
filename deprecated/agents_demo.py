import os
import pickle

import numpy as np

import configs as cfg
import maze


class Boid:
    """ This class defines the boid """

    def __init__(self):
        self.orientation = np.float32(0)  # TODO: randomize
        self.arraySize = cfg.ARRDIM * np.dtype(np.float32).itemsize * cfg.Dimensions
        if cfg.bounding_rects_show:
            self.collided = False

    @staticmethod
    def init_boid(amaze):
        pos = np.zeros(cfg.Dimensions, dtype=np.float32)
        # Start from  a random cell
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

        return np.asarray([pos, vel])


class Flock:
    """ Arrays contain data that will be processed by GPU.
        Objects contain all the other data
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
    if len(flock["flock"].object_list) < cfg.NumberOfBoids:
        print("Not enough boids in the pickle. Generating the missing ones.")
        extended_flock = True
        additional_boids = generate_agents(
            cfg.NumberOfBoids - len(flock["flock"].object_list), amaze)
        flock["flock"].object_list.extend(additional_boids.object_list)
        flock["flock"].np_arrays = np.concatenate([flock["flock"].np_arrays,
                                                   additional_boids.np_arrays])

    if cfg.run_demo_program:
        if len(flock["predators"].object_list) < cfg.NumberOfPredators:
            print("Not enough predators in the pickle. Generating the missing ones.")
            extended_flock = True
            additional_predators = generate_predators(
                cfg.NumberOfPredators - len(flock["predators"].object_list), amaze)
            flock["predators"].object_list.extend(additional_predators.object_list)
            flock["predators"].np_arrays = np.concatenate([flock["predators"].np_arrays,
                                                           additional_predators.np_arrays])

    if len(flock["flock"].object_list) > cfg.NumberOfBoids:
        print("More boids than necessary. Deleting the missing ones")
        flock["flock"].object_list = flock["flock"].object_list[:cfg.NumberOfBoids]
        flock["flock"].np_arrays = np.resize(flock["flock"].np_arrays,
                                             (cfg.NumberOfBoids, cfg.ARRDIM, cfg.Dimensions))
    if cfg.run_demo_program:
        if len(flock["predators"].object_list) > cfg.NumberOfPredators:
            print("More predators than necessary. Deleting the missing ones")
            flock["predators"].object_list = flock["predators"].object_list[:cfg.NumberOfPredators]
            flock["predators"].np_arrays = np.resize(flock["predators"].np_arrays,
                                                     (cfg.NumberOfPredators, cfg.ARRDIM, cfg.Dimensions))

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
    flocks[0]["flock"] = generate_agents(cfg.NumberOfBoids, amaze)
    if cfg.run_demo_program:
        flocks[0]["predators"] = generate_predators(cfg.NumberOfPredators, amaze)

    if cfg.save_agents:
        pickle_agents(flocks[0])
    return flocks


def generate_agents(n, amaze):
    """ Generates agents """
    flock = Flock(n)
    flock.init_array(amaze)
    return flock


def generate_predators(n, amaze):
    """ Generates predators """
    predators = Flock(n)
    predators.init_array(amaze)
    # Predators are centered around starting positions
    # noinspection PyTypeChecker
    for predatorN in range(len(predators.np_arrays)):
        k = 0.2
        rand = np.random.uniform(cfg.center[0] - cfg.center[0] * k, cfg.center[1] + cfg.center[1] * k, cfg.Dimensions)
        for i in range(cfg.Dimensions):
            predators.np_arrays[predatorN][cfg.BOID_POS_VAR][i] = rand[i]
    return predators
