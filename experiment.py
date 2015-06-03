"""
experiment.py allows to run the simulation multiple times using different settings
and record collected data in CSV files for further processing.
Currently the module needs to be debugged. When finished, insert the following in
main.py before simulation is initiated:

for swarm_size in swarm_sizes:
    for maze_size in maze_sizes:
        for _ in range(repeats):
            cfg.NumberOfBoids = swarm_size
            cfg.maze_width = maze_size[0]
            cfg.maze_height = maze_size[1]
            cfg.goal_node = (cfg.maze_width - 2, cfg.maze_height - 1)
            cfg.window_size = cfg.window_width, cfg.window_height = \
                cfg.maze_width * cfg.tile_width, cfg.maze_height * cfg.tile_height
            print("Swarm size = %d agents" % swarm_size)
            print("Maze size = %d x %d nodes" % (maze_size[0], maze_size[1]))

Insert the following after simulation is finished:

            experiment.report(queue, buffers, swarm_size,
                              maze_size[0], maze_size[1],
                              completion_time, t2 - t1, solver)
"""
import csv
from time import strftime
import os
import numpy as np

import opencl_computations as cl_comp
import configs as cfg


def get_tasks():
    """swarm_sizes = [20, 30, 40, 50, 60]
    maze_sizes = [(21, 21), (31, 31), (41, 41), (51, 51), (61, 61)]
    repeats = 2
    solver = "CPU" """
    swarm_sizes = [21]
    maze_sizes = [(41, 41)]
    repeats = 1
    solver = "CPU"
    return swarm_sizes, maze_sizes, repeats, solver


def report(queue, buffers, n_boids, maze_w, maze_h, completion_time, computation_time, solver):
    if not os.path.exists(cfg.reporting_dir):
        os.makedirs(cfg.reporting_dir)
    # Get experiment data
    print("Transferring experiment data from device to the host")

    # Collisions
    experiment_data = cl_comp.get_experiment_data(queue, buffers)
    queue.finish()
    csvfile = open(os.path.join(cfg.reporting_dir, ("%d_%d_%d" % (n_boids,
                                                                  maze_w,
                                                                  maze_h)) +
                                strftime("%Y%m%d_%H%M%S") + "_" +
                                cfg.collisions_csv), 'w')
    csvwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n',
                           quotechar='|', quoting=csv.QUOTE_MINIMAL)
    print("Saving the experiment data in CSV file.")
    results = []

    for boid_experiment in experiment_data:
        csvwriter.writerow([(str(num) if num != 0 else "") for num in boid_experiment[1:]])

    for boid_experiment in experiment_data:
        for collision_time in boid_experiment[1:]:
            if results[int(np.trunc(collision_time / cfg.framespersecond))] is None:
                results[int(np.trunc(collision_time / cfg.framespersecond))] = 0
            results[np.trunc(collision_time / cfg.framespersecond)] += 1

    csvfile.close()

    # Configurations
    csvfile = open(os.path.join(cfg.reporting_dir, cfg.configurations_csv), 'a')
    csvwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n',
                           quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow([strftime("%Y%m%d_%H%M%S")] +
                       [n_boids] +
                       [maze_w] +
                       [maze_h] +
                       [completion_time] +
                       [computation_time] +
                       [solver])
    csvfile.close()
