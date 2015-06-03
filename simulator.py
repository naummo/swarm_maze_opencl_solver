"""
simulator.py runs the simulation - see run(). It runs agent AIs on GPU,
collects the results and generates the environment response.
"""

import numpy as np

import configs as cfg
import maze_solver
import opencl_computations as cl_comp
import collision_detection as coll_detect
import maze

x_var = cfg.X
y_var = cfg.Y
pos = cfg.BOID_POS_VAR * cfg.Dimensions
vel = cfg.BOID_VEL_VAR * cfg.Dimensions


class SensorReadings():
    def __init__(self, northern_reading, eastern_reading,
                 southern_reading, western_reading, bottom_reading):
        self.northern_reading = northern_reading
        self.eastern_reading = eastern_reading
        self.southern_reading = southern_reading
        self.western_reading = western_reading
        self.bottom_reading = bottom_reading


def run(context, device, queue, amaze, flocks, template_triangles):
    """ Simulates agents behaviour in a given maze """

    completion_time = 0
    solver = None
    # Agents-related OpenCL computations
    kernels, gpu_params = cl_comp.build_opencl_program(context, device)

    buffers, amendments, global_map = cl_comp.prepare_host_memory(context)

    cl_comp.set_kernel_arguments(kernels, buffers)

    cl_comp.prepare_device_memory(queue, kernels, buffers, flocks[0])

    # Calculate the orientations for the first flock
    i = 0
    first_flock = flocks[0]
    for boid in first_flock.np_arrays:
        calculate_orientation(boid, first_flock.object_list[i], first_flock.object_list[i])
        i += 1

    amendments.clear()
    # Send first simulation response to the device based on starting values.
    # It will be just sensor readings for starting position.
    intermediary_events = simulation_response(queue, kernels, gpu_params,
                                              buffers, amendments,
                                              first_flock, amaze)

    print("Starting the simulation.")
    for step in range(1, cfg.total_timesteps):
        # print("Computing flock N {0} on GPU.".format(step))
        # Get agents' impulses and partially correct simulation response
        # (in the form of positions)
        cl_comp.gpu_generate_next_flock(step, queue, intermediary_events, kernels,
                                        gpu_params, buffers, flocks, global_map)
        # print(flocks[step].np_arrays)
        # Fix NaN's
        cl_comp.standardize_values(flocks[step])

        # Select flocks
        current_flock = flocks[step]
        previous_flock = flocks[step - 1]

        # Calculate boids' orientations
        i = 0
        for boid in current_flock.np_arrays:
            # TODO accelerate orientation calculation using OpenCL
            calculate_orientation(boid, current_flock.object_list[i], previous_flock.object_list[i])
            i += 1

        # Check for wall collisions and update simulation response accordingly
        if cfg.collision_detection_on:
            coll_detect.run(current_flock, previous_flock, amaze, template_triangles, amendments)

        # Send simulation response to the device
        intermediary_events = simulation_response(queue, kernels, gpu_params,
                                                  buffers, amendments,
                                                  current_flock, amaze)

        # Update map (might even happen during sending simulation response to the device
        cl_comp.update_map(queue, kernels, intermediary_events)

        path_to_goal = maze_solver.solve_maze(global_map[step],
                                              amaze, cfg.starting_node,
                                              cfg.goal_node)
        if path_to_goal is not None:
            if cfg.solver == "CPU":
                for node in path_to_goal:
                    global_map[step][node[0]][node[1]][cfg.NODE_IS_GOAL_VAR] = True
                path_completed = True
            else:
                path_completed = True
                for node in path_to_goal:
                    if not global_map[step][node[0]][node[1]][cfg.NODE_IS_GOAL_VAR]:
                        path_completed = False
                        break
            if path_completed:
                cfg.total_timesteps = step + 1
                print("Maze is solved by " + cfg.solver + "! Completion time = %d !!!!!!!!!" % cfg.total_timesteps)
                completion_time = cfg.total_timesteps
                break

    print("The simulation is finished.")
    return global_map, buffers, completion_time, solver


def calculate_orientation(boid, current_boid_obj, previous_boid_obj):
    """ Calculates orientation based on boid's velocity """
    if boid[vel + x_var] == 0 and boid[vel + y_var] == 0:
        alpha = previous_boid_obj.orientation
    else:
        if boid[vel + x_var] == 0:
            if boid[vel + y_var] > 0:
                alpha = np.pi / 2
            else:
                alpha = np.pi * 1.5
        else:
            if boid[vel + y_var] == 0:
                if boid[vel + x_var] > 0:
                    alpha = 0
                else:
                    alpha = np.pi
            else:
                alpha = np.arctan(boid[vel + y_var] / boid[vel + x_var])
        # noinspection PyChainedComparisons
        if boid[vel + x_var] > 0 and boid[vel + y_var] < 0:
            alpha += np.pi * 2
        else:
            if boid[vel + x_var] < 0 and boid[vel + y_var] != 0:
                alpha = np.pi + alpha

    current_boid_obj.orientation = alpha


def simulation_response(queue, kernels, gpu_params, buffers, amendments, flock, amaze):
    """ Calculates simulation response and sends it to GPU (i.e. agents)
    """
    # Transfer the amended position
    intermediary_events = cl_comp.gpu_amend_values(
        queue, kernels, gpu_params, buffers, amendments)

    # Get sensor readings
    sensor_readings = []
    for boid in flock.np_arrays:
        neighboring_tiles = maze.get_neighboring_tiles(boid[pos + x_var],
                                                       boid[pos + y_var],
                                                       amaze, None,
                                                       include_none=True)
        # Convert coordinates into Square objects
        northern_value = eastern_value = south_value = western_value = bottom_value = None
        coors = neighboring_tiles[maze.Location.top.value]
        if coors is not None:
            northern_value = amaze.matrix[coors[x_var]][coors[y_var]]
        coors = neighboring_tiles[maze.Location.right.value]
        if coors is not None:
            eastern_value = amaze.matrix[coors[x_var]][coors[y_var]]
        coors = neighboring_tiles[maze.Location.bottom.value]
        if coors is not None:
            south_value = amaze.matrix[coors[x_var]][coors[y_var]]
        coors = neighboring_tiles[maze.Location.left.value]
        if coors is not None:
            western_value = amaze.matrix[coors[x_var]][coors[y_var]]
        coors = neighboring_tiles[maze.Location.center.value]
        if coors is not None:
            bottom_value = amaze.matrix[coors[x_var]][coors[y_var]]

        sensor_readings.append(
            SensorReadings(northern_value, eastern_value, south_value, western_value, bottom_value))

    cl_comp.gpu_transfer_readings(queue, buffers, sensor_readings,
                                  intermediary_events)
    return intermediary_events
