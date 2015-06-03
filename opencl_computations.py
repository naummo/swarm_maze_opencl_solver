"""
opencl_computations.py encapsulates all interaction with OpenCL - initialization,
data transfers, kernel execution etc.
"""
import os

import pyopencl as cl
import numpy as np

import configs as cfg
import agents
import collision_detection as coll_detect
import maze


def init_opencl():
    """ Set up OpenCL """
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    device = my_gpu_devices[0]
    context = cl.Context(devices=my_gpu_devices)

    # Create queues
    queue = cl.CommandQueue(context)  # AWOOGA! to uncomment
    # properties=cl.command_queue_properties.PROFILING_ENABLE)

    return context, device, queue


def build_opencl_program(context, device):
    """ Builds the program """
    print("Building an OpenCL program.")
    # Fetch source code
    source = open(os.path.join(os.getcwd(), cfg.cl_dir, cfg.cl_main_url), 'r').read()

    # Program constants for building
    prog_params = {"preferred_work_group_size_multiple": 64,  # Dummy value
                   "max_work_group_size": 256,  # Dummy value

                   "number_of_boids": cfg.NumberOfBoids,

                   "maze_width": cfg.maze_width,
                   "maze_height": cfg.maze_height,

                   "neuron_threshold": cfg.NEURON_THRESHOLD,
                   "square_types_n": maze.Square.square_types_n *
                   maze.Square.square_orientation_n,
                   "square_types_n2": np.power(maze.Square.square_types_n *
                                               maze.Square.square_orientation_n, 2),
                   "square_types_n3": np.power(maze.Square.square_types_n *
                                               maze.Square.square_orientation_n, 3),
                   "weights_uninitialized": cfg.WEIGHTS_UNINITIALIZED,
                   "boid_attractant_dissolution_component":
                       cfg.boid_attractant_dissolution_component,
                   "node_attractant_dissolution_component":
                       cfg.node_attractant_dissolution_component,
                   "node_marker_dissolution_component":
                       cfg.node_marker_dissolution_component,
                   "readings_n": cfg.READINGS_N,
                   "reading_north": 0,
                   "reading_east": 1,
                   "reading_south": 2,
                   "reading_west": 3,
                   "reading_bottom": 4,

                   "minseparation": cfg.MinSeparation,
                   "boidvelocity": cfg.BoidVelocity,
                   "traction": cfg.Traction,
                   "center": cfg.center[0],

                   "dimensions": cfg.Dimensions,
                   "total_timesteps": cfg.total_timesteps,

                   "weightseparation": cfg.WeightSeparation,
                   "weightcenterofmass": cfg.WeightCenterOfMass,
                   "weightalignment": cfg.WeightAlignment,
                   "weightinertia": cfg.WeightInertia,

                   "boid_pos_var": cfg.BOID_POS_VAR,
                   "boid_vel_var": cfg.BOID_VEL_VAR,
                   "boid_pheromone_var": cfg.BOID_PHEROMONE_VAR,
                   "arrdim": cfg.ARRDIM,

                   "prog_size": cfg.PROG_SIZE,
                   "prog_edge_var": cfg.PROG_EDGE_VAR,
                   "prog_src_coors_var": cfg.PROG_SRC_COORS_VAR,
                   "prog_prev_path_var": cfg.PROG_PREV_PATH_VAR,
                   "prog_prev_prev_path_var": cfg.PROG_PREV_PREV_PATH_VAR,
                   "prog_goal_square_var": cfg.PROG_GOAL_SQUARE_VAR,
                   "prog_chosen_path_var": cfg.PROG_CHOSEN_PATH_VAR,
                   "prog_rotated_by_var": cfg.PROG_ROTATED_BY_VAR,
                   "prog_swamp_goal_var": cfg.PROG_SWAMP_GOAL_VAR,
                   "prog_stall_time_var": cfg.PROG_STALL_TIME_VAR,
                   "max_stall_time_allowed": cfg.MAX_STALL_TIME_ALLOWED,
                   "prog_leaving_deadend_var": cfg.PROG_LEAVING_DEADEND_VAR,
                   "prog_leaving_goal_var": cfg.PROG_LEAVING_GOAL_VAR,
                   "prog_rand_var": cfg.PROG_RAND_VAR,
                   "prog_rand_clock_var": cfg.PROG_RAND_CLOCK_VAR,
                   "rand_clock_max": cfg.RAND_CLOCK_MAX,

                   "exp_size": cfg.EXP_SIZE,
                   "exp_boid_hit_count_var": cfg.EXP_BOID_HIT_COUNT_VAR,
                   "exp_boid_hits_var": cfg.EXP_BOID_HITS_VAR,

                   "goal_edge_horizontal": cfg.GOAL_EDGE_HORIZONTAL,
                   "goal_edge_vertical": cfg.GOAL_EDGE_VERTICAL,

                   "node_size": cfg.NODE_SIZE,
                   "node_is_explored_var": cfg.NODE_IS_EXPLORED_VAR,
                   "node_pheromone_a_var": cfg.NODE_PHEROMONE_A_VAR,
                   "node_pheromone_m_var": cfg.NODE_PHEROMONE_M_VAR,
                   "node_is_deadend_var": cfg.NODE_IS_DEADEND_VAR,
                   "node_is_goal_var": cfg.NODE_IS_GOAL_VAR,

                   "entrance_id": maze.Square(
                       maze.Passage.entrance,
                       maze.Orientation.north).get_numeric_value(),
                   "exit_id": maze.Square(
                       maze.Passage.exit,
                       maze.Orientation.north).get_numeric_value()
                   }

    # Specify include directory
    build_options = "-I \"" + os.path.join(os.getcwd(), cfg.cl_dir) + "\""

    # Build program with some of the parameters missing to get their values
    prog = cl.Program(context, source % prog_params).build(options=build_options)

    gpu_params = {}
    # Get the correct values. Any kernel can be used here
    gpu_params["preferred_multiple"] = cl.Kernel(prog, 'k_init_memory').get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        device)
    gpu_params["max_work_group_size"] = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

    prog_params["preferred_work_group_size_multiple"] = gpu_params["preferred_multiple"]
    prog_params["max_work_group_size"] = gpu_params["max_work_group_size"]

    prog = cl.Program(context, source % prog_params).build(options=build_options)
    kernels = {}
    kernels["k_init_memory"] = prog.k_init_memory
    kernels["k_update_map"] = prog.k_update_map
    kernels["k_agent_reynolds_rules13_preprocess"] = prog.k_agent_reynolds_rules13_preprocess
    kernels["k_agent_reynolds_rule2_preprocess"] = prog.k_agent_reynolds_rule2_preprocess
    kernels["k_agent_ai_and_sim"] = prog.k_agent_ai_and_sim
    kernels["k_update_values"] = prog.k_update_values

    return kernels, gpu_params


def prepare_host_memory(context):
    """ Prepare buffers & arrays on host side """

    print("Creating buffers.")
    buffers = {}

    # Iteration number buffer
    buffers["global_iteration"] = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, size=np.dtype(np.uint16).itemsize)

    # Empty buffers for output data
    buffers["global_generated_flocks"] = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, size=agents.Boid.arraySize *
        cfg.total_timesteps * cfg.NumberOfBoids)

    # Buffers for transferring Python-computed amendments to OpenCL values
    buffers["global_amendments_n"] = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, size=np.dtype(np.uint16).itemsize)
    buffers["global_amendment_indices"] = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, size=np.dtype(np.uint16).itemsize *
        cfg.NumberOfBoids)
    buffers["global_amendment_values"] = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, size=np.dtype(np.float32).itemsize *
        cfg.NumberOfBoids * cfg.ARRDIM * cfg.Dimensions)
    # Amendment data holder class
    amendments = coll_detect.Amendments()

    # Buffer for CMMs
    buffers["global_cmms"] = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, size=np.dtype(np.uint8).itemsize *
        cfg.NumberOfBoids * int(np.power(maze.Square.square_types_n *
                                         maze.Square.square_orientation_n, 3)))

    # Buffer for random numbers
    buffers["global_random"] = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, size=np.dtype(np.float32).itemsize)

    # Buffer for agent programmes
    buffers["global_agent_programmes"] = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, size=np.dtype(np.uint16).itemsize *
        cfg.NumberOfBoids * cfg.PROG_SIZE)

    # Buffer for sensor readings
    buffers["global_sensor_readings"] = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, size=np.dtype(np.uint8).itemsize *
        cfg.NumberOfBoids * 5)

    # Buffer for the map
    buffers["global_map"] = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, size=np.dtype(np.float32).itemsize *
        cfg.maze_width * cfg.maze_height * cfg.NODE_SIZE)
    # And array
    if cfg.track_map_changes:
        global_map = np.zeros((cfg.total_timesteps, cfg.maze_width,
                               cfg.maze_height, cfg.NODE_SIZE), dtype=np.float32)
    else:
        global_map = None

    # Buffer for experiment results
    buffers["global_experiment"] = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, size=np.dtype(np.float32).itemsize *
        cfg.NumberOfBoids * cfg.EXP_SIZE)

    buffers["global_boid_avoidance_vectors"] = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, size=np.dtype(np.float32).itemsize *
        cfg.NumberOfBoids * cfg.Dimensions)
    buffers["global_flock_pos_sum"] = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, size=np.dtype(np.float32).itemsize * cfg.Dimensions)
    buffers["global_flock_vel_sum"] = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, size=np.dtype(np.float32).itemsize * cfg.Dimensions)

    # TEST
    buffers["global_test"] = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, size=np.dtype(np.float32).itemsize * 10 * 2)

    # Empty buffers for sharing data between items within groups
    buffers["local_flock_pos_sum"] = cl.LocalMemory(
        np.dtype(np.float32).itemsize * cfg.Dimensions)
    buffers["local_iteration"] = cl.LocalMemory(np.dtype(np.uint16).itemsize)
    buffers["local_flock_vel_sum"] = cl.LocalMemory(
        np.dtype(np.float32).itemsize * cfg.Dimensions)
    buffers["local_amendment_indices"] = cl.LocalMemory(
        np.dtype(np.uint16).itemsize * cfg.NumberOfBoids)
    buffers["local_amendment_values"] = cl.LocalMemory(
        np.dtype(np.float32).itemsize * cfg.NumberOfBoids * cfg.Dimensions)
    buffers["local_readings"] = cl.LocalMemory(
        np.dtype(np.uint8).itemsize * cfg.NumberOfBoids * cfg.Dimensions)

    return buffers, amendments, global_map


def set_kernel_arguments(kernels, buffers):
    """ Set arguments """
    # If same global buffer is passed to multiple kernels,
    # values are preserved between kernel switches.
    # If same local buffer is passed to multiple kernels,
    # values are NOT preserved.

    print("Setting kernel arguments.")

    kernels["k_init_memory"].set_args(
        buffers["global_map"],
        buffers["global_flock_pos_sum"],
        buffers["global_flock_vel_sum"],
        buffers["global_boid_avoidance_vectors"],
        buffers["global_agent_programmes"],
        buffers["global_cmms"]
    )
    kernels["k_update_map"].set_args(
        buffers["global_map"],
        buffers["global_test"]
    )
    kernels["k_agent_reynolds_rules13_preprocess"].set_args(
        buffers["global_generated_flocks"],
        buffers["global_iteration"],
        buffers["global_flock_pos_sum"],
        buffers["local_flock_pos_sum"],
        buffers["global_flock_vel_sum"],
        buffers["local_flock_vel_sum"],
        buffers["global_test"]
    )
    kernels["k_agent_reynolds_rule2_preprocess"].set_args(
        buffers["global_generated_flocks"],
        buffers["global_iteration"],
        buffers["global_boid_avoidance_vectors"]
    )
    kernels["k_agent_ai_and_sim"].set_args(
        buffers["global_generated_flocks"],
        buffers["global_map"],
        buffers["global_cmms"],
        buffers["global_sensor_readings"],
        buffers["global_flock_pos_sum"],
        buffers["local_flock_pos_sum"],
        buffers["global_flock_vel_sum"],
        buffers["local_flock_vel_sum"],
        buffers["global_iteration"],
        buffers["local_iteration"],
        buffers["global_boid_avoidance_vectors"],
        buffers["global_agent_programmes"],
        buffers["global_random"],
        buffers["global_experiment"],
        buffers["global_test"]
    )
    kernels["k_update_values"].set_args(
        buffers["global_generated_flocks"],
        buffers["global_amendments_n"],
        buffers["global_amendment_indices"],
        buffers["local_amendment_indices"],
        buffers["global_amendment_values"],
        buffers["local_amendment_values"],
        buffers["global_iteration"]
    )


def prepare_device_memory(queue, kernels, buffers, flock):
    """
        Initializes device memory and transfers
        first flocks from host to the device.
    """
    print("Initializing the memory and transferring the first flock.")
    intermediary_events = [cl.enqueue_nd_range_kernel(
        queue, kernels["k_init_memory"], [1], [1]),
        cl.enqueue_copy(queue, buffers["global_generated_flocks"], flock.np_arrays)]
    return intermediary_events


def update_map(queue, kernels, intermediary_events):
    """
        Updates map. Updating includes:
        - "Dissoluting" pheromones: pheromone level is reduced regularly to simulate ageing.
    """
    intermediary_events.append(cl.enqueue_nd_range_kernel(
        queue, kernels["k_update_map"], [1], [1]))


def gpu_generate_next_flock(step, queue, intermediary_events, kernels,
                            gpu_params, buffers, flocks, global_map):
    """
        Does one iteration of the computation. To be called in the increasing continuous
        order of the integer "step" argument
        """
    # Prepare memory for the generated flock
    new_flock = agents.Flock(cfg.NumberOfBoids)
    new_flock.init_empty_array()

    events = {}

    # Transfer the iteration number
    iteration = np.uint16(step)
    intermediary_events.append(cl.enqueue_copy(queue, buffers["global_iteration"], iteration))

    # Transfer the random number
    random = np.float32(np.random.rand(1) * 2 * np.pi)
    intermediary_events.append(cl.enqueue_copy(queue, buffers["global_random"], random))

    # -------------------------------------------------------------------------

    # Example workgroup/workitem pair: 7 groups of 64 items (400 boids)
    events["k_agent_reynolds_rules13_preprocess"] = cl.enqueue_nd_range_kernel(
        queue, kernels["k_agent_reynolds_rules13_preprocess"],
        (int(np.ceil(cfg.NumberOfBoids / gpu_params["preferred_multiple"]) *
             gpu_params["preferred_multiple"]),),
        (gpu_params["preferred_multiple"],), global_work_offset=None,
        wait_for=intermediary_events)

    # Example workgroup/workitem pair: 625 groups of 256 items (400x400 boids)
    events["k_agent_reynolds_rule2_preprocess"] = cl.enqueue_nd_range_kernel(
        queue, kernels["k_agent_reynolds_rule2_preprocess"],
        (int(np.ceil(np.square(cfg.NumberOfBoids) / gpu_params["max_work_group_size"]) *
             gpu_params["max_work_group_size"]),),
        (gpu_params["max_work_group_size"],), global_work_offset=None,
        wait_for=intermediary_events)

    # Example workgroup/workitem pair: 7 groups of 64 items (400 boids)
    events["k_agent_ai_and_sim"] = cl.enqueue_nd_range_kernel(
        queue, kernels["k_agent_ai_and_sim"],
        (int(np.ceil(cfg.NumberOfBoids / gpu_params["preferred_multiple"]) *
             gpu_params["preferred_multiple"]),),
        (gpu_params["preferred_multiple"],), global_work_offset=None,
        wait_for=[events["k_agent_reynolds_rules13_preprocess"],
                  events["k_agent_reynolds_rule2_preprocess"]])

    # transfer device -> host -------------------------------------------------
    # Second parameter size defines transfer size
    events["transfer_flocks"] = cl.enqueue_copy(
        queue, new_flock.np_arrays, buffers["global_generated_flocks"],
        device_offset=step * cfg.NumberOfBoids * agents.Boid.arraySize,
        wait_for=[events["k_agent_ai_and_sim"]])
    if cfg.track_map_changes:
        events["transfer_map"] = cl.enqueue_copy(
            queue, global_map[step], buffers["global_map"],
            wait_for=[events["k_agent_ai_and_sim"]])
    else:
        events["transfer_map"] = None

    cl.wait_for_events([events["transfer_flocks"], events["transfer_map"]])
    # print(new_flock.np_arrays)
    flocks.append(new_flock)

    # TEST
    if cfg.debug_on:
        global_test = np.zeros(10).astype(np.float32)
        events["transfer_test"] = cl.enqueue_copy(queue, global_test, buffers["global_test"])
        events["transfer_test"].wait()
        s = "%3d " % step
        for value in global_test:
            s += "%9.3f " % value
        print(s)
    # -------------------------------------------------------------------------


def gpu_amend_values(queue, kernels, gpu_params, buffers, amendments):
    """
        Transfers requested amendments (after collision detection check) to the GPU,
        where a kernel applies them to the data
    """
    intermediary_events = []
    packet = amendments.get_packet()
    if packet[amendments.amount_i] > 0:
        events = [
            cl.enqueue_copy(queue, buffers["global_amendments_n"],
                            packet[amendments.amount_i]),
            cl.enqueue_copy(queue, buffers["global_amendment_indices"],
                            packet[amendments.indices_i]),
            cl.enqueue_copy(queue, buffers["global_amendment_values"],
                            packet[amendments.values_i])]

        # X groups of 64 items (amendments.amount work items)
        intermediary_events.append(
            cl.enqueue_nd_range_kernel(
                queue, kernels["k_update_values"],
                (int(np.ceil(amendments.amount / gpu_params["preferred_multiple"]) *
                     gpu_params["preferred_multiple"]),),
                (gpu_params["preferred_multiple"],), global_work_offset=None,
                wait_for=events))
    return intermediary_events


def standardize_values(flock):
    """
    OpenCL may return values that are smaller than Python's minimum resolution.
    Python treats them as NoN. This function makes them equal to zero.
    """
    for boid in flock.np_arrays:
        for i in range(len(boid)):
            try:
                if np.isnan(boid[i]) or np.isinf(boid[i]):
                    boid[i] = 0
            except:
                boid[i] = 0


def gpu_transfer_readings(queue, buffers,
                          sensor_readings, intermediary_events):
    """
    Transfers the readings calculated by the simulation to the boids on GPU.
    """
    # Convert the list into a packet
    packet = np.zeros((cfg.NumberOfBoids, 5), dtype=np.uint8)
    for i in range(cfg.NumberOfBoids):
        # When a reading is None it means that it is a border of the maze
        if sensor_readings[i].northern_reading is not None:
            packet[i][0] = sensor_readings[i].northern_reading.get_numeric_value()
        else:
            packet[i][0] = maze.Wall.normal.value
        if sensor_readings[i].eastern_reading is not None:
            packet[i][1] = sensor_readings[i].eastern_reading.get_numeric_value()
        else:
            packet[i][1] = maze.Wall.normal.value
        if sensor_readings[i].southern_reading is not None:
            packet[i][2] = sensor_readings[i].southern_reading.get_numeric_value()
        else:
            packet[i][2] = maze.Wall.normal.value
        if sensor_readings[i].western_reading is not None:
            packet[i][3] = sensor_readings[i].western_reading.get_numeric_value()
        else:
            packet[i][3] = maze.Wall.normal.value
        """if sensor_readings[i].bottom_reading is not None:
            packet[i][4] = 1
        else:
            packet[i][4] = 0"""
        if sensor_readings[i].bottom_reading is not None:
            packet[i][4] = sensor_readings[i].bottom_reading.get_numeric_value()
        else:
            packet[i][4] = maze.Passage.normal.value

    intermediary_events.append(
        cl.enqueue_copy(queue, buffers["global_sensor_readings"], packet))


def get_experiment_data(queue, buffers):
    global_experiment = np.zeros((cfg.NumberOfBoids, cfg.EXP_SIZE), dtype=np.float32)
    cl.enqueue_copy(queue, global_experiment, buffers["global_experiment"])
    return global_experiment
