import os

import pyopencl as cl
import numpy as np

import configs as cfg
import agents_demo as agents
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
    print("Building a demo OpenCL program.")
    # Fetch source code
    source = open(os.path.join(os.getcwd(), cfg.cl_demo_dir, cfg.cl_main_url), 'r').read()

    # Program constants for building
    prog_params = {"preferred_work_group_size_multiple": 64,  # Dummy value
                   "max_work_group_size": 256,  # Dummy value

                   "number_of_boids": cfg.NumberOfBoids,
                   "number_of_predators": cfg.NumberOfPredators,

                   "maze_width": cfg.maze_width,
                   "maze_height": cfg.maze_height,

                   "predator_sight": cfg.PredatorSight,
                   "predatorradius": cfg.PredatorRadius,
                   "minseparation": cfg.MinSeparation,
                   "maxvelocityp": cfg.MaxVelocityP,
                   "maxvelocity": cfg.MaxVelocity,
                   "traction": cfg.Traction,
                   "center": cfg.center[0],

                   "dimensions": cfg.Dimensions,
                   "total_timesteps": cfg.total_timesteps,

                   "weightattackboid": cfg.WeightAttackBoid,
                   "weightcenterofmassp": cfg.WeightCenterOfMassP,
                   "weightknot": cfg.WeightKnot,
                   "weightseparation": cfg.WeightSeparation,
                   "weightcenterofmass": cfg.WeightCenterOfMass,
                   "weightalignment": cfg.WeightAlignment,
                   "weightcenter": cfg.WeightCenter,
                   "weightavoidpredator": cfg.WeightAvoidPredator,
                   "weightinertia": cfg.WeightInertia,

                   "pos": cfg.BOID_POS_VAR,
                   "vel": cfg.BOID_VEL_VAR,
                   "arrdim": cfg.ARRDIM
                   }

    # Specify include directory
    build_options = "-I \"" + os.path.join(os.getcwd(), cfg.cl_demo_dir) + "\""

    # Build program with some of the parameters missing to get their values
    prog = cl.Program(context, source % prog_params).build(options=build_options)

    gpu_params = {}
    # Get the correct values. Any kernel can be used here
    gpu_params["preferred_multiple"] = cl.Kernel(prog, 'k_predator_ai_preprocess').get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        device)
    gpu_params["max_work_group_size"] = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    """work_group_size = cl.Kernel(prog, 'k_predator_ai_preprocess').get_work_group_info(
        cl.kernel_work_group_info.COMPILE_WORK_GROUP_SIZE,
        device)"""

    prog_params["preferred_work_group_size_multiple"] = gpu_params["preferred_multiple"]
    prog_params["max_work_group_size"] = gpu_params["max_work_group_size"]

    prog = cl.Program(context, source % prog_params).build(options=build_options)
    kernels = {}
    kernels["k_predator_ai_preprocess"] = prog.k_predator_ai_preprocess
    kernels["k_predator_ai"] = prog.k_predator_ai
    kernels["k_agent_ai_preprocess"] = prog.k_agent_ai_preprocess
    kernels["k_agent_ai"] = prog.k_agent_ai
    kernels["k_update_values"] = prog.k_update_values

    return kernels, gpu_params


def prepare_host_memory(context):
    """ Prepare buffers & arrays """
    print("Creating buffers.")
    buffers = {}

    # Iteration number buffer
    buffers["global_iteration"] = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, size=np.dtype(np.uint16).itemsize)

    # Empty buffers for output data
    buffers["generated_flocks"] = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, size=np.dtype(np.float32).itemsize *
        cfg.total_timesteps * cfg.NumberOfBoids * cfg.ARRDIM * cfg.Dimensions)
    buffers["generated_predators"] = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, size=np.dtype(np.float32).itemsize *
        cfg.total_timesteps * cfg.NumberOfPredators * cfg.ARRDIM * cfg.Dimensions)

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

    # Buffer for sensor readings
    buffers["global_readings"] = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, size=np.dtype(np.uint8).itemsize *
        cfg.NumberOfBoids * 4)

    # Empty buffers for sharing data between groups
    buffers["global_predator_attack_vector"] = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, size=np.dtype(np.float32).itemsize *
        cfg.NumberOfPredators * cfg.Dimensions)
    buffers["global_boid_avoidance_vector"] = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, size=np.dtype(np.float32).itemsize *
        cfg.NumberOfBoids * cfg.Dimensions)
    buffers["global_flock_pos_sum"] = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, size=np.dtype(np.float32).itemsize * cfg.Dimensions)
    buffers["global_flock_vel_sum"] = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, size=np.dtype(np.float32).itemsize * cfg.Dimensions)

    # TEST
    buffers["global_test"] = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, size=np.dtype(np.float32).itemsize * cfg.Dimensions)

    # Empty buffers for sharing data between items within groups
    buffers["local_flock_pos_sum"] = cl.LocalMemory(
        np.dtype(np.float32).itemsize * cfg.Dimensions)
    buffers["local_predator_attack_vector"] = cl.LocalMemory(
        np.dtype(np.float32).itemsize * cfg.NumberOfPredators * cfg.Dimensions)
    buffers["local_iteration"] = cl.LocalMemory(np.dtype(np.uint16).itemsize)
    buffers["local_flock_vel_sum"] = cl.LocalMemory(
        np.dtype(np.float32).itemsize * cfg.Dimensions)
    buffers["local_amendment_indices"] = cl.LocalMemory(
        np.dtype(np.uint16).itemsize * cfg.NumberOfBoids)
    buffers["local_amendment_values"] = cl.LocalMemory(
        np.dtype(np.float32).itemsize * cfg.NumberOfBoids * cfg.Dimensions)
    buffers["local_readings"] = cl.LocalMemory(
        np.dtype(np.uint8).itemsize * cfg.NumberOfBoids * cfg.Dimensions)

    return buffers, amendments


def set_kernel_arguments(kernels, buffers):
    """ Set arguments """
    # If same global buffer is passed to multiple kernels,
    # values are preserved between kernel switches.
    # If same local buffer is passed to multiple kernels,
    # values are NOT preserved.
    print("Setting kernel arguments.")

    kernels["k_predator_ai_preprocess"].set_args(
        buffers["generated_flocks"],
        buffers["generated_predators"],
        buffers["global_flock_pos_sum"],
        buffers["local_flock_pos_sum"],
        buffers["global_predator_attack_vector"],
        buffers["local_predator_attack_vector"],
        buffers["global_iteration"],
        buffers["local_iteration"],
        buffers["global_flock_vel_sum"],
        buffers["global_boid_avoidance_vector"],
        buffers["global_test"]
    )
    kernels["k_predator_ai"].set_args(
        buffers["generated_predators"],
        buffers["global_flock_pos_sum"],
        buffers["local_flock_pos_sum"],
        buffers["global_predator_attack_vector"],
        buffers["global_iteration"],
        buffers["local_iteration"],
        buffers["global_test"]
    )
    kernels["k_agent_ai_preprocess"].set_args(
        buffers["generated_flocks"],
        buffers["global_boid_avoidance_vector"],
        buffers["global_flock_vel_sum"],
        buffers["local_flock_vel_sum"],
        buffers["global_iteration"],
        buffers["local_iteration"],
        buffers["global_test"]
    )
    kernels["k_agent_ai"].set_args(
        buffers["generated_flocks"],
        buffers["generated_predators"],
        buffers["global_flock_pos_sum"],
        buffers["local_flock_pos_sum"],
        buffers["global_boid_avoidance_vector"],
        buffers["global_flock_vel_sum"],
        buffers["local_flock_vel_sum"],
        buffers["global_iteration"],
        buffers["local_iteration"],
        buffers["global_test"]
    )
    kernels["k_update_values"].set_args(
        buffers["generated_flocks"],
        buffers["global_amendments_n"],
        buffers["global_amendment_indices"],
        buffers["local_amendment_indices"],
        buffers["global_amendment_values"],
        buffers["local_amendment_values"],
        buffers["global_iteration"],
        buffers["local_iteration"]
    )


# noinspection PyUnusedLocal
def prepare_device_memory(queue, kernels, buffers, flock):
    """
        Transfers first flocks from host to the device.
        This is not done in prepare_memory as then the whole array
        of flocks would be transferred, and it is almost full of zeros
        at the beginning.
    """
    print("Transferring the first flock.")
    intermediary_events = [
        cl.enqueue_copy(queue, buffers["generated_flocks"],
                        flock["flock"].np_arrays),
        cl.enqueue_copy(queue, buffers["generated_predators"],
                        flock["predators"].np_arrays)]

    return intermediary_events


# noinspection PyUnusedLocal
def update_map(queue, kernels, intermediary_events):
    """
        Place holder for a function that exists only in normal mode.
    """


def gpu_generate_next_flock(step, queue, intermediary_events, kernels, gpu_params, buffers, flocks):
    """
        Does one iteration of the computation. To be called in the increasing continuous
        order of the integer "step" argument
        """
    # Prepare memory for the generated flock
    new_flock = {}
    new_flock["flock"] = agents.Flock(cfg.NumberOfBoids)
    new_flock["flock"].np_arrays = np.zeros(
        (cfg.NumberOfBoids, cfg.ARRDIM, cfg.Dimensions), dtype=np.float32)
    new_flock["predators"] = agents.Flock(cfg.NumberOfPredators)
    new_flock["predators"].np_arrays = np.zeros(
        (cfg.NumberOfPredators, cfg.ARRDIM, cfg.Dimensions), dtype=np.float32)

    events = {}

    # Transfer the iteration number
    iteration = np.uint16(step)
    intermediary_events.append(cl.enqueue_copy(queue, buffers["global_iteration"], iteration))

    # 7 groups of 64 items (400 boids)
    events["k_predator_ai_preprocess"] = cl.enqueue_nd_range_kernel(
        queue, kernels["k_predator_ai_preprocess"],
        (int(np.ceil(cfg.NumberOfBoids / gpu_params["preferred_multiple"]) * gpu_params["preferred_multiple"]),),
        (gpu_params["preferred_multiple"],), global_work_offset=None,
        wait_for=intermediary_events)

    # 1 group of 5 items (5 predators)
    events["k_predator_ai"] = cl.enqueue_nd_range_kernel(
        queue, kernels["k_predator_ai"],
        (cfg.NumberOfPredators,), (cfg.NumberOfPredators,), global_work_offset=None,
        wait_for=(events["k_predator_ai_preprocess"],))

    # transfer device -> host -------------------------------------------------
    # Second parameter size defines transfer size
    events["transfer_predators"] = cl.enqueue_copy(
        queue, new_flock["predators"].np_arrays, buffers["generated_predators"],
        device_offset=step * cfg.NumberOfPredators * cfg.ARRDIM *
        cfg.Dimensions * np.dtype(np.float32).itemsize,
        wait_for=(events["k_predator_ai"],))
    # -------------------------------------------------------------------------

    # 625 groups of 256 items (400x400 boids)
    events["k_agent_ai_preprocess"] = cl.enqueue_nd_range_kernel(
        queue, kernels["k_agent_ai_preprocess"],
        (int(np.ceil(np.square(cfg.NumberOfBoids) / gpu_params["max_work_group_size"]) * gpu_params[
            "max_work_group_size"]),),
        (gpu_params["max_work_group_size"],), global_work_offset=None,
        wait_for=(events["k_predator_ai"],))
    # 7 groups of 64 items (400 boids)
    events["k_agent_ai"] = cl.enqueue_nd_range_kernel(
        queue, kernels["k_agent_ai"],
        (int(np.ceil(cfg.NumberOfBoids / gpu_params["preferred_multiple"]) * gpu_params["preferred_multiple"]),),
        (gpu_params["preferred_multiple"],), global_work_offset=None,
        wait_for=(events["k_agent_ai_preprocess"],))

    # transfer device -> host -------------------------------------------------
    # Second parameter size defines transfer size
    events["transfer_flocks"] = cl.enqueue_copy(
        queue, new_flock["flock"].np_arrays, buffers["generated_flocks"],
        device_offset=step * cfg.NumberOfBoids * cfg.ARRDIM *
        cfg.Dimensions * np.dtype(np.float32).itemsize,
        wait_for=(events["k_agent_ai"],))

    cl.wait_for_events([events["transfer_predators"], events["transfer_flocks"]])
    flocks.append(new_flock)

    # TEST
    # global_test = np.zeros(cfg.Dimensions).astype(np.float32)
    # events["transfer_test"] = cl.enqueue_copy(queue, global_test, buffers["global_test"])
    # events["transfer_test"].wait()
    # print("global_test=", global_test)
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
    for creature_type in flock:
        for boid in flock[creature_type].np_arrays:
            for metric in boid:
                for i in range(len(metric)):
                    try:
                        if np.isnan(metric[i]):
                            metric[i] = 0
                    except:
                        metric[i] = 0


def gpu_transfer_readings(queue, buffers,
                          sensor_readings, intermediary_events):
    """
    Transfers the readings calculated by the simulation to the boids on GPU.
    """
    # Convert the list into a packet
    packet = np.zeros((cfg.NumberOfBoids, 4), dtype=np.uint8)
    for i in range(cfg.NumberOfBoids):
        if sensor_readings[i].top_reading is not None:
            packet[i][0] = sensor_readings[i].top_reading.get_numeric_value()
        else:
            packet[i][0] = maze.Wall.normal.value
        if sensor_readings[i].right_reading is not None:
            packet[i][1] = sensor_readings[i].right_reading.get_numeric_value()
        else:
            packet[i][1] = maze.Wall.normal.value
        if sensor_readings[i].bottom_reading is not None:
            packet[i][2] = sensor_readings[i].bottom_reading.get_numeric_value()
        else:
            packet[i][2] = maze.Wall.normal.value
        if sensor_readings[i].left_reading is not None:
            packet[i][3] = sensor_readings[i].left_reading.get_numeric_value()
        else:
            packet[i][3] = maze.Wall.normal.value

    intermediary_events.append(
        cl.enqueue_copy(queue, buffers["global_readings"], packet))
