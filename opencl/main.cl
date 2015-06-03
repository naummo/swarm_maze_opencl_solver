/* Macros */
// OpenCL
#define PREFERRED_WORK_GROUP_SIZE_MULTIPLE      %(preferred_work_group_size_multiple)d
#define MAX_WORK_GROUP_SIZE                     %(max_work_group_size)d

// Maze
#define MAZE_WIDTH                              %(maze_width)d
#define MAZE_HEIGHT                             %(maze_height)d

// Agent intelligence
#define NEURON_THRESHOLD                        %(neuron_threshold)d
#define SQUARE_TYPES_N                          %(square_types_n)d
#define SQUARE_TYPES_N2                         %(square_types_n2)d
#define SQUARE_TYPES_N3                         %(square_types_n3)d
#define WEIGHTS_UNINITIALIZED                   %(weights_uninitialized)d
#define BOID_ATTRACTANT_DISSOLUTION_COMPONENT   %(boid_attractant_dissolution_component)f
#define NODE_ATTRACTANT_DISSOLUTION_COMPONENT   %(node_attractant_dissolution_component)f
#define NODE_MARKER_DISSOLUTION_COMPONENT       %(node_marker_dissolution_component)f
#define NEURONS_N                               8
#define READINGS_N                              %(readings_n)d
#define TRIGRAMS_N                              (%(readings_n)d - 1)
#define READING_NORTH                           %(reading_north)d
#define READING_EAST                            %(reading_east)d
#define READING_SOUTH                           %(reading_south)d
#define READING_WEST                            %(reading_west)d
#define READING_BOTTOM                          %(reading_bottom)d

#define ENTRANCE_ID                             %(entrance_id)d
#define EXIT_ID                                 %(exit_id)d

// Boids
#define NUMBER_OF_BOIDS                         %(number_of_boids)d
#define MINSEPARATION                           %(minseparation)f
#define BOIDVELOCITY                            %(boidvelocity)f
#define TRACTION                                %(traction)f
#define CENTER                                  %(center)f

// Reynold's weights
#define WEIGHTSEPARATION                        %(weightseparation)f
#define WEIGHTCENTEROFMASS                      %(weightcenterofmass)f
#define WEIGHTALIGNMENT                         %(weightalignment)f

// Simulation
#define WEIGHTINERTIA                           %(weightinertia)f
#define TOTAL_TIMESTEPS                         %(total_timesteps)d
#define DIMENSIONS                              %(dimensions)d

/* Memory structure */
#define BOID_POS_VAR                            %(boid_pos_var)d
#define BOID_VEL_VAR                            %(boid_vel_var)d
#define BOID_PHEROMONE_VAR                      %(boid_pheromone_var)d
#define ARRDIM                                  %(arrdim)d

#define FLOATN                                  float%(dimensions)d
#define UCHARN                                  uchar%(dimensions)d
#define VLOADN(p)                               vload%(dimensions)d(0, p)
#define VSTOREN(data, p)                        vstore%(dimensions)d(data, 0, p)

#define BOID_SIZE                               (ARRDIM * DIMENSIONS + 1)
#define FLOCK_SIZE(boids)                       ((boids) * BOID_SIZE)

#define INDEX_IN_BOID(var)                      ((var) * DIMENSIONS)
#define INDEX_IN_FLOCK(boid, var)               (boid * BOID_SIZE + INDEX_IN_BOID(var))
#define INDEX_IN_ALL_FLOCKS(step, boid, var)    ((step) * NUMBER_OF_BOIDS * BOID_SIZE + INDEX_IN_FLOCK(boid, var))

// TODO: pack pairs of programme small uchar values in single ushort values
#define PROG_SIZE                               %(prog_size)d
#define PROG_EDGE_VAR                           %(prog_edge_var)d
#define PROG_SRC_COORS_VAR                      %(prog_src_coors_var)d
#define PROG_PREV_PATH_VAR                      %(prog_prev_path_var)d
#define PROG_PREV_PREV_PATH_VAR                 %(prog_prev_prev_path_var)d
#define PROG_GOAL_SQUARE_VAR                    %(prog_goal_square_var)d
#define PROG_CHOSEN_PATH_VAR                    %(prog_chosen_path_var)d
#define PROG_ROTATED_BY_VAR                     %(prog_rotated_by_var)d
#define PROG_SWAMP_GOAL_VAR                     %(prog_swamp_goal_var)d
#define PROG_STALL_TIME_VAR                     %(prog_stall_time_var)d
#define PROG_RAND_VAR                           %(prog_rand_var)d
#define PROG_RAND_CLOCK_VAR                     %(prog_rand_clock_var)d
#define RAND_CLOCK_MAX                          %(rand_clock_max)d
#define MAX_STALL_TIME_ALLOWED                  %(max_stall_time_allowed)d
#define PROG_LEAVING_DEADEND_VAR                %(prog_leaving_deadend_var)d
#define PROG_LEAVING_GOAL_VAR                   %(prog_leaving_goal_var)d

#define EXP_SIZE                                %(exp_size)d
#define INDEX_IN_EXP(boid)                      (boid * EXP_SIZE)
#define EXP_BOID_HIT_COUNT_VAR                  %(exp_boid_hit_count_var)d
#define EXP_BOID_HITS_VAR                       %(exp_boid_hits_var)d

#define ERROR_EXPONENT                          0.001
#define ERROR_MARGIN                            0.02
#define INDEX_IN_PROGS(boid)                    (boid * PROG_SIZE)
#define GOAL_NOT_SET                            0
#define GOAL_EDGE_HORIZONTAL                    %(goal_edge_horizontal)d
#define GOAL_EDGE_VERTICAL                      %(goal_edge_vertical)d

/* agents.cl-related macros */
#define NODE_SIZE                               %(node_size)d
#define NODE_PHEROMONE_A_VAR                    %(node_pheromone_a_var)d
#define NODE_PHEROMONE_M_VAR                    %(node_pheromone_m_var)d
// TODO: combine NODE_IS_EXPLORED_VAR and NODE_IS_DEADEND_VAR booleans
// into a one uchar variable node_status
#define NODE_IS_EXPLORED_VAR                    %(node_is_explored_var)d
#define NODE_IS_DEADEND_VAR                     %(node_is_deadend_var)d
#define NODE_IS_GOAL_VAR                        %(node_is_goal_var)d
#define INDEX_IN_MAP(x, y, var)                 (x * MAZE_HEIGHT * NODE_SIZE + y * NODE_SIZE + var)
#define DELTA_COOR_X(c, i)                      (c.x + delta_coors[i * 2])
#define DELTA_COOR_Y(c, i)                      (c.y + delta_coors[i * 2 + 1])
#define NORTH                                   0
#define EAST                                    1
#define SOUTH                                   2
#define WEST                                    3
#define TRUE                                    1
#define MAYBE                                   2

// This is the only place where source code is linked
// Other .cl files will be including only header files
#include "utilities.cl"
#include "agents.cl"
#include "simulation.cl"

// - Local memory is on a per kernel instance basis
// Thus I can accelerate by merging kernels (__local reduction)
// - Memory access coalescing
// - Constant memory on GPU is cached and is only slightly slower than local memory
// - I assume that all buffers are copied to global memory. If I 
// pass the same buffer to two parameters, global and const, then
// global will just get a pointer, and const will first cache it
// and then store pointer
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void
k_init_memory(__global float* global_map,
              __global float* global_flock_pos_sum,
              __global float* global_flock_vel_sum,
              __global float* global_boid_avoidance_vectors,
              __global ushort* global_agent_programmes,
              __global uchar* global_cmms) {
    int i;
    // Clean global memory
    for (i = 0; i < DIMENSIONS; i++) {
        global_flock_pos_sum[i] = 0.0f;
        global_flock_vel_sum[i] = 0.0f;
    }
    for (i = 0; i < NUMBER_OF_BOIDS * ARRDIM * DIMENSIONS; i++)
        global_boid_avoidance_vectors[i] = 0.0f;
    for (i = 0; i < MAZE_WIDTH * MAZE_HEIGHT * NODE_SIZE; i++)
        global_map[i] = 0;
    // Initialize CMM
    for (i = 0; i < NUMBER_OF_BOIDS * SQUARE_TYPES_N3; i++)
        global_cmms[i] = 0xF;
    // Initialize agent programmes
    for (i = 0; i < NUMBER_OF_BOIDS * PROG_SIZE; i++)
        global_agent_programmes[i] = 0;
}



__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void
k_update_map(__global float* global_map,
             __global float* global_test) {
    dissolute_pheromones(global_map);
}



__kernel __attribute__((reqd_work_group_size(PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 1, 1)))
void
k_agent_reynolds_rules13_preprocess(__global float* global_generated_flocks/*[TOTAL_TIMESTEPS][NUMBER_OF_BOIDS][ARRDIM]*/,
                                    __global ushort* global_iteration,
                                    __global float* global_flock_pos_sum,
                                    __local float* local_flock_pos_sum,
                                    __global float* global_flock_vel_sum,
                                    __local float* local_flock_vel_sum,
                                    __global float* global_test) {

    /*--------- Prepare memory and filter out surplus workitems ---------------*/

    float previous_position[DIMENSIONS];
    float previous_velocity[DIMENSIONS];
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int i;

    // We have more workitems than boids
    if (global_id >= NUMBER_OF_BOIDS)
        return;

    /* PER WORKGROUP */
    /* Clean local memory */
    if (local_id == 0)
        for (i = 0; i < DIMENSIONS; i++) {
            local_flock_pos_sum[i] = 0.0f;
            local_flock_vel_sum[i] = 0.0f;
        }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Fetch agent's previous position
    fetch_global(&global_generated_flocks[INDEX_IN_ALL_FLOCKS((int)(*global_iteration - 1), global_id, BOID_POS_VAR)],
                 previous_position, DIMENSIONS);
    // Fetch agent's previous velocity
    fetch_global(&global_generated_flocks[INDEX_IN_ALL_FLOCKS((int)(*global_iteration - 1), global_id, BOID_VEL_VAR)],
                 previous_velocity, DIMENSIONS);

    /*---------           Compute outputs           ---------------*/

    agent_reynolds_rules13_preprocess(local_flock_pos_sum,
                                      local_flock_vel_sum,
                                      previous_position,
                                      previous_velocity);

    /*---------            Save outputs            ---------------*/

    /* PER WORKGROUP */
    // Exit fetch (local -> global) 
    barrier(CLK_LOCAL_MEM_FENCE);
    // Doesn't matter which work item will do it, but it must be unique within workgroup
    if (local_id == get_local_size(0) - 1) {
        float vector[DIMENSIONS];
        // Add partial sum up to the full sum in global memory
        fetch_local(local_flock_pos_sum, vector, DIMENSIONS);
        atomic_add_global_arr(global_flock_pos_sum, vector);
        // Add partial sum up to the full sum in global memory
        fetch_local(local_flock_vel_sum, vector, DIMENSIONS);
        atomic_add_global_arr(global_flock_vel_sum, vector);
    }
}



__kernel __attribute__((reqd_work_group_size(MAX_WORK_GROUP_SIZE, 1, 1)))
void
k_agent_reynolds_rule2_preprocess(__global float* global_generated_flocks, /*[TOTAL_TIMESTEPS][NUMBER_OF_BOIDS][ARRDIM]*/
                                  __global ushort* global_iteration,
                                  __global float* global_boid_avoidance_vectors) {

    /*--------- Prepare memory and filter out surplus workitems ---------------*/

    int global_id           = get_global_id(0);
    int local_id            = get_local_id(0);
    int main_boid_id        = trunc((float)global_id / (float)NUMBER_OF_BOIDS);;
    int compared_boid_id    = fmod((float)global_id, NUMBER_OF_BOIDS);

    // We have more workitems than boids
    if (main_boid_id >= NUMBER_OF_BOIDS)
        return;

    /*---------           Compute outputs           ---------------*/

    agent_reynolds_rule2_preprocess(&global_generated_flocks[
                                      INDEX_IN_ALL_FLOCKS((int)(*global_iteration - 1), 0, 0)],
                                    global_boid_avoidance_vectors,
                                    main_boid_id,
                                    compared_boid_id);
}



__kernel __attribute__((reqd_work_group_size(PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 1, 1)))
void
k_agent_ai_and_sim(__global float* global_generated_flocks, /*[TOTAL_TIMESTEPS][NUMBER_OF_BOIDS][ARRDIM]*/
                  __global float* global_map,
                  __global uchar* global_cmms,
                  __global uchar* global_sensor_readings,
                  __global float* global_flock_pos_sum,
                  __local float* local_flock_pos_sum,
                  __global float* global_flock_vel_sum,
                  __local float* local_flock_vel_sum,
                  __global ushort* global_iteration,
                  __local ushort* local_iteration,
                  __global float* global_boid_avoidance_vectors,
                  __global ushort* global_agent_programmes,
                  __global float* global_random,
                  __global float* global_experiment,
                  __global float* global_test) {

    /*--------- Prepare memory and filter out surplus workitems ---------------*/

    __global float* global_previous_flock;
    __global float* global_new_flock;
    __global float* global_test_subarr;
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    FLOATN impulse;
    FLOATN velocity;
    uchar sensor_readings[READINGS_N];
    int i;

    // We have more workitems than boids
    if (global_id >= NUMBER_OF_BOIDS)
        return;

    /* PER WORKGROUP */
    /* Entry fetch (global -> local) */
    if (local_id == 0) {
        for (i = 0; i < DIMENSIONS; i++) {
            local_flock_pos_sum[i] = global_flock_pos_sum[i];
            local_flock_vel_sum[i] = global_flock_vel_sum[i];
        }

        *local_iteration = *global_iteration;
    }
    if (global_id == 16)
        global_test_subarr = global_test;
    else
        global_test_subarr = global_test + 10;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Entry fetch (global -> private) */
    for (i = 0; i < READINGS_N; i++)
        sensor_readings[i] = global_sensor_readings[READINGS_N * global_id + i];

    /* Select previous data out of the whole dataset */
    global_previous_flock = &global_generated_flocks[
        INDEX_IN_ALL_FLOCKS(*local_iteration - 1, 0, 0)];

    /* Select current data out of the whole dataset */
    global_new_flock = &global_generated_flocks[
        INDEX_IN_ALL_FLOCKS(*local_iteration, 0, 0)];

    /*---------           Compute outputs           ---------------*/

    agent_ai(global_map,
             &global_cmms[global_id * SQUARE_TYPES_N3],
             global_boid_avoidance_vectors,
             local_flock_pos_sum,
             local_flock_vel_sum,
             global_previous_flock,
             global_new_flock,
             &global_agent_programmes[INDEX_IN_PROGS(global_id)],
             sensor_readings,
             global_id,
             &impulse,
             global_random,
             local_iteration,
             &global_experiment[INDEX_IN_EXP(global_id)],
             global_test_subarr);

    simulate_locomotion(global_previous_flock,                  
                        local_iteration,
                        impulse,
                        global_id,
                        &velocity);

    /*---------            Save outputs            ---------------*/

    // Move boids
    VSTOREN(
        VLOADN(&global_previous_flock[INDEX_IN_FLOCK(global_id, BOID_POS_VAR)]) + velocity,
            &global_new_flock[INDEX_IN_FLOCK(global_id, BOID_POS_VAR)]);

    // Update velocity
    VSTOREN(velocity, 
        &global_new_flock[INDEX_IN_FLOCK(global_id, BOID_VEL_VAR)]);
}



__kernel __attribute__((reqd_work_group_size(PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 1, 1)))
void
k_update_values(__global float* global_generated_flocks/*[TOTAL_TIMESTEPS][NUMBER_OF_BOIDS][ARRDIM]*/,
                __global ushort* global_amendments_n,
                __global ushort* global_amendment_indices/*[amendments_n]*/,
                __local ushort* local_amendment_indices/*[amendments_n]*/,
                __global float* global_amendment_values /*[amendments_n][DIMENSIONS]*/,
                __local float* local_amendment_values /*[amendments_n][DIMENSIONS]*/,
                __global ushort* global_iteration) {

    /*---------Prepare memory and filter out surplus workitems---------------*/

    int global_id;
    int local_id;
    int i, j;

    global_id = get_global_id(0);
    local_id = get_local_id(0);

    // We probably have more workitems than amendments
    if (global_id >= *global_amendments_n)
        return;

    /* Entry fetch (global -> local) */
    if (local_id == 0) {
        int limit;
        if (global_id + get_local_size(0) < *global_amendments_n)
            limit = global_id + get_local_size(0);
        else
            limit = *global_amendments_n;
        for (i = global_id; i < limit; i++) {
            local_amendment_indices[i] = global_amendment_indices[i];
            for (j = 0; j < DIMENSIONS; j++)
                local_amendment_values[INDEX_IN_BOID(i) + j] = global_amendment_values[INDEX_IN_BOID(i) + j];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /*---------            Save outputs            ---------------*/
    
    for (i = 0; i < DIMENSIONS; i++)
        global_generated_flocks[
        INDEX_IN_ALL_FLOCKS(*global_iteration, local_amendment_indices[global_id], BOID_POS_VAR) + i] = 
            local_amendment_values[INDEX_IN_BOID(global_id) + i];

    /* Exit clean up (global) */
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (global_id == 0)
        *global_amendments_n = 0;
}
