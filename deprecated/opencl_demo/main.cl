// Constant values
#define PREFERRED_WORK_GROUP_SIZE_MULTIPLE      %(preferred_work_group_size_multiple)d
#define MAX_WORK_GROUP_SIZE                     %(max_work_group_size)d

#define NUMBER_OF_BOIDS                         %(number_of_boids)d
#define NUMBER_OF_PREDATORS                     %(number_of_predators)d

#define MAZE_WIDTH                              %(maze_width)d
#define MAZE_HEIGHT                             %(maze_height)d

#define PREDATOR_SIGHT                          %(predator_sight)f
#define PREDATORRADIUS                          %(predatorradius)f
#define MINSEPARATION                           %(minseparation)f
#define MAXVELOCITYP                            %(maxvelocityp)f
#define MAXVELOCITY                             %(maxvelocity)f
#define TRACTION                                %(traction)f
#define CENTER                                  %(center)f

#define WEIGHTSEPARATION                        %(weightseparation)f
#define WEIGHTATTACKBOID                        %(weightattackboid)f
#define WEIGHTCENTEROFMASSP                     %(weightcenterofmassp)f
#define WEIGHTKNOT                              %(weightknot)f
#define WEIGHTCENTEROFMASS                      %(weightcenterofmass)f
#define WEIGHTALIGNMENT                         %(weightalignment)f
#define WEIGHTCENTER                            %(weightcenter)f
#define WEIGHTAVOIDPREDATOR                     %(weightavoidpredator)f
#define WEIGHTINERTIA                           %(weightinertia)f

#define TOTAL_TIMESTEPS                         %(total_timesteps)d
#define DIMENSIONS                              %(dimensions)d
#define POS                                     %(pos)d
#define VEL                                     %(vel)d
#define ARRDIM                                  %(arrdim)d

#define FLOATN                                  float%(dimensions)d
#define VLOADN(p)                               vload%(dimensions)d(0, p)
#define VSTOREN(data, p)                        vstore%(dimensions)d(data, 0, p)

#define BOID_SIZE                               ARRDIM * DIMENSIONS
#define FLOCK_SIZE(boids)                       boids * BOID_SIZE

#define VAR_INDEX(var)                          var * DIMENSIONS
#define BOID_INDEX(boid,var)                    boid * BOID_SIZE + VAR_INDEX(var)
#define FLOCK_INDEX(boidsN,step,boid,var)       step * FLOCK_SIZE(boidsN) + BOID_INDEX(boid,var)

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

__kernel __attribute__((reqd_work_group_size(PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 1, 1)))
void
k_predator_ai_preprocess( __global float* global_generated_flocks/*[TOTAL_TIMESTEPS][NUMBER_OF_BOIDS][ARRDIM]*/,
                          __global float* global_generated_predators/*[TOTAL_TIMESTEPS][NUMBER_OF_PREDATORS][ARRDIM]*/,
                          __global float* global_flock_pos_sum,
                          __local float* local_flock_pos_sum,
                          __global float* global_predator_attack_vector/*[NUMBER_OF_PREDATORS]*/,
                          __local float* local_predator_attack_vector/*[NUMBER_OF_PREDATORS]*/,
                          __global ushort* global_iteration,
                          __local ushort* local_iteration,
                          __global float* global_flock_vel_sum,
                          __global float* global_boid_avoidance_vector/*[NUMBER_OF_BOIDS]*/,
                          __global float* global_test) {

    /*---------Prepare memory and filter out surplus workitems---------------*/

    __global float* previous_flock;
    __global float* previous_predators;
    int i;
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);

    // We have more workitems than boids
    if (global_id >= NUMBER_OF_BOIDS)
        return;

    // As the first kernel to execute, this function is
    // responsible for cleaning global memory
    if (global_id == 0)
        if (*global_iteration != 0) {
            for (i = 0; i < NUMBER_OF_PREDATORS * ARRDIM * DIMENSIONS; i++)
                global_predator_attack_vector[i] = 0.0f;
            for (i = 0; i < DIMENSIONS; i++) {
                global_flock_pos_sum[i] = 0.0f;
                global_flock_vel_sum[i] = 0.0f;
            }
            for (i = 0; i < NUMBER_OF_BOIDS * ARRDIM * DIMENSIONS; i++)
                global_boid_avoidance_vector[i] = 0.0f;
        }

    /* Entry fetch (global -> local) */
    if (local_id == 0) {
        *local_iteration = *global_iteration;
        for (i = 0; i < DIMENSIONS; i++)
            local_flock_pos_sum[i] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Select previous data out of the whole dataset */
    previous_flock     = &global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS, (int)(*local_iteration - 1), 0, 0)];
    previous_predators = &global_generated_predators[FLOCK_INDEX(NUMBER_OF_PREDATORS, (int)(*local_iteration - 1), 0, 0)];

    /*---------           Compute outputs           ---------------*/

    agents_predator_ai_preprocess(global_predator_attack_vector,
                                  local_predator_attack_vector,
                                  global_flock_pos_sum,
                                  local_flock_pos_sum,
                                  previous_flock,
                                  previous_predators,
                                  global_id,
                                  local_id);
}



__kernel __attribute__((reqd_work_group_size(NUMBER_OF_PREDATORS, 1, 1)))
void
k_predator_ai(__global float* global_generated_predators/*[TOTAL_TIMESTEPS][NUMBER_OF_PREDATORS][ARRDIM]*/,
            __global float* global_flock_pos_sum,
            __local float* local_flock_pos_sum,
            __global float* global_predator_attack_vector/*[NUMBER_OF_PREDATORS]*/,
            __global ushort* global_iteration,
            __local ushort* local_iteration,
                   __global float* global_test) {

    /*---------             Prepare memory             ---------------*/

    __global float* previous_predators;
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);

    /* Entry fetch (global -> local) */
    if (local_id == 0) {
        // Rule 1
        for (int i = 0; i < DIMENSIONS; i++)
            local_flock_pos_sum[i] = global_flock_pos_sum[i];

        *local_iteration = *global_iteration;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Select previous data out of the whole dataset */
    previous_predators = &global_generated_predators[FLOCK_INDEX(NUMBER_OF_PREDATORS, (int)(*local_iteration - 1), 0, 0)];

    /* Entry fetch (global -> private) */
    //predator_attack_vector = VLOADN(&global_predator_attack_vector[VAR_INDEX(global_id));
    //fetch_global(&global_predator_attack_vector[VAR_INDEX(global_id)], u_global_predator_attack_vector.a, DIMENSIONS);

    /*---------           Compute outputs           ---------------*/

    agents_predator_ai(global_generated_predators,
                       global_predator_attack_vector,
                       local_flock_pos_sum,
                       previous_predators,
                       local_iteration,
                       global_id,
                       local_id);

        // Maze bounds
    if (global_generated_predators[FLOCK_INDEX(NUMBER_OF_PREDATORS, *local_iteration, global_id, POS) + 0] >=
        MAZE_WIDTH)
        // Arbitrarily chosen offset
        global_generated_predators[FLOCK_INDEX(NUMBER_OF_PREDATORS, *local_iteration, global_id, POS) + 0] = MAZE_WIDTH - 0.1;
    if (global_generated_predators[FLOCK_INDEX(NUMBER_OF_PREDATORS, *local_iteration, global_id, POS) + 1] >=
        MAZE_HEIGHT)
        // Arbitrarily chosen offset
        global_generated_predators[FLOCK_INDEX(NUMBER_OF_PREDATORS, *local_iteration, global_id, POS) + 1] = MAZE_HEIGHT - 0.1;
}



__kernel __attribute__((reqd_work_group_size(MAX_WORK_GROUP_SIZE, 1, 1)))
void
k_boid_ai_preprocess(__global float* global_generated_flocks/*[TOTAL_TIMESTEPS][NUMBER_OF_BOIDS][ARRDIM]*/,
                     __global float* global_boid_avoidance_vector/*[NUMBER_OF_BOIDS]*/,
                     __global float* global_flock_vel_sum,
                     __local float* local_flock_vel_sum,
                     __global ushort* global_iteration,
                     __local ushort* local_iteration,
                     __global float* global_test) {

    /*---------Prepare memory and filter out surplus workitems---------------*/

    __global float* previous_flock;
    int global_id;
    int local_id;
    int main_boid_id;
    int compared_boid_id;

    global_id        = get_global_id(0);
    local_id         = get_local_id(0);
    main_boid_id     = trunc((float)global_id / (float)NUMBER_OF_BOIDS);
    compared_boid_id = fmod((float)global_id, NUMBER_OF_BOIDS);

    /* Entry fetch (global -> local) */
    if (local_id == 0) {
        *local_iteration = *global_iteration;
        for (int i = 0; i < DIMENSIONS; i++)
            local_flock_vel_sum[i] = 0.0f;
    }

    if (compared_boid_id == main_boid_id)
        // We don't compare a boid with itself
        return;

    barrier(CLK_LOCAL_MEM_FENCE);

    /* Select previous data out of the whole dataset */
    previous_flock = &global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS, (int)(*local_iteration - 1), 0, 0)];

    /*---------           Compute outputs           ---------------*/

    agents_boid_ai_preprocess(global_boid_avoidance_vector,
                              global_flock_vel_sum,
                              local_flock_vel_sum,
                              previous_flock,
                              compared_boid_id,
                              main_boid_id,
                              local_id);    
}



__kernel __attribute__((reqd_work_group_size(PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 1, 1)))
void
k_boid_ai(__global float* global_generated_flocks/*[TOTAL_TIMESTEPS][NUMBER_OF_BOIDS][ARRDIM]*/,
          __global float* global_generated_predators/*[TOTAL_TIMESTEPS][NUMBER_OF_PREDATORS][ARRDIM]*/,
          /*__global uchar* global_maze,
          __local uchar* local_maze,*/
          __global float* global_flock_pos_sum,
          __local float* local_flock_pos_sum,
          __global float* global_boid_avoidance_vector/*[NUMBER_OF_BOIDS]*/,
          __global float* global_flock_vel_sum,
          __local float* local_flock_vel_sum,
          __global ushort* global_iteration,
          __local ushort* local_iteration,
          __global float* global_test) {

    /*---------Prepare memory and filter out surplus workitems---------------*/

    __global float* previous_flock;
    __global float* previous_predators;
    int global_id;
    int local_id;
    float boid_avoidance_vector[DIMENSIONS];
    FLOATN impulse;
    FLOATN velocity;
    int i;//, j;
    
    global_id = get_global_id(0);
    local_id = get_local_id(0);

    // We have more workitems than boids
    if (global_id >= NUMBER_OF_BOIDS)
        return;

    /* Entry fetch (global -> local) */
    if (local_id == 0) {
        /*for (i = 0; i < MAZE_WIDTH; i++)
            for (j = 0; j < MAZE_HEIGHT; j++)
                local_maze[i * MAZE_WIDTH + j] = global_maze[i * MAZE_WIDTH + j];*/

        for (i = 0; i < DIMENSIONS; i++) {
            local_flock_pos_sum[i] = global_flock_pos_sum[i];
            local_flock_vel_sum[i] = global_flock_vel_sum[i];
        }

        *local_iteration = *global_iteration;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Select previous data out of the whole dataset */
    previous_flock     = &global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS, (int)(*local_iteration - 1), 0, 0)];
    previous_predators = &global_generated_predators[FLOCK_INDEX(NUMBER_OF_PREDATORS, (int)(*local_iteration - 1), 0, 0)];    
    
     /* Entry fetch (global -> private) */
    fetch_global(&global_boid_avoidance_vector[VAR_INDEX(global_id)], boid_avoidance_vector, DIMENSIONS);

    /*---------           Compute outputs           ---------------*/

    agent_ai(global_generated_flocks,
                   local_flock_pos_sum,
                   local_flock_vel_sum,
                   local_iteration,
                   boid_avoidance_vector,
                   previous_flock,
                   previous_predators,
                   &impulse,
                   global_id,
                   local_id);

    simulate_locomotion(global_generated_flocks,
                        previous_flock,
                        /*local_maze,*/
                        local_iteration,
                        impulse,
                        global_id,
                        &velocity);

    /*---------            Save outputs            ---------------*/

    // Move the boids
    VSTOREN(
        VLOADN(&previous_flock[BOID_INDEX(global_id,POS)]) + velocity,
        &global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS,*local_iteration,global_id,POS)]);

    // Update velocity
    VSTOREN(velocity, 
        &global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS,*local_iteration,global_id,VEL)]);
    if (global_id == 0) {
      global_test[0] = global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS,2,global_id,POS) + 0];
      global_test[1] = global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS,2,global_id,POS) + 1];
    }
}

__kernel __attribute__((reqd_work_group_size(PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 1, 1)))
void
k_update_values(__global float* global_generated_flocks/*[TOTAL_TIMESTEPS][NUMBER_OF_BOIDS][ARRDIM]*/,
                __global ushort* global_amendments_n,
                __global ushort* global_amendment_indices/*[amendments_n]*/,
                __local ushort* local_amendment_indices/*[amendments_n]*/,
                __global float* global_amendment_values /*[amendments_n][DIMENSIONS]*/,
                __local float* local_amendment_values /*[amendments_n][DIMENSIONS]*/,
                __global ushort* global_iteration,
                __local ushort* local_iteration) {

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
                local_amendment_values[VAR_INDEX(i) + j] = global_amendment_values[VAR_INDEX(i) + j];
        }

        *local_iteration = *global_iteration;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /*---------            Save outputs            ---------------*/
    
    for (i = 0; i < DIMENSIONS; i++)
        global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS, *local_iteration, local_amendment_indices[global_id], POS) + i] = 
            local_amendment_values[VAR_INDEX(global_id) + i];

    /* Exit clean up (global) */
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (global_id == 0)
        *global_amendments_n = 0;
}