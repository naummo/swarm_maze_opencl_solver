#include "utilities.h"

void
simulate_locomotion(__global float* global_generated_flocks/*[TOTAL_TIMESTEPS][NUMBER_OF_BOIDS][ARRDIM]*/,
                    __global float* previous_flock,
                    //__local uchar* local_maze,
                    __local ushort* local_iteration,
                    FLOATN impulse,
                    int global_id,
                    FLOATN* moving_vector/*,
                    FLOATN* dragging_vector*/) {
    *moving_vector = impulse;

    // Traction
    *moving_vector = *moving_vector * (1 - TRACTION);

    // Inertia
    *moving_vector = VLOADN(&previous_flock[BOID_INDEX(global_id,VEL)]) * WEIGHTINERTIA +
                      *moving_vector * (1 - WEIGHTINERTIA);

    // Maze bounds
    if (global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS, *local_iteration, global_id, POS) + 0] >=
        MAZE_WIDTH)
        // Arbitrarily chosen offset
        global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS, *local_iteration, global_id, POS) + 0] = MAZE_WIDTH - 0.1;
    if (global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS, *local_iteration, global_id, POS) + 1] >=
        MAZE_HEIGHT)
        // Arbitrarily chosen offset
        global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS, *local_iteration, global_id, POS) + 1] = MAZE_HEIGHT - 0.1;

    // Collision detection
    /* Colliding with a wall means moving along velocity vector until
       the wall is hit and dragging along the wall for some time, i.e. two vectors.
       The position will be updated using a sum of these two vectors, but in
       global_generated_flocks only the last vector will be saved.
       If no wall was hit, first vector is original velocity, and second is null. */
    /*
    moving_vector
    dragging_vector*/

}