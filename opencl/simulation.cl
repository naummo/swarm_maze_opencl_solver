#include "utilities.h"

void
dissolute_pheromones(__global float* global_map) {
    int x, y, boid_id;
    boolean is_node_outdated;
    for (x = 0; x < MAZE_WIDTH; x++)
        for (y = 0; y < MAZE_HEIGHT; y++) {
            if (global_map[INDEX_IN_MAP(x, y, NODE_PHEROMONE_A_VAR)] > NODE_ATTRACTANT_DISSOLUTION_COMPONENT)
                global_map[INDEX_IN_MAP(x, y, NODE_PHEROMONE_A_VAR)] -= NODE_ATTRACTANT_DISSOLUTION_COMPONENT;
            else
                global_map[INDEX_IN_MAP(x, y, NODE_PHEROMONE_A_VAR)] = 0;
            for (boid_id = 0; boid_id < NUMBER_OF_BOIDS; boid_id++) {
                if (global_map[INDEX_IN_MAP(x, y, NODE_PHEROMONE_M_VAR) + boid_id] > NODE_MARKER_DISSOLUTION_COMPONENT)
                    global_map[INDEX_IN_MAP(x, y, NODE_PHEROMONE_M_VAR) + boid_id] -= NODE_MARKER_DISSOLUTION_COMPONENT;
                else
                    global_map[INDEX_IN_MAP(x, y, NODE_PHEROMONE_M_VAR) + boid_id] = 0;
            }
            if (global_map[ INDEX_IN_MAP(x, y, NODE_IS_DEADEND_VAR )] == false) {                
                is_node_outdated = true;
                for (boid_id = 0; boid_id < NUMBER_OF_BOIDS; boid_id++) {
                    if (global_map[INDEX_IN_MAP(x, y, NODE_PHEROMONE_M_VAR) + boid_id] > 0) {
                        is_node_outdated = false;
                        break;
                    }
                }
                if (is_node_outdated == true)
                    global_map[INDEX_IN_MAP(x, y, NODE_IS_EXPLORED_VAR)] = false;
            }
        }
}



void
simulate_locomotion(__global float* global_previous_flock,/*[NUMBER_OF_BOIDS][ARRDIM]*/
                    __local ushort* local_iteration,
                    FLOATN impulse,
                    int global_id,
                    FLOATN* velocity) {
    *velocity = impulse;

    // Traction
    *velocity = *velocity * (1 - TRACTION);

    // Inertia
    *velocity = VLOADN(&global_previous_flock[INDEX_IN_FLOCK(global_id, BOID_VEL_VAR)]) * WEIGHTINERTIA +
                      *velocity * (1 - WEIGHTINERTIA);

    // Maze bounds
    if (global_previous_flock[INDEX_IN_FLOCK(global_id, BOID_POS_VAR)] + (*velocity).x > MAZE_WIDTH || 
        global_previous_flock[INDEX_IN_FLOCK(global_id, BOID_POS_VAR)] + (*velocity).x < 0)
        (*velocity).x = 0;
    if (global_previous_flock[INDEX_IN_FLOCK(global_id, BOID_POS_VAR) + 1] + (*velocity).y > MAZE_HEIGHT ||
        global_previous_flock[INDEX_IN_FLOCK(global_id, BOID_POS_VAR) + 1] + (*velocity).y < 0)
        (*velocity).y = 0;
}