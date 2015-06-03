#include "utilities.h"
#include "agents.h"

// Will be called NUMBER_OF_BOIDS times
void
agent_reynolds_rules13_preprocess(__local float* local_flock_pos_sum,
                                  __local float* local_flock_vel_sum,
                                  float* previous_position,
                                  float* previous_velocity) {
    // Reynold's rule 1. Cohesion - Steer to move towards the center of mass
    // Create partial (for <group_size> boids) sum in local memory
    atomic_add_local_arr(local_flock_pos_sum, previous_position);
    
    // Reynold's rule 3. Alignment - Steer towards the average heading of local flockmates
    // Create partial (for <group_size> boids) sum in local memory
    atomic_add_local_arr(local_flock_vel_sum, previous_velocity);
}



// Will be called NUMBER_OF_BOIDS * NUMBER_OF_BOIDS times
void
agent_reynolds_rule2_preprocess(__global float* previous_flock,
                                __global float* global_boid_avoidance_vectors,
                                int main_boid_id,
                                int compared_boid_id) {
    float adistance;
    FLOATN difference;

    // Reynold's rule 2. Separation - steer to avoid crowding local flockmates
    // Create an avoidance vector for each boid
    difference = VLOADN(&previous_flock[INDEX_IN_FLOCK(compared_boid_id, BOID_POS_VAR)]) - 
                    VLOADN(&previous_flock[INDEX_IN_FLOCK(main_boid_id, BOID_POS_VAR)]);
    adistance = length(difference);
    if (adistance > 0 && adistance < MINSEPARATION)
        atomic_add_global_v(&global_boid_avoidance_vectors[INDEX_IN_BOID(main_boid_id)], &difference);
}



void
agent_ai(__global float* global_map,
         __global uchar* global_cmm,
         __global float* global_boid_avoidance_vectors,
         __local float* local_flock_pos_sum,
         __local float* local_flock_vel_sum,
         __global float* global_previous_flock,
         __global float* global_new_flock,
         __global ushort* global_agent_programme,
         uchar* sensor_readings,
         int global_id,
         FLOATN* impulse,
         __global float* global_random,
         __local ushort* local_iteration,
         __global float* global_experiment,
         __global float* global_test) {
    uchar i;
    UCHARN current_node;
    float previous_boid[BOID_SIZE];
    uchar paths[4];
    uchar swamps[4];
    uchar forbidden_paths[4];
    uchar deadends_n = 0;
    uint trigram_ids[READINGS_N];
    uchar most_attractive_node;
    uchar unexplored_passes_n;
    boolean is_pheromone_level_shared;
    uchar chosen_path;
    uchar rotated_by;
    boolean is_agent_stalled = false;
    boolean is_wall_detected = false;
    boolean is_same_src_square = false;
    uchar passes_n;
    // North, East, South and West
    char delta_coors[8] = { 0, -1,
                           +1,  0,
                            0, +1,
                           -1,  0};

    /* Fetch boid's previous instance from global memory */
    for (i = 0; i < BOID_SIZE; i++)
        previous_boid[i] = global_previous_flock[INDEX_IN_FLOCK(global_id, 0) + i];

    /* Fetch boid's current instance from global memory */

    /* Get current coordinates */
    current_node = (UCHARN)(floor(previous_boid[INDEX_IN_BOID(BOID_POS_VAR)]),
                            floor(previous_boid[INDEX_IN_BOID(BOID_POS_VAR) + 1]));
    
    /* Check if agent has stayed on the same square */
    if (current_node.x == floor(int_to_float(global_agent_programme[PROG_SRC_COORS_VAR])) &&
        current_node.y == floor(int_to_float(global_agent_programme[PROG_SRC_COORS_VAR + 1])))
        is_same_src_square = true;

    /* Update the previous chosen paths */
    if (is_same_src_square == false)  {
        global_agent_programme[PROG_PREV_PREV_PATH_VAR] = global_agent_programme[PROG_PREV_PATH_VAR];
        global_agent_programme[PROG_PREV_PATH_VAR] = global_agent_programme[PROG_CHOSEN_PATH_VAR];
    }

    /* Preproccess sensor readings for CMM */
    preprocess_for_cmm(sensor_readings,
                       &rotated_by,
                       trigram_ids,
                       global_test);

    /* Consult the programme and update CMM:
        - If agent is in a swamp and the corresponding SLOW_GO weights are not set, set them.
        - If agent is in clear passage and there are swamps surrounding the previous node, mark
        them as NO_GO.
        - If agent hasn't moved along one of the axis, the goal square is a wall. Set it to NO_GO. */
    update_cmm(global_agent_programme,
               global_cmm,
               current_node,
               sensor_readings,
               previous_boid,
               &is_agent_stalled,
               &is_wall_detected,
               rotated_by,
               trigram_ids,
               global_test);

    // Choose a new goal in any case - if agent has reached the goal, it needs a new one, and
    // if it hasn't, the exploration statuses of surrounding nodes might have changed and
    // agent may have to switch to a new goal

    /* Get available paths */
    get_paths(global_cmm,
              global_map,
              global_agent_programme,
              current_node,
              paths,
              swamps,
              forbidden_paths,
              &deadends_n,
              trigram_ids,
              rotated_by,
              delta_coors,
              global_test);
    
    /* Gather information about available passages */
    browse_passages(global_map,
                    &is_pheromone_level_shared,
                    &most_attractive_node,
                    &unexplored_passes_n,
                    paths,
                    forbidden_paths,
                    current_node,
                    delta_coors,
                    &passes_n,
                    global_test);

    /* Choose a desired direction: */
    chosen_path = choose_passage(global_map,
                                 local_flock_pos_sum,
                                 local_flock_vel_sum,
                                 global_boid_avoidance_vectors,
                                 global_agent_programme,
                                 global_id,
                                 previous_boid,
                                 unexplored_passes_n,
                                 is_pheromone_level_shared,
                                 most_attractive_node,
                                 paths,
                                 forbidden_paths,
                                 current_node,
                                 delta_coors,
                                 global_test);

    /* Get impulse vector from the current position to the 
       middle point of the edge of the chosen passage */
    get_impulse(previous_boid,
                chosen_path,
                impulse);

    /* Update agent's programme with info about the new goal */
    update_programme(global_map,
                     global_agent_programme,
                     previous_boid,
                     current_node,
                     chosen_path,
                     delta_coors,
                     swamps,
                     deadends_n,
                     rotated_by,
                     trigram_ids,
                     is_agent_stalled,
                     is_wall_detected,
                     is_same_src_square,
                     passes_n,
                     unexplored_passes_n,
                     sensor_readings,
                     global_random,
                     global_test);

    /* Update node and boid's pheromone levels */
    update_pheromone_levels(global_map,
                            global_new_flock,
                            global_agent_programme,
                            previous_boid,
                            current_node,
                            global_id,
                            unexplored_passes_n,
                            is_same_src_square,
                            is_wall_detected,
                            chosen_path,
                            paths,
                            forbidden_paths,
                            delta_coors,
                            global_test);
    /* Update node attributes other than pheromone level */
    update_node_status(global_map,
                       global_agent_programme,
                       passes_n,
                       unexplored_passes_n,
                       current_node,
                       global_test);

    // TEST
    global_test[0] = chosen_path;
    global_test[1] = passes_n;
    global_test[2] = unexplored_passes_n;
    global_test[3] = deadends_n;
    global_test[8] = global_agent_programme[PROG_LEAVING_GOAL_VAR];
    global_test[9] = global_agent_programme[PROG_LEAVING_DEADEND_VAR];

    // Experiment results
    if (is_wall_detected) {
        int hit_count = (int)global_experiment[EXP_BOID_HIT_COUNT_VAR];
        global_experiment[EXP_BOID_HITS_VAR + hit_count] = (float)*local_iteration;
        global_experiment[EXP_BOID_HIT_COUNT_VAR] = hit_count + 1;
    }
}



void
preprocess_for_cmm(uchar* sensor_readings,
                   uchar* rotated_by,
                   uint* trigram_ids,
                   __global float* global_test) {
    // Sensor readings are split into 4 trigrams. In original array trigrams are ordered by
    // the index of their first element in the array of readings.
    uint smallest_trigram_id = 0;
    uint biggest_trigram_id = 0;
    uint unsorted_trigram_ids[TRIGRAMS_N];
    //uchar rotated_id;
    int i, j;
    for (i = 0; i < TRIGRAMS_N; i++) {
        unsorted_trigram_ids[i] = 0;
        unsorted_trigram_ids[i] = sensor_readings[i] * SQUARE_TYPES_N2 +
                                  sensor_readings[(i + 1 < TRIGRAMS_N ? i + 1 : i + 1 - TRIGRAMS_N)] * SQUARE_TYPES_N +
                                  sensor_readings[(i + 2 < TRIGRAMS_N ? i + 2 : i + 2 - TRIGRAMS_N)];
    }
    // Rows in CMM are not associated with any orientation, so we need to rotate outputs.
    // In the CMM North is associated with the first element of the smallest trigram
    // in a quadruple of trigrams. (in original array the quadruple is sorted in such a way 
    // that North is associated with the first element in the first trigram)
    for (i = 1; i < TRIGRAMS_N; i++) {
        if (unsorted_trigram_ids[i] < unsorted_trigram_ids[smallest_trigram_id])
            smallest_trigram_id = i;
        if (unsorted_trigram_ids[i] > unsorted_trigram_ids[biggest_trigram_id])
            biggest_trigram_id = i;
    }
    // smallest_trigram_id is the order of the smallest trigram in the quadruple
    // It is also the real orientation of the wall, which in CMM corresponds to NORTH
    // Hence it is a value by which we should shift the array
    *rotated_by = smallest_trigram_id;
    trigram_ids[0] = unsorted_trigram_ids[smallest_trigram_id];
    unsorted_trigram_ids[smallest_trigram_id] = unsorted_trigram_ids[biggest_trigram_id] + 1;
    for (i = 1; i < TRIGRAMS_N; i++) {
        for (j = 0; j < TRIGRAMS_N; j++)
            if (unsorted_trigram_ids[j] < unsorted_trigram_ids[smallest_trigram_id])
                smallest_trigram_id = j;
        trigram_ids[i] = unsorted_trigram_ids[smallest_trigram_id];
        unsorted_trigram_ids[smallest_trigram_id] = unsorted_trigram_ids[biggest_trigram_id] + 1;
    }
}


void
update_cmm(__global ushort* global_agent_programme,
           __global uchar* global_cmm,
           UCHARN current_node,
           uchar* sensor_readings,
           float* previous_boid,
           boolean* is_agent_stalled,
           boolean* is_wall_detected,
           uchar rotated_by,
           uint* trigram_ids,
           __global float* global_test) {
    int i, j;
    /* Consult the programme and update CMM:
        - If agent is in a swamp and the corresponding SLOW_GO weights are not set, set them.
        - If agent is in clear passage and there are swamps surrounding the previous node, mark
        them as NO_GO.
        - If agent hasn't moved along one of the axis, the goal square is a wall. Set it to NO_GO. */
    if (global_agent_programme[PROG_EDGE_VAR] != GOAL_NOT_SET)
        if (global_agent_programme[PROG_GOAL_SQUARE_VAR] == current_node.x &&
            global_agent_programme[PROG_GOAL_SQUARE_VAR + 1] == current_node.y) {
            // If agent has reached the goal
            if (sensor_readings[READING_BOTTOM] == true) {
                // If agent is in a swamp

                // Change CMM: for each trigram set that side's SLOW_GO to 1
                if (global_agent_programme[PROG_SWAMP_GOAL_VAR] == false) {
                    // If it hasn't been done already,
                    // change CMM: for each trigram set that side's SLOW_GO to 1
                    char neuron_id;
                    // global_agent_programme[PROG_CHOSEN_PATH_VAR] is relative to the agent's current orientation.
                    // In CMM weights are set relative to the trigram with the smallest id out of the quadruple.
                    neuron_id = global_agent_programme[PROG_CHOSEN_PATH_VAR] - global_agent_programme[PROG_ROTATED_BY_VAR];
                    neuron_id = (neuron_id >= 0 ? neuron_id : 4 + neuron_id);
                    // SLOW_GO neurons start at index 4
                    neuron_id += 4;
                    for (i = 0; i < TRIGRAMS_N; i++) {
                        // Trigram IDs are sorted
                        global_cmm[trigram_ids[i]] = 
                            global_cmm[trigram_ids[i]] | (1 << neuron_id);
                    }
                }
            }
            else {
                // If agent is not in a swamp, we are now sure there is
                // at least one more passage besides swamp, so:
                // Change CMM - for each trigram out of the original quadruple
                // for each side if it has SLOW_GO set, remove SLOW_GO and set the side to NO_GO
                for (i = 0; i < TRIGRAMS_N; i++) {
                    for (j = 0; j < TRIGRAMS_N; j++) {
                        // Trigram IDs are sorted
                        // SLOW_GO neurons start at index 4
                        if (global_cmm[trigram_ids[i]] & (1 << (j + 4)) == true) {
                            // If SLOW_GO is set for this side, unset it and unset GO for this side
                            // This means that as we know there is a clear passage, going via
                            // other swamps in undesirable
                            global_cmm[trigram_ids[i]] =
                                global_cmm[trigram_ids[i]] &
                                (0xFF ^ (0x11 << j)); //(0b11111111 ^ (0b10001 << j))
                        }
                    }
                }
            }
        }
        else {// If agent hasn't reached the goal
            if ((global_agent_programme[PROG_EDGE_VAR] == GOAL_EDGE_VERTICAL &&
                 in_proximity(previous_boid[INDEX_IN_BOID(BOID_POS_VAR)],
                              int_to_float(global_agent_programme[PROG_SRC_COORS_VAR]))) ||
                (global_agent_programme[PROG_EDGE_VAR] == GOAL_EDGE_HORIZONTAL &&
                 in_proximity(previous_boid[INDEX_IN_BOID(BOID_POS_VAR) + 1],
                              int_to_float(global_agent_programme[PROG_SRC_COORS_VAR + 1])))) {
                // If appropriate coordinate has NOT changed
                *is_agent_stalled = true;
            }
            if (*is_agent_stalled == false) {
                ushort chosen_path = global_agent_programme[PROG_CHOSEN_PATH_VAR];
                float src_coors[2] = {int_to_float(global_agent_programme[PROG_SRC_COORS_VAR]),
                                      int_to_float(global_agent_programme[PROG_SRC_COORS_VAR + 1])};
                // Detect if the agent was pushed backwards by rotating on the same spot near a wall
                if ((chosen_path == NORTH &&
                     previous_boid[INDEX_IN_BOID(BOID_POS_VAR) + 1] > src_coors[1]) ||
                    (chosen_path == EAST &&
                     previous_boid[INDEX_IN_BOID(BOID_POS_VAR)] < src_coors[0]) ||
                    (chosen_path == SOUTH &&
                     previous_boid[INDEX_IN_BOID(BOID_POS_VAR) + 1] < src_coors[1]) ||
                    (chosen_path == WEST &&
                     previous_boid[INDEX_IN_BOID(BOID_POS_VAR)] > src_coors[0])) {
                    *is_agent_stalled = true;
                }
            }
            if (*is_agent_stalled == true) {              
                if (global_agent_programme[PROG_STALL_TIME_VAR] == MAX_STALL_TIME_ALLOWED) {
                    // Wall detected - the agent has been stalling for too long
                    *is_wall_detected = true;
                    // Change CMM - for each trigram set goal edge side as NO_GO (i.e. unset GO)
                    char rotated_path_id;
                    // global_agent_programme[PROG_CHOSEN_PATH_VAR] is relative to the agent's current orientation.
                    // In CMM weights are set relative to the trigram with the smallest id out of the quadruple.
                    rotated_path_id = global_agent_programme[PROG_CHOSEN_PATH_VAR] - global_agent_programme[PROG_ROTATED_BY_VAR];
                    rotated_path_id = rotated_path_id >= 0 ? rotated_path_id : 4 + rotated_path_id;
                    for (i = 0; i < TRIGRAMS_N; i++) {
                        global_cmm[trigram_ids[i]] =
                            global_cmm[trigram_ids[i]] &
                            (0xFF ^ (1 << rotated_path_id)); //(0b11111111 ^ (1 << rotated_path_id));
                        // We don't care about SLOW_GO for this side here, and it should not be set anyway
                    }
                }
            }
        }
}



uchar
in_proximity(float what, float of) {
    return (what > of - ERROR_MARGIN &&
            what < of + ERROR_MARGIN);
}



void
get_paths(__global uchar* global_cmm,
          __global float* global_map,
          __global ushort* global_agent_programme,
          UCHARN current_node,
          uchar* paths,
          uchar* swamps,
          uchar* forbidden_paths,
          uchar* deadends_n,
          uint* trigram_ids,
          uchar rotated_by,
          char* delta_coors,
          __global float* global_test) {
    uchar i, j;
    uchar weights;
    uchar rotated_id;
    // Inputs to neurons
    int inputs[NEURONS_N];
    // Clean memory
    for (j = 0; j < NEURONS_N; j++) 
        inputs[j] = 0;

    for (i = 0; i < TRIGRAMS_N; i++) {
        weights = global_cmm[trigram_ids[i]];
        // Update inputs to neurons
        // CMM columns are NORTH-SLOW, EAST-SLOW, SOUTH-SLOW, WEST-SLOW,
        //                 NORTH, EAST, SOUTH, WEST
        // E.g. (0b 0000 0100) is EAST = GO
        // E.g. (0b 0001 1000) is NORTH = SLOW_GO and SOUTH = GO
        for (j = 0; j < NEURONS_N; j++) 
            inputs[j] = inputs[j] + ((weights >> j) & 1);
    }

    // Get list of paths
    for (i = 0; i < 4; i++) {
        rotated_id = i + rotated_by < TRIGRAMS_N ?
                     i + rotated_by :
                     i + rotated_by - TRIGRAMS_N;
        paths[rotated_id] = false;
        // Apply fixed threshold and don't let coordinates get negative
        if (inputs[i] >= NEURON_THRESHOLD &&
            DELTA_COOR_X( current_node, rotated_id ) >= 0 &&
            DELTA_COOR_Y( current_node, rotated_id ) >= 0)
            paths[rotated_id] = true;
    }
    // Get list of swamps
    for (i = 4; i < 8; i++) {
        rotated_id = i + rotated_by < TRIGRAMS_N ?
                     i + rotated_by :
                     i + rotated_by - TRIGRAMS_N;
        swamps[rotated_id] = false;
        // Apply fixed threshold
        // The last reading is not considered here
        if (inputs[i] > TRIGRAMS_N)
            swamps[rotated_id] = true;
    }
    // Filter the lists based on:
    // Does it lead to deadend?
    // Have agent entered into a loop between current node and the suggested path
    for (i = 0; i < 4; i++) {
        forbidden_paths[i] = false;
        if (global_map[ INDEX_IN_MAP(DELTA_COOR_X( current_node, i ),
                                     DELTA_COOR_Y( current_node, i ),
                                     NODE_IS_DEADEND_VAR)] == true ||
            global_map[ INDEX_IN_MAP(DELTA_COOR_X( current_node, i ),
                                     DELTA_COOR_Y( current_node, i ),
                                     NODE_IS_GOAL_VAR)] == true) {
            *deadends_n += 1;
            forbidden_paths[i] = true;
        }
        if (i == reverse_direction(global_agent_programme[PROG_PREV_PATH_VAR]) &&
            i == global_agent_programme[PROG_PREV_PREV_PATH_VAR]) {
            // The square before previous one that agent visited is the current square
            // The agent returned to the same square via the same path
            forbidden_paths[i] = true;
        }
    }
}



void
browse_passages(__global float* global_map,
                boolean* is_pheromone_level_shared,
                uchar* most_attractive_node,
                uchar* unexplored_passes_n,
                boolean* is_passage,
                boolean* is_passage_forbidden,
                UCHARN current_node,
                char* delta_coors,
                uchar* passes_n,
                __global float* global_test) {
    float max_pheromone_level = 0;
    uchar forbidden_paths_n = 0;
    int i;
    *most_attractive_node = 0;
    *unexplored_passes_n = 0;
    *is_pheromone_level_shared = false;
    *passes_n = 0;
    // If all passes are forbidden, let's permit all of them so that
    // later a path will be chosen based on Reynold's rules
    for (i = 0; i < 4; i++) {
        if (is_passage[i]) {
            *passes_n += 1;
            if (is_passage_forbidden[i] == true)
                forbidden_paths_n += 1;
        }
    }
    if (forbidden_paths_n == *passes_n) {
        for (i = 0; i < 4; i++)
            is_passage_forbidden[i] = false;
        }

    for (i = 0; i < 4; i++) {
        // For 4 possible directions
        if (is_passage[i]) {
            // If neural network says the square is a passage, not an obstacle
            if (global_map[ INDEX_IN_MAP(DELTA_COOR_X( current_node, i ),
                                         DELTA_COOR_Y( current_node, i ),
                                         NODE_IS_EXPLORED_VAR)]) {
                // If the square is explored
                if (is_passage_forbidden[i] == false) {
                    // If the square is not forbidden. Otherwise we are not interested in its attractant level
                    if (global_map[ INDEX_IN_MAP( DELTA_COOR_X( current_node, i ),
                                                  DELTA_COOR_Y( current_node, i ),
                                                  NODE_PHEROMONE_A_VAR)] > max_pheromone_level) {
                        // If square has the highest attractant level so far, save it
                        *most_attractive_node = i;
                        max_pheromone_level = global_map[ 
                            INDEX_IN_MAP( DELTA_COOR_X( current_node, i ),
                                          DELTA_COOR_Y( current_node, i ),
                                          NODE_PHEROMONE_A_VAR)];
                        *is_pheromone_level_shared = false;
                    }
                    else
                        if (global_map[ INDEX_IN_MAP( DELTA_COOR_X( current_node, i ),
                                                      DELTA_COOR_Y( current_node, i ),
                                                      NODE_PHEROMONE_A_VAR)] == max_pheromone_level)
                            *is_pheromone_level_shared = true;
                }
            }
            else {
                // If the square is not explored
                *unexplored_passes_n += 1;
            }
        }
    }
}



uchar
choose_passage(__global float* global_map,
               __local float* local_flock_pos_sum,
               __local float* local_flock_vel_sum,
               __global float* global_boid_avoidance_vectors,
               __global ushort* global_agent_programme,
               int global_id,
               float* previous_boid,
               uchar unexplored_passes_n,
               boolean is_pheromone_level_shared,
               uchar most_attractive_node,
               boolean* is_passage,
               boolean* is_passage_forbidden,
               UCHARN current_node,
               char* delta_coors,
               __global float* global_test) {
    boolean is_path_chosen = false;
    uchar chosen_path = 0;
    int i;
    // If all paths are explored, choose the one that leads to a node
    if (unexplored_passes_n == 0 && is_pheromone_level_shared == false) {        
        chosen_path = most_attractive_node;
        is_path_chosen = true;
    }
    else
        // If only one unexplored path - choose it
        if (unexplored_passes_n == 1) {
            for (i = 0; i < 4; i++)
                if (is_passage[i] && is_passage_forbidden[i] == false &&
                   global_map[ INDEX_IN_MAP(DELTA_COOR_X( current_node, i ),
                                            DELTA_COOR_Y( current_node, i ),
                                            NODE_IS_EXPLORED_VAR) ] == false) {
                    chosen_path = i;
                    is_path_chosen = true;
                    break;
                }
        }

    if (is_path_chosen == false) {
        /* The choice of direction is not that easy and we'll need a Reynold's vector */
        FLOATN reynolds_vector;
        float reynolds_angle;
        float half_pi = M_PI / 2;
        float angle_difference;
        float minimum_angle = 2 * M_PI;

        // Calculate Reynold's rules vector
        // Reynold's rule 1. Cohesion - Steer to move towards the center of mass
        // Center of mass vector was calculated during preprocessing
        // Here we are using previous coordinates as the current ones will get changed in parallel
        reynolds_vector = (VLOADN(local_flock_pos_sum) / NUMBER_OF_BOIDS - VLOADN(&previous_boid[INDEX_IN_BOID(BOID_POS_VAR)])) *
                        WEIGHTCENTEROFMASS;

        // Reynold's rule 2. Separation - steer to avoid crowding local flockmates
        // Avoidance vectors were calculated during preprocessing
        reynolds_vector = reynolds_vector - VLOADN(&global_boid_avoidance_vectors[INDEX_IN_BOID(global_id)]) * WEIGHTSEPARATION;

        // Reynold's rule 3. Alignment - Steer towards the average heading of local flockmates
        // Center of flock velocity was calculated during preprocessing
        reynolds_vector = reynolds_vector + (VLOADN(local_flock_vel_sum) / NUMBER_OF_BOIDS -
            VLOADN(&previous_boid[INDEX_IN_BOID(BOID_VEL_VAR)])) * WEIGHTALIGNMENT;

        // Reynold's vector angle
        reynolds_angle = atan(reynolds_vector.y / reynolds_vector.x);
        // Rotate and flip so that axis X > +positive is 0, axis Y > +posiitve is pi/2
        // and axis Y < -negative is 3pi/2
        if (reynolds_vector.x < 0)
            reynolds_angle = M_PI + reynolds_angle;
        if (reynolds_vector.x > 0 && reynolds_vector.y < 0)
            reynolds_angle = 2 * M_PI + reynolds_angle;
    
        if (unexplored_passes_n > 0) {
            // If multiple unexplored paths are available
            // choose the path, to which the Reynold's vector is closest
            for (i = 0; i < 4; i++) {
                if (is_passage[i] && is_passage_forbidden[i] == false &&
                   global_map[ INDEX_IN_MAP(DELTA_COOR_X( current_node, i ), 
                                            DELTA_COOR_Y( current_node, i ),
                                            NODE_IS_EXPLORED_VAR)] == false) {
                    // Inline if for translating North=0, East=1 etc into North=1,
                    // East=0, South=3, West=2 (half_pi multiplier)
                    angle_difference = min(fabs(reynolds_angle - 
                                                (1 - i >= 0 ? 1 - i : 5 - i) * half_pi),
                                           fabs((float)M_PI - reynolds_angle - (1 - i >= 0 ? 1 - i : 5 - i) * half_pi));
                    if (angle_difference < minimum_angle) {
                        minimum_angle = angle_difference;
                        chosen_path = i;
                    }                    
                }
            }
        }
        else
            if (is_pheromone_level_shared) {
                reynolds_angle = global_agent_programme[PROG_RAND_VAR];
                // If multiple explored paths have maximum attractant level
                // Choose the path, to which the Reynold's vector is closest
                for (i = 0; i < 4; i++) {
                    if (is_passage[i] && is_passage_forbidden[i] == false &&
                       global_map[ INDEX_IN_MAP(DELTA_COOR_X( current_node, i ),
                                                DELTA_COOR_Y( current_node, i ),
                                                NODE_IS_EXPLORED_VAR)] == true &&
                       global_map[ INDEX_IN_MAP( DELTA_COOR_X( current_node, i ),
                                                 DELTA_COOR_Y( current_node, i ),
                                                 NODE_PHEROMONE_A_VAR)] == 
                       global_map[ INDEX_IN_MAP( DELTA_COOR_X( current_node, most_attractive_node),
                                                 DELTA_COOR_Y( current_node, most_attractive_node),
                                                 NODE_PHEROMONE_A_VAR )]) {
                        // Inline if for translating North=0, East=1 etc into North=1,
                        // East=0, South=3, West=2 (half_pi multiplier)
                        angle_difference = fabs(reynolds_angle - 
                                                (1 - i >= 0 ? 1 - i : 5 - i) * half_pi);
                        if (angle_difference < minimum_angle) {
                            minimum_angle = angle_difference;
                            chosen_path = i;
                        }
                    }
                }
            }
    }
    return chosen_path;
}



void
get_impulse(float* previous_boid,
            uchar chosen_path,
            FLOATN* impulse) {
    float goal_point[DIMENSIONS];

    if (chosen_path == NORTH) {
        goal_point[0] = floor(previous_boid[INDEX_IN_BOID(BOID_POS_VAR)]) + 0.5;
        goal_point[1] = floor(previous_boid[INDEX_IN_BOID(BOID_POS_VAR) + 1]);
    }
    if (chosen_path == EAST) {
        goal_point[0] = ceil(previous_boid[INDEX_IN_BOID(BOID_POS_VAR)]);
        goal_point[1] = floor(previous_boid[INDEX_IN_BOID(BOID_POS_VAR) + 1]) + 0.5;
    }
    if (chosen_path == SOUTH) {
        goal_point[0] = floor(previous_boid[INDEX_IN_BOID(BOID_POS_VAR)]) + 0.5;
        goal_point[1] = ceil(previous_boid[INDEX_IN_BOID(BOID_POS_VAR) + 1]);
    }
    if (chosen_path == WEST) {
        goal_point[0] = floor(previous_boid[INDEX_IN_BOID(BOID_POS_VAR)]);
        goal_point[1] = floor(previous_boid[INDEX_IN_BOID(BOID_POS_VAR) + 1]) + 0.5;
    }
    // Create a vector directed from the previous position to the goal point
    *impulse = VLOADN(goal_point) - VLOADN(&previous_boid[INDEX_IN_BOID(BOID_POS_VAR)]);

    // Scale vector according to the set speed
    *impulse = *impulse * (BOIDVELOCITY / sqrt(pow((*impulse).x, 2) + pow((*impulse).y, 2)));
}



void
update_programme(__global float* global_map,
                 __global ushort* global_agent_programme,
                 float* previous_boid,
                 UCHARN current_node,
                 uchar chosen_path,
                 char* delta_coors,
                 uchar* swamps,
                 uchar deadends_n,
                 uchar rotated_by,
                 uint* trigram_ids,
                 boolean is_agent_stalled,
                 boolean is_wall_detected,
                 boolean is_same_src_square,
                 uchar passes_n,
                 uchar unexplored_passes_n,
                 uchar* sensor_readings,
                 __global float* global_random,
                 __global float* global_test) {
    boolean is_same_goal_square = false;
    global_agent_programme[PROG_RAND_CLOCK_VAR] += 1;
    if (global_agent_programme[PROG_RAND_CLOCK_VAR] == RAND_CLOCK_MAX) {
        global_agent_programme[PROG_RAND_CLOCK_VAR] = 0;
        global_agent_programme[PROG_RAND_VAR] = *global_random;
    }
    if (global_agent_programme[PROG_GOAL_SQUARE_VAR] == current_node.x +
                                                        delta_coors[chosen_path * 2] &&
        global_agent_programme[PROG_GOAL_SQUARE_VAR + 1] == current_node.y +
                                                            delta_coors[chosen_path * 2 + 1]) {
        // New goal is the same as the old one
        is_same_goal_square = true;
    }
    if (is_same_goal_square == true) { 
        // The variables below are to be updated even if the goal is the same
        /* Stall time counter (Agent is allowed to stall for some time to take into account an inertia) */
        if (is_agent_stalled)
            global_agent_programme[PROG_STALL_TIME_VAR] += 1;
        else
            global_agent_programme[PROG_STALL_TIME_VAR] = 0;
    }
    else {
        // The variables below don't need to be updated if the goal hasn't changed

        /* Is agent near a goal path? */
        boolean is_near_goal_path = false;
        for (int i = 0; i < 4; i++) {
            if (global_map[INDEX_IN_MAP(DELTA_COOR_X( current_node, i ), 
                                        DELTA_COOR_Y( current_node, i ),
                                        NODE_IS_GOAL_VAR)] == true) {
                is_near_goal_path = true;
                break;
            }
        }
        if (passes_n - deadends_n <= 1) {
            if (is_near_goal_path == true || 
                sensor_readings[4] == ENTRANCE_ID ||
                sensor_readings[4] == EXIT_ID) {
                // Just visited a goal node or near the path to the goal
                global_agent_programme[PROG_LEAVING_GOAL_VAR] = TRUE;
                global_agent_programme[PROG_LEAVING_DEADEND_VAR] = TRUE;
            }
        }  
        /* Is the agent leaving the deadend? */
        if (global_agent_programme[PROG_LEAVING_DEADEND_VAR] == false) {
            // Agent hasn't been leaving deadend before
            if (passes_n == 1) {
                // Just visited a deadend
                global_agent_programme[PROG_LEAVING_DEADEND_VAR] = TRUE;
            }            
        }
        else {
            if (global_agent_programme[PROG_LEAVING_DEADEND_VAR] == TRUE) {
                // Agent has been leaving deadend before

                if (passes_n - deadends_n - unexplored_passes_n > 1) {
                    // Agent is on crossroads and there are at least two explored paths
                    // (unexplored paths can be walls, so this is not actually crossroads)
                    global_agent_programme[PROG_LEAVING_DEADEND_VAR] = false;
                    global_agent_programme[PROG_LEAVING_GOAL_VAR] = false;
                    // If there is an unexplored path, the agent will first check it out before
                    // unsetting the PROG_LEAVING_DEADEND_VAR flag. If the unexplored square turns
                    // out to be a wall, agent continues leaving the deadend. If it's a passage,
                    // the progamme will be reset including the PROG_LEAVING_DEADEND_VAR flag.
                }
                else
                    if (unexplored_passes_n > 0 && passes_n > 1) {
                        // There are at least 2 known paths (one of the is the one agent came from)
                        // and at least 1 unknown. Agent will take the unknown one which may be
                        // a passage or a wall.
                        global_agent_programme[PROG_LEAVING_DEADEND_VAR] = MAYBE;
                    }
            }
            else { //if (global_agent_programme[PROG_LEAVING_DEADEND_VAR] == MAYBE) 
                if (is_same_src_square == true) {
                    if (is_wall_detected) {
                        // If agent was checking if it's on crossroads or not - bumping into a wall
                        // means we haven't confirmed it's crossroads
                        if (unexplored_passes_n == 0 && passes_n - deadends_n <= 1) {
                            // If there are no unexplored paths anymore or it's another deadend
                            global_agent_programme[PROG_LEAVING_DEADEND_VAR] = TRUE;
                        }
                    }
                }
                else {
                    // Agent has successfully left crossroads
                    global_agent_programme[PROG_LEAVING_DEADEND_VAR] = false;                    
                    global_agent_programme[PROG_LEAVING_GOAL_VAR] = false;
                }
            }
        }
        /* Stall time counter */
        global_agent_programme[PROG_STALL_TIME_VAR] = 0;

        /* Goal square coors */
        global_agent_programme[PROG_GOAL_SQUARE_VAR] = (float)current_node.x +
                                                       delta_coors[chosen_path * 2];
        global_agent_programme[PROG_GOAL_SQUARE_VAR + 1] = (float)current_node.y +
                                                           delta_coors[chosen_path * 2 + 1];

        /* Edge orientation (GOAL_NOT_SET initially) */
        if (chosen_path == NORTH || chosen_path == SOUTH)
            global_agent_programme[PROG_EDGE_VAR] = GOAL_EDGE_HORIZONTAL;
        else
            global_agent_programme[PROG_EDGE_VAR] = GOAL_EDGE_VERTICAL;

        /* Directions in CMM are always orientated with the assumption that
        the first reading in the trigram with the smallest ID out of the quadruple
        is from a northern sensor. If we trigrams in quadruple in increasing order of IDs,
        the smallest trigram order equals rotated_by value - a rotation offset. */
        global_agent_programme[PROG_ROTATED_BY_VAR] = rotated_by;

        /* Do we know that the chose path leats to the swamp? If we don't and it does lead
        to a swamp, CMM will need to be updated. */
        if (swamps[chosen_path] == true)
            global_agent_programme[PROG_SWAMP_GOAL_VAR] = true;

        /* Chosen path ID (NORTH, EAST etc) */
        global_agent_programme[PROG_CHOSEN_PATH_VAR] = chosen_path;
    }
    /* Source position coors */
    global_agent_programme[PROG_SRC_COORS_VAR] = float_to_int(previous_boid[INDEX_IN_BOID(BOID_POS_VAR)]);
    global_agent_programme[PROG_SRC_COORS_VAR + 1] = float_to_int(previous_boid[INDEX_IN_BOID(BOID_POS_VAR) + 1]);    
}



void
update_pheromone_levels(__global float* global_map,
                        __global float* global_new_flock,
                        __global ushort* global_agent_programme,
                        float *previous_boid,
                        UCHARN current_node,
                        int global_id,
                        uchar unexplored_passes_n,
                        boolean is_same_src_square,
                        boolean is_wall_detected,
                        uchar chosen_path,
                        boolean* is_passage,
                        boolean* is_passage_forbidden,
                        char* delta_coors,
                        __global float* global_test) {
    float new_boid_pheromone_level;
    // Attractant pheromone level shows how many unexplored paths agent has witnessed
    // on its way up to the current moment, which it hasn't visited.
    // As an agent visits nodes, it adds its attractant pheromones to node attractant pheromones
    // in order to attract other agents to the route it take as it leads to unexplored nodes.

    // Get agent's previous attractant pheromone level
    new_boid_pheromone_level = previous_boid[ INDEX_IN_BOID( BOID_PHEROMONE_VAR )]; 
    // Dissolute it slighty to simulate attractant pheromone ageing
    if (new_boid_pheromone_level > BOID_ATTRACTANT_DISSOLUTION_COMPONENT)
        new_boid_pheromone_level -= BOID_ATTRACTANT_DISSOLUTION_COMPONENT;
    else
        new_boid_pheromone_level = 0;

    // Update boid's marker pheromone on this node
    global_map[ INDEX_IN_MAP( current_node.x,
                              current_node.y,
                              NODE_PHEROMONE_M_VAR ) + global_id] = 1;

    if (is_wall_detected)
        // The wall agent has just detected was previously an unknown path and has increased
        // agent's attractant pheromone level by 1. Here we reverse it.
        if (new_boid_pheromone_level > 1)
            new_boid_pheromone_level -= 1;
        else
            new_boid_pheromone_level = 0;

    // We don't fetch node pheromone level into the private memory as it is a shared memory
    // and can get updated in parallel by other agents, so operations have to be as atomic
    // as possible
    if (global_agent_programme[PROG_LEAVING_DEADEND_VAR] == TRUE) {
        // The agent is on a path to a deadend and still on an unexplored track
        if (unexplored_passes_n == 0) {
            // Agent will only remove all node pheromones if it is sure it's not on crossroads
            global_map[ INDEX_IN_MAP( current_node.x, current_node.y, NODE_PHEROMONE_A_VAR )] = 0;
        }
    }
    else {
        if (is_same_src_square == false) {
            // Update node attractant pheromone level
            uchar trailing_itself = false;
            if (unexplored_passes_n == 0) {
                // An agent can be trailing itself only if there are no unexplored passes available
                trailing_itself = true;
                /*uchar current_node_pheromones = 
                    global_map[ INDEX_IN_MAP( current_node.x, current_node.y, NODE_PHEROMONE_A_VAR )];
                for (int i = 0; i < 4; i++) {
                    if (i != chosen_path && is_passage[i] == true && is_passage_forbidden[i] == false) {
                        if (global_map[ INDEX_IN_MAP( DELTA_COOR_X( current_node, i ),
                                                      DELTA_COOR_Y( current_node, i ),
                                                      NODE_PHEROMONE_A_VAR)] > current_node_pheromones) {
                            trailing_itself = false;
                        }
                    }
                }*/                
                if (global_map[ INDEX_IN_MAP( DELTA_COOR_X( current_node, chosen_path ),
                                              DELTA_COOR_Y( current_node, chosen_path ),
                                              NODE_PHEROMONE_M_VAR) + global_id] == 0) {
                    trailing_itself = false;
                }
                if (trailing_itself == true) {
                    // The agent has reached an explored crossroads and its path contains more attractants, than
                    // the new paths. In order to prevent the agent from updating the attractant levels of nodes
                    // it just visited up to very high values, the agent will set the attractant levels to its own
                    // attractant level (instead of adding its level to node's level).
                    global_map[ INDEX_IN_MAP( current_node.x, current_node.y, NODE_PHEROMONE_A_VAR )] = new_boid_pheromone_level;
                }
            }
            if (trailing_itself == false) {
            // Agent consumes 1 attractant pheromone on each node it visits
                if (global_map[ INDEX_IN_MAP( current_node.x, current_node.y, NODE_PHEROMONE_A_VAR )] >= 1) {
                    float decrement = -1;
                    atomic_add_global_float(&global_map[ INDEX_IN_MAP( current_node.x,
                                                                       current_node.y,
                                                                       NODE_PHEROMONE_A_VAR )],
                                            &decrement);
                }
                else
                    // If node attractant level is below 1, set it to zero
                    global_map[ INDEX_IN_MAP( current_node.x, current_node.y, NODE_PHEROMONE_A_VAR )] = 0;
            }

            if (unexplored_passes_n > 1 &&
                global_map[ INDEX_IN_MAP( current_node.x, current_node.y, NODE_PHEROMONE_A_VAR )] < unexplored_passes_n - 1) {
                // Node attractant pheromone level must be always no less than number of unexplored passes it has
                // minus one, as agent will explore one. It will still be reduced by one if two agents
                // in the same node are targeting the same unexplored path.
                global_map[ INDEX_IN_MAP( current_node.x, current_node.y, NODE_PHEROMONE_A_VAR )] = unexplored_passes_n - 1;
            }

            if (trailing_itself == false) {
                // Add agent's own attractant level to the node's attractant level
                global_map[ INDEX_IN_MAP( current_node.x, current_node.y, NODE_PHEROMONE_A_VAR )] += new_boid_pheromone_level;

                // Update agent's attractant level:
                // If unexplored paths remain on current node, increase attractant level
                if (unexplored_passes_n > 1)
                    new_boid_pheromone_level += unexplored_passes_n - 1;
            }
        }
    }
    // Save new boid attractant level to global memory
    global_new_flock[ INDEX_IN_FLOCK( global_id, BOID_PHEROMONE_VAR)] = new_boid_pheromone_level;
}



void
update_node_status(__global float* global_map,
                   __global ushort* global_agent_programme,
                   uchar passes_n,
                   uchar unexplored_passes_n,
                   UCHARN current_node,
                   __global float* global_test) {
    if (global_agent_programme[PROG_LEAVING_DEADEND_VAR] == TRUE) {
        // The agent is on a path to a deadend and still on an unexplored track
        //if (unexplored_passes_n == 0 || passes_n < 2) {
            // If there is an unexplored square and it turns out to be a path, there is no need to set crossroads 
            // attractant level to 0. If it turns out to be a wall, agent will return back to the previous
            // square and set its attractant level to 0.
            // If there is less then 2 roads, it's definitely not crossroads.
            if (global_agent_programme[PROG_LEAVING_GOAL_VAR] == false)
                global_map[ INDEX_IN_MAP( current_node.x, current_node.y, NODE_IS_DEADEND_VAR )] = true;
            else
                global_map[ INDEX_IN_MAP( current_node.x, current_node.y, NODE_IS_GOAL_VAR )] = true;
        //}
    }
    
    if (global_agent_programme[PROG_EDGE_VAR] != GOAL_NOT_SET)
        // If agent has just entered the node
        // Update node exploration status
        if (global_map[ INDEX_IN_MAP(current_node.x,
                                     current_node.y,
                                     NODE_IS_EXPLORED_VAR)] == false)
            global_map[ INDEX_IN_MAP(current_node.x,
                                     current_node.y,
                                     NODE_IS_EXPLORED_VAR)] = true;
}



uchar
reverse_direction(uchar direction) {
    return (direction == NORTH ? SOUTH :
            (direction == EAST ? WEST :
             (direction == SOUTH ? NORTH :
              (direction == WEST ? EAST : 0))));
}