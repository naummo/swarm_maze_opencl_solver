#include "utilities.h"

/* Prototypes */
FLOATN limit_velocity(FLOATN velocity, float max_velocity);

void
agents_predator_ai_preprocess(__global float* global_predator_attack_vector/*[NUMBER_OF_PREDATORS]*/,
                              __local float* local_predator_attack_vector/*[NUMBER_OF_PREDATORS]*/,
                              __global float* global_flock_pos_sum,
                              __local float* local_flock_pos_sum,
                              __global float* previous_flock,
                              __global float* previous_predators,
                              int global_id,
                              int local_id) {
    int i;
    FLOATN difference;
    FLOATN velocity;
    // Predator RULE 1.  Cohesion - Steer to move towards the center of mass
    // Create partial (for <group_size> boids) sum in local memory
    {
        float pos[DIMENSIONS];
        fetch_global(&previous_flock[BOID_INDEX(global_id,POS)], pos, DIMENSIONS);
        atomic_add_local(local_flock_pos_sum, pos);
    }

    // Predator RULE 3. Attack boids within range
    // Each workitem changes all 5 predator velocities slightly
    // (in local memory)
    for (i = 0; i < NUMBER_OF_PREDATORS; i++) {
        difference = VLOADN(&previous_predators[BOID_INDEX(i,POS)]) - 
                            VLOADN(&previous_flock[BOID_INDEX(global_id,POS)]);
        if (length(difference) < PREDATOR_SIGHT) {
            velocity = normalize(difference);
            // Multiple work items are writing in the same array element, so
            // first we work on it in local memory, and then
            // transfer it once into global memory
            atomic_add_local_v(&local_predator_attack_vector[VAR_INDEX(i)], &velocity);
        }
    }

    // Exit fetch (local -> global) 
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        // Rule 1
        // Add partial sums up to full sum in global memory
        float flock_pos_sum[DIMENSIONS];
        fetch_local(local_flock_pos_sum, flock_pos_sum, DIMENSIONS);
        atomic_add_global(global_flock_pos_sum, flock_pos_sum);

        // Rule 3
        // Add local velocity to global velocity
        float attack_vector[DIMENSIONS];
        for (i = 0; i < NUMBER_OF_PREDATORS; i++) {
            fetch_local(&local_predator_attack_vector[VAR_INDEX(i)], attack_vector, DIMENSIONS);
            atomic_add_global(&global_predator_attack_vector[VAR_INDEX(i)], 
                              attack_vector);
        }
    }
    /*
    // TEST
    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_id == 0) {
        global_test[0] = global_flock_pos_sum[0];
        global_test[1] = global_flock_pos_sum[1];
        global_test[2] = global_flock_pos_sum[2];
    }*/
}



void
agents_predator_ai(__global float* global_generated_predators/*[TOTAL_TIMESTEPS][NUMBER_OF_PREDATORS][ARRDIM]*/,
                   __global float* global_predator_attack_vector/*[NUMBER_OF_PREDATORS]*/,
                   __local float* local_flock_pos_sum,
                   __global float* previous_predators,
                   __local ushort* local_iteration,
                   int global_id,
                   int local_id) {
    int i;
    FLOATN predator_velocity;

    //Rule 1: Cohesion - Steer to move towards the center of mass
    // Center of mass vector was computed during preprocessing
    predator_velocity = (VLOADN(local_flock_pos_sum) / NUMBER_OF_BOIDS - VLOADN(&previous_predators[BOID_INDEX(local_id,POS)])) *
                                 WEIGHTCENTEROFMASSP;

    //Rule 3: Attack boids within range
    // Predator attack vector was computed during preprocessing
    predator_velocity = predator_velocity + VLOADN(&global_predator_attack_vector[VAR_INDEX(global_id)]) *
                                            WEIGHTATTACKBOID;

    {
        float t;
        float predator_velocity_a[DIMENSIONS];
        float equations[3];

        // Rule 4: Move the predator around in smooth way around the center of the cube
        // http://en.wikipedia.org/wiki/Trefoil_knot
        t = ((float)*local_iteration / (float)TOTAL_TIMESTEPS) * 4.0 * M_PI + (float)global_id * (M_PI / 4.0);
        equations[0] = (2.0 + cos(3.0 * t)) * cos(2.0 * t);
        equations[1] = (2.0 + cos(3.0 * t)) * sin(2.0 * t);
        equations[2] = sin(3.0 * t);
        for (i = 0; i < DIMENSIONS; i++)
            predator_velocity_a[i] = equations[i];

        predator_velocity = predator_velocity + VLOADN(predator_velocity_a) * WEIGHTKNOT;
    }
    // Introduce traction
    predator_velocity = predator_velocity * (1 - TRACTION);

    // Introduce inertia
    predator_velocity = VLOADN(&previous_predators[BOID_INDEX(global_id,VEL)]) * WEIGHTINERTIA +
                        predator_velocity * (1 - WEIGHTINERTIA);

    // Limit velocity
    predator_velocity = limit_velocity(predator_velocity, MAXVELOCITYP);

    // Update positon
    VSTOREN(VLOADN(&previous_predators[BOID_INDEX(local_id,POS)]) + predator_velocity, 
        &global_generated_predators[FLOCK_INDEX(NUMBER_OF_PREDATORS,*local_iteration,local_id,POS)]);

    // Update velocity
    VSTOREN(predator_velocity, &global_generated_predators[FLOCK_INDEX(NUMBER_OF_PREDATORS,*local_iteration,local_id,VEL)]);
/*
    // TEST
    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_id == 1) {
        //VSTOREN(predator_velocity, global_test);
        //global_test[0] = distance;
        //global_test[1] = main_boid_id;
        //global_test[2] = compared_boid_id;
    }*/
}



void
agents_boid_ai_preprocess(__global float* global_boid_avoidance_vector,
                          __global float* global_flock_vel_sum,
                          __local float* local_flock_vel_sum,
                          __global float* previous_flock,
                          int compared_boid_id,
                          int main_boid_id,
                          int local_id) {
    float adistance;
    FLOATN difference;
    FLOATN boid_velocity;

    // Rule 2
    difference = VLOADN(&previous_flock[BOID_INDEX(compared_boid_id,POS)]) - 
                        VLOADN(&previous_flock[BOID_INDEX(main_boid_id,POS)]);
    adistance = length(difference);
    if (adistance > 0 && adistance < MINSEPARATION) {
        boid_velocity = normalize(difference) / adistance; // Why second time? Who knows.
        atomic_add_global_v(&global_boid_avoidance_vector[VAR_INDEX(main_boid_id)], &boid_velocity);
    }

    float vel[DIMENSIONS];
    // Rule 3
    if (compared_boid_id == 0 || (compared_boid_id == 1 && main_boid_id == 0)) {
        fetch_global(&previous_flock[BOID_INDEX(main_boid_id,VEL)], vel, DIMENSIONS);
        // Create partial (for <group_size> boids) sum in local memory
        atomic_add_local(local_flock_vel_sum, vel);
    }

    /* Exit fetch (local -> global) */
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == get_local_size(0) - 1) {
        float flock_vel_sum[DIMENSIONS];
        fetch_local(local_flock_vel_sum, flock_vel_sum, DIMENSIONS);
        // Rule 3
        // Add partial sums up to full sum in global memory
        atomic_add_global(global_flock_vel_sum, flock_vel_sum);
    }
    
    // TEST
    /*barrier(CLK_LOCAL_MEM_FENCE);
    if (global_id == 1) {
        //VSTOREN(VLOADN(local_flock_vel_sum), global_test);
        //global_test[0] = distance;
        //global_test[1] = main_boid_id;
        //global_test[2] = compared_boid_id;
    }*/
    
}



void
agent_ai(__global float* global_generated_flocks/*[TOTAL_TIMESTEPS][NUMBER_OF_BOIDS][ARRDIM]*/,
         __local float* local_flock_pos_sum,
         __local float* local_flock_vel_sum,
         __local ushort* local_iteration,
         float boid_avoidance_vector[DIMENSIONS],
         __global float* previous_flock,
         __global float* previous_predators,
         FLOATN* impulse,
         int global_id,
         int local_id) {
    int i;
    FLOATN difference;
    
    // Boid RULE 1. Cohesion - Steer to move towards the center of mass
    // Center of mass vector was computed during predator preprocessing
    *impulse = (VLOADN(local_flock_pos_sum) / NUMBER_OF_BOIDS - VLOADN(&previous_flock[BOID_INDEX(global_id,POS)])) *
                    WEIGHTCENTEROFMASS;
    // Boid RULE 2. Separation - steer to avoid crowding local flockmates
    // Avoidance vectors were calculated during boid preprocessing
    // Done
    *impulse = *impulse - VLOADN(boid_avoidance_vector) * WEIGHTSEPARATION; // subtracting on purpose
    // FLOATN test2;
    // test2 = VLOADN(&global_boid_avoidance_vector[VAR_INDEX(global_id)]);

    // Boid RULE 3. Alignment - Steer towards the average heading of local flockmates
    // Compute center of velocity during preprocessing 
    *impulse = *impulse + (VLOADN(local_flock_vel_sum) / NUMBER_OF_BOIDS -
        VLOADN(&previous_flock[BOID_INDEX(global_id,VEL)])) * WEIGHTALIGNMENT;

    // Boid RULE 4. Try to move towards the center of the grid (just for fun)
    *impulse = *impulse + (CENTER - VLOADN(&previous_flock[BOID_INDEX(global_id,POS)])) * WEIGHTCENTER;

    // Boid RULE 5. Add some randomness
    // AWOOGA! Skip for now - hard to get random using openCL.
    // I might transfer a number from python.

    // Boid RULE 6. Avoid the predator
    for (i = 0; i < NUMBER_OF_PREDATORS; i++) {
        difference = VLOADN(&previous_predators[BOID_INDEX(i,POS)]) -
            VLOADN(&previous_flock[BOID_INDEX(global_id,POS)]);
        if (length(difference) < PREDATORRADIUS)
            *impulse = *impulse - difference * WEIGHTAVOIDPREDATOR;
    }

    // Boid last RULE. Limit velocity
    *impulse = limit_velocity(*impulse, MAXVELOCITY);

    //TEST
    /*
    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_id == 0) {
        //float test[DIMENSIONS];
        //VSTOREN(VLOADN(global_flock_vel_sum), global_test);
        //global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS,*local_iteration,global_id,POS)] = *local_flock_pos_sum;
        //global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS,*local_iteration,global_id,POS)+1] = test[1];
        //global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS,*local_iteration,global_id,POS)+2] = test[2];
        //global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS,*local_iteration,global_id,VEL)] = global_boid_avoidance_vector[VAR_INDEX(global_id)];
        //global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS,*local_iteration,global_id,VEL)+1] = global_boid_avoidance_vector[VAR_INDEX(global_id)+1];
        //global_generated_flocks[FLOCK_INDEX(NUMBER_OF_BOIDS,*local_iteration,global_id,VEL)+2] = global_boid_avoidance_vector[VAR_INDEX(global_id)+2];
    }
    */
}

FLOATN limit_velocity(FLOATN velocity, float max_velocity) {
    /* Limiting the speed to avoid unphysical jerks in motion */
    int velocity_length = length(velocity);
    if (velocity_length > max_velocity)
        return velocity * max_velocity / velocity_length;
    return velocity;
}
