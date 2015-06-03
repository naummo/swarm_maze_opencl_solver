/* Prototypes */
void
agent_reynolds_rules13_preprocess(__local float* local_flock_pos_sum,
                                  __local float* local_flock_vel_sum,
                                  float* previous_position,
                                  float* previous_velocity);
void
agent_reynolds_rule2_preprocess(__global float* previous_flock,
                                __global float* global_boid_avoidance_vectors,
                                int main_boid_id,
                                int compared_boid_id);
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
         __global float* global_test);
void
preprocess_for_cmm(uchar* sensor_readings,
                   uchar* rotated_by,
                   uint* trigram_ids,
                   __global float* global_test);
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
           __global float* global_test);
uchar
in_proximity(float what, float of);
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
          __global float* global_test);
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
                __global float* global_test);
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
               __global float* global_test);
void
get_impulse(float* previous_boid,
            uchar chosen_path,
            FLOATN* impulse);
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
                 __global float* global_test);
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
                        __global float* global_test);
void
update_node_status(__global float* global_map,
                   __global ushort* global_agent_programme,
                   uchar passes_n,
                   uchar unexplored_passes_n,
                   UCHARN current_node,
                   __global float* global_test);
uchar
reverse_direction(uchar direction);