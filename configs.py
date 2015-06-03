import numpy as np
import pygame as pg

" Main simulation settings "
NumberOfBoids       = 21
framespersecond     = 24  # 24
output_images       = False
output_video        = not output_images
maze_size           = maze_width, maze_height = 21, 21 # Unit squares (they have to be odd)
# Simulation length
startSecond         = 0
endSecond           = 10
# Pause at the end of the video
end_pause_dur       = (4 if output_images is False else 0)
# "GPU" = let agents solve the maze via propagation
# "CPU" = Dijkstra's algorithm will check if the maze is explored enough to be solved on each iteration
solver              = "CPU"
# Set reuse_data and save_data to False to randomize initial flock state and maze structure
reuse_data          = False
save_data           = False

" Additional simulation settings "
Dimensions                  = 2
seconds                     = endSecond
# Should be divisible by framespersecond
total_timesteps             = int(seconds * framespersecond)
debug_on                    = False
collision_detection_on      = True
# track_map_changes controls whether the map is to be
# transferred from GPU to CPU on each iteration.
# Must be true if solver == "True"
track_map_changes           = True
reuse_agents                = reuse_maze = reuse_data
save_agents                 = save_maze = save_data
# How many exotic obstacles per area
exotic_obstacle_density     = 0.1
tile_size                   = tile_width, tile_height = 30, 30 # Pixels
center                      = [maze_width / 2, maze_height / 2]
collision_check_step        = 0.1 # Unit squares

" Boids' behavior "
# Minimum separation for Reynolds rule of separation
MinSeparation                           = 1
WeightCenterOfMass                      = 0.03
WeightSeparation                        = 1
WeightAlignment                         = 0.125
WeightInertia                           = 0.4
boid_attractant_dissolution_component   = 0.01  # pheromones per square
node_attractant_dissolution_component   = 0.01 # pheromones per timestep
node_marker_dissolution_component       = 0.001 # pheromones per timestep
BoidVelocity                            = 0.25  # Squares per timestep
Traction                                = 0.1
READINGS_N                              = 5
NEURON_THRESHOLD                        = 1
# Weights unitialized value = assume all surrounding squares are passages
# For 8 neurons
WEIGHTS_UNINITIALIZED                   = 0b00001111
# Duration of time a random value is stored by the agent until CPU updates it
RAND_CLOCK_MAX                          = 10
# Agent is allowed to stall for some time to take into account an inertia.
MAX_STALL_TIME_ALLOWED                  = 2 # Timesteps
starting_node                           = (1, 0)
goal_node                               = (maze_width - 2, maze_height - 1)

" Rendering settings "
# Just generate the simulation or render it as well?
render_display_on           = True
bounding_rects_show         = False
window_size                 = window_width, window_height = maze_width * tile_width, maze_height * tile_height
# Boid triangle dimensions
triangle_rotation_res       = 1 # Degrees
middle_line_size            = 17 # Pixels
center_to_top_vertex        = 13 # Pixels
center_to_bottom_line       = middle_line_size - center_to_top_vertex
bottom_line_size            = 6 # Pixels
center_to_bottom_vertex     = np.hypot(center_to_bottom_line, bottom_line_size / 2)
# noinspection PyTypeChecker
central_bottom_half_angle   = np.arccos(center_to_bottom_line / center_to_bottom_vertex)
rect_diagonal_size          = np.hypot(bottom_line_size, middle_line_size)
# Graphics settings
maze_bg_color           = pg.Color(255, 255, 255)
maze_wall_color         = pg.Color(167, 175, 143)
maze_wall_bg_color      = pg.Color(218, 221, 208)
maze_passage_color      = pg.Color(211, 227, 228)
maze_unexplored_color   = pg.Color(63, 72, 204, 32)
maze_deadend_color      = pg.Color(184, 12, 16, 64)
maze_goal_color         = pg.Color(11, 185, 19, 128)
maze_boid_contour_color = pg.Color(107, 113, 6)
# maze_boid_fill_color    = pg.Color(231, 242, 13)
maze_boid_fill_color    = maze_boid_contour_color
maze_boid_oustide_color = pg.Color(0, 0, 255, 64)
maze_boid_inside_color  = pg.Color(255, 0, 0, 127)
label_face              = "monospace"
label_size              = 15
pheromone_level_face    = label_face
pheromone_level_size    = 12
boid_label_color        = pg.Color(0, 0, 0)
node_label_color        = pg.Color(127, 127, 127)

" Files settings "
cl_dir                  = "opencl"
cl_main_url             = "main.cl"
movie_url               = "opencl_video.mp4"
images_filename         = "image"
images_format           = ".png"
flock_pickle_url        = "agents.pkl"
maze_pickle_url         = "maze.pkl"
reporting_dir           = "reports"
collisions_csv          = "collisions.csv"
configurations_csv      = "configurations.csv"

" Array maps allow to store complex data structures in flat continuous arrays "
" Boid array map "
BOID_POS_VAR        = 0
BOID_VEL_VAR        = 1
ARRDIM              = 2
BOID_PHEROMONE_VAR  = 2
X                   = 0
Y                   = 1

" Programme array map "
ProgrammeSize               = READINGS_N + 3
PROG_SIZE                   = 15
PROG_EDGE_VAR               = 0
PROG_STALL_TIME_VAR         = 1
PROG_CHOSEN_PATH_VAR        = 2
PROG_ROTATED_BY_VAR         = 3
PROG_SWAMP_GOAL_VAR         = 4
PROG_GOAL_SQUARE_VAR        = 5
PROG_SRC_COORS_VAR          = 7
PROG_PREV_PATH_VAR          = 9
PROG_PREV_PREV_PATH_VAR     = 10
PROG_LEAVING_DEADEND_VAR    = 11
PROG_LEAVING_GOAL_VAR       = 12
PROG_RAND_VAR               = 13
PROG_RAND_CLOCK_VAR         = 14
# Goal status values
GOAL_NOT_SET                = 0
GOAL_EDGE_HORIZONTAL        = 1
GOAL_EDGE_VERTICAL          = 2

" Experiment array map "
EXP_SIZE                    = 1 + total_timesteps
EXP_BOID_HIT_COUNT_VAR      = 0
EXP_BOID_HITS_VAR           = 1

" Maze map array map "
NODE_SIZE               = 4 + NumberOfBoids
NODE_IS_EXPLORED_VAR    = 0
NODE_IS_DEADEND_VAR     = 1
NODE_IS_GOAL_VAR        = 2
NODE_PHEROMONE_A_VAR    = 3
NODE_PHEROMONE_M_VAR    = 4
