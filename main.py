"""
main.py is the top-level module which controls the simulator.
"""
from time import time
import os

import pygame as pg

import configs as cfg
import renderer
import displayer
import maze
import simulator
import opencl_computations as cl_comp
import agents
import experiment

# Set environment variables
# Work process-related settings
os.environ["PYOPENCL_NO_CACHE"] = "1"
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

swarm_sizes, maze_sizes, repeats, solver = experiment.get_tasks()
cfg.solver = solver

# Set timer
t1 = time()

# Initialize pygame
pg.init()

# Initialize opencl
context, device, queue = cl_comp.init_opencl()

# Generate or load maze
failed_to_load_maze = False
amaze = None
if cfg.reuse_maze:
    amaze = maze.load_from_pickle()
    if amaze is None:
        failed_to_load_maze = True
if not cfg.reuse_maze or failed_to_load_maze:
    amaze = maze.Maze()


template_triangles = renderer.render_template_triangles()

# Generate or load agents
failed_to_load_agents = False
flocks = None
if cfg.reuse_agents:
    flocks = agents.load_from_pickle(amaze)
    if flocks is None:
        failed_to_load_agents = True
if not cfg.reuse_agents or failed_to_load_agents:
    flocks = agents.generate_first_flock(amaze)

global_map, buffers, completion_time, solver =\
    simulator.run(context, device, queue, amaze, flocks, template_triangles)

t2 = time()
print("Calculations [s]:", t2 - t1)

if cfg.render_display_on:
    animation = renderer.render_animation(amaze, flocks, template_triangles, global_map)

    if cfg.output_video:
        displayer.savevideo(animation)
    if cfg.output_images:
        displayer.saveimages(animation)

t3 = time()
print("Rendering [s]:", t3 - t2)
print("Total [s]:", t3 - t1)
print("Done! See you next time.")
