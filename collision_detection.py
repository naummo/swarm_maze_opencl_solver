"""
collision_detection.py is used on each iteration to detect whether
an agent has collided with walls and to provide an adequate environment
response (i.e. updated position & velocity such that agen slides along the wall).
"""

import numpy as np
import pygame as pg
from decimal import Decimal

import configs as cfg
import maze

x_var = cfg.X
y_var = cfg.Y
pos = cfg.BOID_POS_VAR * cfg.Dimensions
vel = cfg.BOID_VEL_VAR * cfg.Dimensions


class Amendments:
    """ Amendment data holder class """
    # Field indices in the packet generated by self.get_packet()
    amount_i = 0
    indices_i = 1
    values_i = 2

    def __init__(self):
        self.amount = 0
        self.indices = []
        self.values = []

    def get_packet(self):
        """ Returns all amendments in a packet format """
        return (np.uint16(self.amount),
                np.asarray(self.indices, dtype=np.uint16),
                np.asarray(self.values, dtype=np.float32))

    def clear(self):
        self.amount = 0
        self.indices = []
        self.values = []


def run(flock, previous_flock, amaze, template_triangles, amendments):
    """
        Detects collisions and calculates required amendments that
        allow boid to avoid collisions.
        For each boid it first checks if boid collides with the wall by rotating on the
        same spot. If it is, boid is moved out of the wall. If it isn't, the checking continues:
        it calculates its impulse (desired dislocation vector) and
        breaks it into steps. For each step (partial impulse) it checks if a wall
        is hit. If it is, boid slides along it. Multiple walls will be properly processed.

        TODO: Currently it's imprecise near the corners - there's a small transparent square
        on the corner of the wall with the size (cfg.collision_check_stop, cfg.collision_check_stop),
        and boid can go through it. Implementing proper processing may require more complex logic
        and is out of the scope of this project.
    """
    amendments.clear()
    i = 0
    for boid in flock.np_arrays:
        impulse = np.hypot(boid[vel + x_var], boid[vel + y_var])
        if impulse > 0:
            # We'll start from previous position and if no walls are hit,
            # increase it up to the new boid position
            boid[pos + x_var] = previous_flock.np_arrays[i][pos + x_var]
            boid[pos + y_var] = previous_flock.np_arrays[i][pos + y_var]

            template_triangle = template_triangles[min(
                int(np.round(np.degrees(flock.object_list[i].orientation))),
                359)]
            triangle_offset = template_triangle.get_triangle_top_left()

            triangle_rect = template_triangle.rect.copy()

            collision_detected = False

            # Fisrt check if the boid has collided into a wall without
            # moving (e.g. rotated near the wall)
            # ------------------------------------------------------
            hit_top, hit_right, hit_bottom, hit_left = \
                check_for_collision([boid[pos + x_var],
                                     boid[pos + y_var]],
                                    [boid[vel + x_var],
                                     boid[vel + y_var]],
                                    triangle_rect,
                                    triangle_offset,
                                    amaze)

            if hit_right or hit_left or hit_top or hit_bottom:
                collision_detected = True
                if cfg.bounding_rects_show:
                    flock.object_list[i].collided = True
                dx = dy = 0
                if hit_right:
                    wall_left_x = np.trunc(triangle_rect.right / cfg.tile_width) * cfg.tile_width
                    # dx will be negative
                    dx = wall_left_x - triangle_rect.right
                if hit_left:
                    wall_right_x = np.ceil(triangle_rect.left / cfg.tile_width) * cfg.tile_width
                    # dx will be positive
                    dx = wall_right_x - triangle_rect.left
                if hit_top:
                    wall_above_y = np.ceil(triangle_rect.top / cfg.tile_height) * cfg.tile_height
                    # dy will be positive
                    dy = wall_above_y - triangle_rect.top
                if hit_bottom:
                    wall_below_y = np.trunc(triangle_rect.bottom / cfg.tile_height) * cfg.tile_height
                    # dy will be negative
                    dy = wall_below_y - triangle_rect.bottom
                deltas_in_tiles = maze.to_unit_tiles(dx, dy)
                boid[pos + x_var] = boid[pos + x_var] + deltas_in_tiles[x_var]
                boid[pos + y_var] = boid[pos + y_var] + deltas_in_tiles[y_var]
                # Collision check for this boid is finished

            if not collision_detected:
                # First position is unobstructed, so check positions ahead
                # ------------------------------------------------------
                unit_impulse = cfg.collision_check_step
                # noinspection PyTypeChecker
                dx = boid[vel + x_var] * unit_impulse / impulse  # Unit squares
                # noinspection PyTypeChecker
                dy = boid[vel + y_var] * unit_impulse / impulse  # Unit squares
                number_of_checks = int(np.ceil(impulse / unit_impulse))
                for j in range(0, number_of_checks):
                    if (j + 1) * unit_impulse > impulse:  # Last step can be smaller
                        # Using Decimal here as float != float - 0 and Decimal is exact.
                        # Python uses approximate values and it negatively manifests itself here.
                        unit_impulse = np.float32(Decimal(impulse - unit_impulse * j))
                        dx = boid[vel + x_var] * unit_impulse / impulse  # Unit squares
                        dy = boid[vel + y_var] * unit_impulse / impulse  # Unit squares

                    hit_top, hit_right, hit_bottom, hit_left = \
                        check_for_collision([boid[pos + x_var] + dx,
                                             boid[pos + y_var] + dy],
                                            [boid[vel + x_var],
                                             boid[vel + y_var]],
                                            triangle_rect,
                                            triangle_offset,
                                            amaze)
                    if hit_right or hit_left or hit_top or hit_bottom:
                        collision_detected = True
                        if cfg.bounding_rects_show:
                            flock.object_list[i].collided = True
                        # Nullify impulse if a wall is on the way
                        if (dx > 0 and hit_right) or (dx < 0 and hit_left):
                            dx = 0
                        if (dy > 0 and hit_bottom) or (dy < 0 and hit_top):
                            dy = 0
                        if dx == 0 and dy == 0:
                            # Can't proceed
                            break

                    if not maze.outside_maze(boid[pos + x_var] + dx,
                                             boid[pos + y_var] + dy):
                        # The boid was moved outside the maze
                        # Apply amendments to the host data according to the type of collision
                        # I.e. slide along the wall
                        boid[pos + x_var] = boid[pos + x_var] + dx
                        boid[pos + y_var] = boid[pos + y_var] + dy
                    else:
                        # Boid is outside the maze, no point continuing the check
                        break

            if collision_detected:
                # Save amendments to transfer them later to the GPU
                amendments.values.append(np.copy([boid[pos + x_var],
                                                  boid[pos + y_var]]))
                amendments.indices.append(i)
                amendments.amount += 1
        i += 1


def check_for_collision(boid_center, boid_impulse, triangle_rect, triangle_offset, amaze):
    """ Returns collision types (left, right, top, bottom) """
    triangle_rect_coors = maze.to_coors(
        boid_center[x_var],
        boid_center[y_var])
    triangle_rect.left = triangle_rect_coors[x_var] + triangle_offset[x_var]
    triangle_rect.top = triangle_rect_coors[y_var] + triangle_offset[y_var]

    # Get new neighboring walls as a list of coordinate pairs
    neighboring_walls = \
        maze.get_neighboring_tiles(boid_center[x_var], boid_center[y_var],
                                   amaze, maze.Wall, include_none=False)

    # Convert coordinates into rects
    neighboring_walls_rects = []
    for wall in neighboring_walls:
        neighboring_walls_rects.append(
            pg.Rect(wall[x_var] * cfg.tile_width, wall[y_var] * cfg.tile_height,
                    cfg.tile_width, cfg.tile_height))

    # Check if triangle collides with any of them
    colliding_walls = triangle_rect.collidelistall(neighboring_walls_rects)

    hit_top = hit_bottom = hit_left = hit_right = False
    diagonal_collision = None
    if colliding_walls:
        # Collision detected
        for wall_i in colliding_walls:
            # Get collision type (horizontal/vertical)
            collision_types = get_collision_type(neighboring_walls[wall_i][x_var],
                                                 neighboring_walls[wall_i][y_var],
                                                 maze.to_unit_tiles(triangle_rect.centerx,
                                                                    triangle_rect.centery),
                                                 triangle_rect)
            if collision_types[0] == maze.Orientation.diagonal:
                diagonal_collision = collision_types[1:]
            else:
                for collision_type in collision_types:
                    if collision_type == maze.Location.top:
                        hit_top = True
                    if collision_type == maze.Location.bottom:
                        hit_bottom = True
                    if collision_type == maze.Location.left:
                        hit_left = True
                    if collision_type == maze.Location.right:
                        hit_right = True
        if diagonal_collision is not None:
            if not (hit_top or hit_bottom or hit_left or hit_right):
                # If boid has collided only with a diagonal wall, then alter
                # its velocity, otherwise ignore it.
                if diagonal_collision == [maze.Location.left, maze.Location.bottom]:
                    if np.abs(boid_impulse[y_var]) > np.abs(boid_impulse[x_var]):
                        hit_left = True
                    else:
                        hit_bottom = True
                if diagonal_collision == [maze.Location.right, maze.Location.top]:
                    if np.abs(boid_impulse[y_var]) > np.abs(boid_impulse[x_var]):
                        hit_right = True
                    else:
                        hit_top = True
                if diagonal_collision == [maze.Location.right, maze.Location.bottom]:
                    if np.abs(boid_impulse[y_var]) > np.abs(boid_impulse[x_var]):
                        hit_right = True
                    else:
                        hit_bottom = True
    return hit_top, hit_right, hit_bottom, hit_left


def get_collision_type(wall_x_float, wall_y_float, boid_pos_float, triangle_rect):
    """
        Returns thetype of collision (horizontal/vertical).
        C H C
        V b V
        C H C
        (H - horizontal, V - vertical, C - corner, b - boid previous position)
    """
    wall_x = int(wall_x_float)
    wall_y = int(wall_y_float)
    boid_x = int(boid_pos_float[x_var])
    boid_y = int(boid_pos_float[y_var])
    if wall_x != boid_x and wall_y != boid_y:
        # Corner wall
        return get_diagonal_collision_type(wall_x, wall_y, [boid_x, boid_y], triangle_rect)
    if wall_y != boid_y:
        # Horizontal wall
        if wall_y < boid_y:
            return [maze.Location.top, ]
        else:
            return [maze.Location.bottom, ]
    # Vertical wall
    if wall_x > boid_x:
        return [maze.Location.right, ]
    else:
        return [maze.Location.left, ]


def get_diagonal_collision_type(wall_x, wall_y, boid_center, triangle_rect):
    """ Checks with which side of the diagonally positioned (not oriented) wall boid has collided """
    # Get wall type
    diagonal_wall_position = 0
    if wall_x == np.trunc(boid_center[x_var]) - 1:
        """ T F F
            F F F
            T F F
            (one of the "True" walls) """
        if wall_y == np.trunc(boid_center[y_var]) - 1:
            diagonal_wall_position = (maze.Location.left, maze.Location.top)
        else:
            diagonal_wall_position = (maze.Location.left, maze.Location.bottom)

    if wall_x == np.trunc(boid_center[x_var]) + 1:
        """ F F T
            F F F
            F F T
            (one of the "True" walls) """
        if wall_y == np.trunc(boid_center[y_var]) - 1:
            diagonal_wall_position = (maze.Location.right, maze.Location.top)
        else:
            diagonal_wall_position = (maze.Location.right, maze.Location.bottom)
    wall_left, wall_top = maze.to_coors(wall_x,
                                        wall_y)
    wall_right, wall_bottom = maze.to_coors(wall_x + 1,
                                            wall_y + 1)

    precision_x = cfg.collision_check_step * cfg.window_width
    precision_y = cfg.collision_check_step * cfg.window_height
    # Get collision type
    wall_on_left = None
    wall_on_right = None
    wall_above = None
    wall_below = None

    if diagonal_wall_position[1] == maze.Location.top and triangle_rect.top >= wall_top - precision_y:
        wall_above = True
    if diagonal_wall_position[1] == maze.Location.bottom and triangle_rect.bottom <= wall_top + precision_y:
        wall_below = True

    if diagonal_wall_position[0] == maze.Location.right:
        # One of the walls on right from the boid's position
        if triangle_rect.right <= wall_left + precision_x:
            # Boid is at least on the left edge of the wall
            wall_on_right = True

        if wall_on_right and (wall_above or wall_below):
            # Boid is on both edges of the wall, i.e. on its corner
            return [maze.Orientation.diagonal, maze.Location.right, diagonal_wall_position[1]]
        if wall_on_right:
            # Bois is only on the left edge of the wall
            return [maze.Orientation.diagonal, maze.Location.right]
    else:  # diagonal_wall_position[0] == maze.Location.left
        # One of the walls on left from the boid's position
        if triangle_rect.left >= wall_right - precision_x:
            # Boid is at least on the right edge of the wall
            wall_on_left = True

        if wall_on_left and (wall_above or wall_below):
            # Boid is on both edges of the wall, i.e. on its corner
            return [maze.Orientation.diagonal, maze.Location.left, diagonal_wall_position[1]]
        if wall_on_right:
            # Bois is only on the right edge of the wall
            return [maze.Orientation.diagonal, maze.Location.left]

    if wall_above or wall_below:
        return [maze.Orientation.diagonal, diagonal_wall_position[1]]
