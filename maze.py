"""
maze.py contains all maze-related data structures and functions.
"""

import os
import pickle
import random
from enum import Enum

import numpy as np

import configs as cfg


class Square():
    square_types_n = 12
    square_orientation_n = 4

    def __init__(self, shape, orientation):
        self.shape = shape
        self.orientation = orientation

    def get_numeric_value(self):
        """
        Converts shape & orientation pair into a unique
        numeric value.
        """
        return self.shape.value * self.square_orientation_n + self.orientation.value


class Passage(Enum):
    """ Contains passage type symbols. Values must be unique
    in the scope Passage + Wall.
    """
    normal = 0
    striped = 1
    crossed = 2
    tiled = 3
    dotted = 4
    entrance = 5
    exit = 6


class PassageShapes:
    def __init__(self):
        self.shapes = []
        # Square contour
        contour = [("rect", [0, 0, cfg.tile_width, 1]),  # top edge
                   ("rect", [0, cfg.tile_height - 1,
                             cfg.tile_width - 1, 1]),  # bottom edge
                   ("rect", [0, 1,
                             1, cfg.tile_height - 1]),  # left edge
                   ("rect", [cfg.tile_width - 1, 1,
                             1, cfg.tile_height - 1])]  # right edge
        # striped
        self.shapes.append(contour + [
            ("rect", [1, int((cfg.tile_height - 2) / 4) + 1,
                      cfg.tile_width - 2, int((cfg.tile_height - 2) / 4)]),
            ("rect", [1, int(3 * (cfg.tile_height - 2) / 4) + 2,
                      cfg.tile_width - 2, int((cfg.tile_height - 2) / 4) - 1])])
        # crossed
        self.shapes.append(contour + [
            ("polygon", [(1, 1), (cfg.tile_width - 2, 1),
                         (int(cfg.tile_width / 2), int(cfg.tile_height / 2))]),
            ("polygon", [(1, cfg.tile_height - 2),
                         (cfg.tile_width - 2, cfg.tile_height - 2),
                         (int(cfg.tile_width / 2), int(cfg.tile_height / 2))])])
        # tiled
        self.shapes.append(contour + [
            ("rect", [1, 1, int((cfg.tile_width - 2) / 2),
                      int((cfg.tile_height - 2) / 2)]),
            ("rect", [int((cfg.tile_width - 2) / 2) + 1,
                      int((cfg.tile_height - 2) / 2) + 1,
                      int((cfg.tile_width - 2) / 2),
                      int((cfg.tile_height - 2) / 2)])])
        # dotted
        self.shapes.append(contour + [
            ("circle", [int(cfg.tile_width / 2), int(cfg.tile_height / 2),
                        int(min(cfg.tile_width, cfg.tile_height) / 4)])])
        # entrance
        self.shapes.append(None)
        # exit
        self.shapes.append(None)

    def get_shapes(self, wall_type):
        # Exotic passage indices start from 1. Their shapes,
        # however, have indices starting from 0
        return self.shapes[wall_type.value - 1]


passage_shapes = PassageShapes()


class Wall(Enum):
    """ Contains wall type symbols. Values must be unique
    in the scope Passage + Wall.
    """
    normal = 7
    small_triangular = 8
    big_triangular = 9
    forward_slash = 10
    backward_slash = 11


class WallShapes:
    def __init__(self):
        self.shapes = []
        # small_triangular
        # TODO: fix misplacement of one of the lines
        self.shapes.append([
            ("polygon",
             [(0, 0), (cfg.tile_width / 2, cfg.tile_height / 2),
              (cfg.tile_width, 0),
              (cfg.tile_width, cfg.tile_height),
              (0, cfg.tile_height)])])
        # big_triangular
        self.shapes.append([
            ("polygon",
             [(0, 0), (cfg.tile_width / 2, cfg.tile_height),
              (cfg.tile_width, 0),
              (cfg.tile_width, cfg.tile_height),
              (0, cfg.tile_height)])])
        # forward_slash
        self.shapes.append([
            ("polygon",
             [(0, 0), (cfg.tile_width, cfg.tile_height),
              (0, cfg.tile_height)])])
        # backward_slash
        self.shapes.append([
            ("polygon",
             [(0, cfg.tile_height), (cfg.tile_width, 0),
              (cfg.tile_width, cfg.tile_height)])])

    def get_shapes(self, wall_type):
        # Exotic wall indices start after all Passage indices and
        # normal wall index. Their tuples, however, have indices
        # starting from 0
        return self.shapes[wall_type.value - len(Passage) - 1]


wall_shapes = WallShapes()


class Orientation(Enum):
    """ Orientation types. First four symbols must have
    values from 0 to 3 (look Square.get_numeric_value()).
    """
    north = 0
    east = 1
    south = 2
    west = 3
    horizontal = 4
    vertical = 5
    diagonal = 6


class Location(Enum):
    """ Is usually used just as enum, but values also
    correspond to the tile ordering in the list returned
    by get_neighboring_tiles().
    """
    top = 1
    right = 4
    bottom = 6
    left = 3
    center = 8


class Maze:
    """ Stores maze matrix """

    def __init__(self):
        """ Creates maze both as a matrix and a list of Rects """
        print("Generating a maze.")
        self.matrix = generate_matrix()
        # Upper left quarter
        self.introduce_exotic_obstacles(cfg.exotic_obstacle_density * cfg.maze_width *
                                        cfg.maze_height / 4,
                                        0, cfg.maze_width / 2,
                                        0, cfg.maze_height / 2,
                                        [Wall.small_triangular, ],
                                        [Passage.striped, ])
        # Upper right quarter
        self.introduce_exotic_obstacles(cfg.exotic_obstacle_density * cfg.maze_width *
                                        cfg.maze_height / 4,
                                        cfg.maze_width / 2, cfg.maze_width,
                                        0, cfg.maze_height / 2,
                                        [Wall.big_triangular, ],
                                        [Passage.crossed, ])
        # Lower left quarter
        self.introduce_exotic_obstacles(cfg.exotic_obstacle_density * cfg.maze_width *
                                        cfg.maze_height / 4,
                                        0, cfg.maze_width / 2,
                                        cfg.maze_height / 2, cfg.maze_height,
                                        [Wall.forward_slash, ],
                                        [Passage.tiled, ])
        # Lower right quarter
        self.introduce_exotic_obstacles(cfg.exotic_obstacle_density * cfg.maze_width *
                                        cfg.maze_height / 4,
                                        cfg.maze_width / 2, cfg.maze_width,
                                        cfg.maze_height / 2, cfg.maze_height,
                                        [Wall.backward_slash, ],
                                        [Passage.dotted, ])

        if cfg.save_maze:
            self.pickle_maze()

    def pickle_maze(self):
        """ Pickles the maze into a file """
        print("Pickling a maze.")
        try:
            pickle_f = open(os.path.join(os.getcwd(), cfg.maze_pickle_url), "wb")
            pickle.dump(self, pickle_f)
            pickle_f.close()
        except IOError:
            print("Failed to save pickle into", os.path.join(os.getcwd(), cfg.maze_pickle_url))

    def introduce_exotic_obstacles(self, n_obstacles, x_min, x_max, y_min, y_max, wall_list, passage_list):
        """ Changes types of walls and passages randomly within specific areas """
        # Range is a quarter of maze surface multipled by dancity
        # for _ in range(cfg.exotic_obstacle_density * cfg.maze_width * cfg.maze_height / 4):
        for _ in range(int(n_obstacles)):
            [x, y] = np.random.rand(2)
            x = int(x_min + np.trunc(x * (x_max - x_min)))
            y = int(y_min + np.trunc(y * (y_max - y_min)))

            while self.matrix[x][y].shape != Wall.normal:
                [x, y] = np.random.rand(2)
                x = int(x_min + np.trunc(x * (x_max - x_min)))
                y = int(y_min + np.trunc(y * (y_max - y_min)))

            # Random wall (but not normal) and random orientation
            self.matrix[x][y] = Square(random.choice(wall_list), random.choice(list(Orientation)[0:4]))

            while self.matrix[x][y].shape != Passage.normal:
                [x, y] = np.random.rand(2)
                x = int(x_min + np.trunc(x * (x_max - x_min)))
                y = int(y_min + np.trunc(y * (y_max - y_min)))

            # Random passage(but not normal) and random orientation
            self.matrix[x][y] = Square(random.choice(passage_list), random.choice(list(Orientation)[0:4]))


def generate_matrix():
    """ Creates maze matrix (Randomized Prim's algorithm) """
    # 1. Start with a grid full of walls.
    matrix = []
    for x in range(cfg.maze_width):
        matrix.append([])
        for y in range(cfg.maze_height):
            matrix[x].append(Square(Wall.normal, Orientation.north))

    # 2. Pick a cell, mark it as part of the maze. Add the walls of the cell to the wall list.
    starting_cell = 1, 1
    walls = []
    matrix[starting_cell[0]][starting_cell[1]] = \
        Square(Passage.normal, Orientation.north)
    walls.extend(get_walls(*starting_cell, matrix=matrix))
    # 3. While there are walls in the list:
    while len(walls) > 0:
        # 3.1 Pick a random wall from the list. If the cell on the opposite side isn't in the maze yet:
        wall_id = int(np.random.rand(1) * len(walls))
        wall = walls[wall_id]
        if not is_edge_wall(*wall):
            opposite_cell = get_opposite_cell(*wall, matrix=matrix)
            if isinstance(matrix[opposite_cell[0]][opposite_cell[1]].shape, Wall):
                # 3.1.1 Make the wall a passage and mark the cell on the opposite side as part of the maze.
                matrix[wall[0]][wall[1]].shape = Passage.normal
                matrix[opposite_cell[0]][opposite_cell[1]].shape = Passage.normal
                # 3.1.2 Add the neighboring walls of the cell to the wall list.
                walls.extend(get_walls(*opposite_cell, matrix=matrix))
        # 3.2 Remove the wall from the list.
        walls.pop(wall_id)
    # Add exits
    matrix[cfg.starting_node[0]][cfg.starting_node[1]].shape = Passage.entrance
    matrix[cfg.goal_node[0]][cfg.goal_node[1]].shape = Passage.exit

    return matrix


def load_from_pickle():
    """ Loads previously generated maze from a pickle file """
    print("Unpickling a maze.")
    try:
        pickle_f = open(os.path.join(os.getcwd(), cfg.maze_pickle_url), "rb")
        amaze = pickle.load(pickle_f)
        pickle_f.close()
        if (len(amaze.matrix) != cfg.maze_width or
           len(amaze.matrix[0]) != cfg.maze_height):
            print("The pickled maze has a wrong size.")
            return None
        return amaze
    except (IOError, EOFError):
        print("Failed to load the pickle", os.path.join(os.getcwd(), cfg.maze_pickle_url))
        return None


def get_opposite_cell(x, y, orientation, matrix):
    """ Get coordinates of the cell opposite to the wall """
    if orientation == Orientation.horizontal:
        if isinstance(matrix[x - 1][y].shape, Wall):
            return x - 1, y
        else:
            return x + 1, y
    if orientation == Orientation.vertical:
        if isinstance(matrix[x][y - 1].shape, Wall):
            return x, y - 1
        else:
            return x, y + 1


def get_walls(x, y, matrix):
    """ Get coordinates of walls of the cell """
    result = []
    if x > 0:
        if matrix[x - 1][y].shape == Wall.normal:
            result.append((x - 1, y, Orientation.horizontal))
    if y > 0:
        if matrix[x][y - 1].shape == Wall.normal:
            result.append((x, y - 1, Orientation.vertical))
    if y < cfg.maze_height - 1:
        if matrix[x][y + 1].shape == Wall.normal:
            result.append((x, y + 1, Orientation.vertical))
    if x < cfg.maze_width - 1:
        if matrix[x + 1][y].shape == Wall.normal:
            result.append((x + 1, y, Orientation.horizontal))
    return result


def is_edge_wall(x, y, orientation):
    """ Returns boolean indicating if the wall is on the edge """
    if orientation == Orientation.horizontal and (x == 0 or x == cfg.maze_width - 1):
        return True
    if orientation == Orientation.vertical and (y == 0 or y == cfg.maze_height - 1):
        return True
    return False


def to_coors(unit_tile_x, unit_tile_y):
    return [int(unit_tile_x * cfg.tile_width),
            int(unit_tile_y * cfg.tile_height)]


def to_unit_tiles(px_x, px_y):
    return [px_x / cfg.tile_width,
            px_y / cfg.tile_height]


def wall_dimensions(i, j, width, height):
    """ Returns wall dimensions according to wall dpi-s """
    return (i * cfg.tile_width, j * cfg.tile_height,
            width * cfg.tile_width, height * cfg.tile_height)


def get_neighboring_tiles(maze_x, maze_y, amaze, afilter, include_none=True):
    """
    Returns a list of coordinate pairs of neighbour tiles which
    match the afilter.
    Inputs are in unit tiles.
    """
    check_class = check_instance = False
    if afilter == Wall or afilter == Passage:
        check_class = True
    else:
        if isinstance(afilter, Wall) or \
                isinstance(afilter, Passage):
            check_instance = False

    x = int(maze_x)
    y = int(maze_y)
    tiles = []
    for i in range(9):
        tiles.append(None)
    if y - 1 >= 0:
        if x - 1 >= 0:
            # Top left
            if typeless_equals(amaze.matrix[x - 1][y - 1].shape,
                               afilter, check_class, check_instance):
                tiles[0] = [x - 1, y - 1]
        # Top center
        if typeless_equals(amaze.matrix[x][y - 1].shape,
                           afilter, check_class, check_instance):
            tiles[1] = [x, y - 1]
        if x + 1 <= cfg.maze_width - 1:
            # Top right
            if typeless_equals(amaze.matrix[x + 1][y - 1].shape,
                               afilter, check_class, check_instance):
                tiles[2] = [x + 1, y - 1]
    # Middle left
    if x - 1 >= 0:
        if typeless_equals(amaze.matrix[x - 1][y].shape,
                           afilter, check_class, check_instance):
            tiles[3] = [x - 1, y]
    # Middle right
    if x + 1 <= cfg.maze_width - 1:
        if typeless_equals(amaze.matrix[x + 1][y].shape,
                           afilter, check_class, check_instance):
            tiles[4] = [x + 1, y]

    if y + 1 <= cfg.maze_height - 1:
        if x - 1 >= 0:
            # Bottom left
            if typeless_equals(amaze.matrix[x - 1][y + 1].shape,
                               afilter, check_class, check_instance):
                tiles[5] = [x - 1, y + 1]
        # Bottom center
        if typeless_equals(amaze.matrix[x][y + 1].shape,
                           afilter, check_class, check_instance):
            tiles[6] = [x, y + 1]
        if x + 1 <= cfg.maze_width - 1:
            # Bottom right
            if typeless_equals(amaze.matrix[x + 1][y + 1].shape,
                               afilter, check_class, check_instance):
                tiles[7] = [x + 1, y + 1]

    # Middle center
    if typeless_equals(amaze.matrix[x][y].shape,
                       afilter, check_class, check_instance):
        tiles[8] = [x, y]
    # Remove Nones if requested
    if not include_none:
        tiles = [tile for tile in tiles if tile is not None]

    return tiles


def typeless_equals(entity1, entity2, check_class, check_instance):
    """
    Checks if entities are equal. The check is different whether
    entities are classes or instances, which is specified in
    corresponding parameters. If neither checks are specified,
    True is returned
    """
    if check_class:
        return isinstance(entity1, entity2)
    if check_instance:
        return entity1 == entity2
    return True


def outside_maze(x_float, y_float):
    """ Checks if the coordinates are inside the maze """
    return (x_float < 0 or y_float < 0 or
            x_float > cfg.maze_width or
            y_float > cfg.maze_height)