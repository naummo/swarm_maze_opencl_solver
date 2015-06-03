"""
renderer.py renders the simulation animation using the data produced by simulator.py.
"""

import numpy as np

import pygame as pg

import configs as cfg
import maze

x_var = cfg.X
y_var = cfg.Y
pos_var = cfg.BOID_POS_VAR * cfg.Dimensions
pheromone_lvl_var = cfg.BOID_PHEROMONE_VAR * cfg.Dimensions


class Point:
    """ Defines a single point in 2D. All values are in pixels. """

    def __init__(self, x=0, y=0):
        self.x = np.float32(x)
        self.y = np.float32(y)


class Triangle:
    """ Defines a triangle. All values are in pixels. """

    def __init__(self, top, bottom_r, bottom_l):
        self.top = top
        self.bottom_r = bottom_r
        self.bottom_l = bottom_l
        self.surf = None
        self.rect = None

    def get_tuples(self, offset=None):
        """ Returns a list of point tuples """
        if not offset:
            offset = [0, 0]
        return [(self.top.x + offset[x_var], self.top.y + offset[y_var]),
                (self.bottom_r.x + offset[x_var], self.bottom_r.y + offset[y_var]),
                (self.bottom_l.x + offset[x_var], self.bottom_l.y + offset[y_var])]

    def generate_2d_structs(self):
        """
        Generates a surface and the least bounding rect containing the triangle.
        They won't be used for drawing, as using the alpha values distorts the triangle.
        """
        self.surf = pg.Surface((cfg.rect_diagonal_size, cfg.rect_diagonal_size), pg.SRCALPHA)

        triangle_pos = self.get_triangle_top_left()
        # Point coordinates are relative to the center, so some will be negative
        # When drawing a surface, move points until they are all positive
        tuples = self.get_tuples([(-1) * triangle_pos[x_var] if triangle_pos[x_var] < 0 else 0,
                                  (-1) * triangle_pos[y_var] if triangle_pos[y_var] < 0 else 0])
        pg.draw.polygon(self.surf, cfg.maze_boid_fill_color, tuples)
        # Antialiasing
        pg.draw.aalines(self.surf, cfg.maze_boid_contour_color, True, tuples, 1)
        self.rect = self.surf.get_bounding_rect()
        self.rect.inflate_ip(2, 2)

    def get_triangle_top_left(self):
        minimum_x = min(self.top.x, self.bottom_r.x, self.bottom_l.x)
        minimum_y = min(self.top.y, self.bottom_r.y, self.bottom_l.y)
        return [minimum_x, minimum_y]


def render_template_triangles():
    """ Computes the relative coordinates of boid triangle vertices for different orientations """
    # TODO: accelerate using OpenCL
    print("Rendering template triangles.")
    template_triangles = []
    for deg in range(0, 360, cfg.triangle_rotation_res):
        triangle = Triangle(top=Point(),
                            bottom_r=Point(),
                            bottom_l=Point())
        alpha = np.radians(deg)

        triangle.top.x = cfg.center_to_top_vertex * np.cos(alpha)
        triangle.top.y = cfg.center_to_top_vertex * np.sin(alpha)

        angle = alpha - cfg.central_bottom_half_angle
        triangle.bottom_r.x = cfg.center_to_bottom_vertex * np.sin(angle)
        triangle.bottom_r.y = (-1) * cfg.center_to_bottom_vertex * np.cos(angle)

        angle = cfg.central_bottom_half_angle - (np.pi - alpha)
        triangle.bottom_l.x = cfg.center_to_bottom_vertex * np.sin(angle)
        triangle.bottom_l.y = (-1) * cfg.center_to_bottom_vertex * np.cos(angle)

        triangle.generate_2d_structs()

        template_triangles.append(triangle)
    return template_triangles


def render_animation(amaze, flocks, template_triangles, global_map):
    """ Generate simulation """
    # Pre-calculate triangles of different orientations
    # initialize font; must be called after 'pygame.init()' to avoid 'Font not Initialized' error
    print("Rendering the animation.")
    label_font = pg.font.SysFont(cfg.label_face, cfg.label_size)
    pheromone_level_font = pg.font.SysFont(cfg.pheromone_level_face, cfg.pheromone_level_size)

    frames = []
    # TODO: use pygame.OPENGL
    screen = pg.display.set_mode(cfg.window_size)
    maze_surf = pg.Surface(cfg.window_size)

    maze_surf.fill(color=cfg.maze_bg_color)
    # Render static background -------------------------------------
    for x in range(cfg.maze_width):
        for y in range(cfg.maze_height):
            if amaze.matrix[x][y].shape == maze.Wall.normal:
                # Normal wall
                maze_surf.fill(color=cfg.maze_wall_color,
                               rect=pg.Rect(x * cfg.tile_width,
                                            y * cfg.tile_height,
                                            cfg.tile_width, cfg.tile_height))
            else:
                if isinstance(amaze.matrix[x][y].shape, maze.Wall):
                    # Exotic wall
                    render_shape(
                        x, y, amaze.matrix[x][y].orientation,
                        cfg.maze_wall_color, cfg.maze_wall_bg_color,
                        maze.wall_shapes.get_shapes(amaze.matrix[x][y].shape),
                        maze_surf)
                else:
                    if isinstance(amaze.matrix[x][y].shape, maze.Passage):
                        # Exotic passage
                        render_shape(
                            x, y, amaze.matrix[x][y].orientation,
                            cfg.maze_passage_color, cfg.maze_bg_color,
                            maze.passage_shapes.get_shapes(amaze.matrix[x][y].shape),
                            maze_surf)
    # Render frames -----------------------------------------------
    for step in range(int(cfg.startSecond * cfg.framespersecond),
                      cfg.total_timesteps):
        # Render dynamic background -------------------------------------
        new_maze_surf = maze_surf.copy().convert_alpha()
        for x in range(cfg.maze_width):
            for y in range(cfg.maze_height):
                if isinstance(amaze.matrix[x][y].shape, maze.Passage):
                    if not global_map[step][x][y][cfg.NODE_IS_EXPLORED_VAR]:
                        # Is the node explored?
                        render_overlay(x, y, cfg.maze_unexplored_color, new_maze_surf)
                    else:
                        if global_map[step][x][y][cfg.NODE_IS_DEADEND_VAR]:
                            render_overlay(x, y, cfg.maze_deadend_color, new_maze_surf)
                    if global_map[step][x][y][cfg.NODE_IS_GOAL_VAR]:
                        render_overlay(x, y, cfg.maze_goal_color, new_maze_surf)
                    # Node pheromone level
                    render_node_pheromone_level(x, y,
                                                global_map[step][x][y][cfg.NODE_PHEROMONE_A_VAR],
                                                pheromone_level_font, new_maze_surf)

        # Copy the maze onto the screen Surface
        screen.blit(new_maze_surf, (0, 0))

        i = 0
        for boid in flocks[step].np_arrays:
            triangle_center = render_triangle(screen, flocks[step], boid, i, template_triangles,
                                              cfg.maze_boid_fill_color, cfg.maze_boid_contour_color)
            render_triangle_label(triangle_center, i, label_font, screen)
            render_agent_pheromone_level(triangle_center, boid[pheromone_lvl_var], label_font, screen)
            i += 1

        frames.append(screen.copy())

    for _ in range(cfg.end_pause_dur * cfg.framespersecond):
        frames.append(screen.copy())
    return frames


def render_shape(x, y, orientation, fill_color, bg_color, shapes, surf):
    if shapes is not None:
        square_surf = pg.Surface((cfg.tile_width, cfg.tile_height), pg.SRCALPHA)
        square_surf.fill(bg_color)

        for shape, values in shapes:
            if shape == "polygon":
                pg.draw.polygon(square_surf, fill_color, values)
            if shape == "rect":
                square_surf.fill(fill_color, pg.Rect(values))
            if shape == "circle":
                pg.draw.circle(square_surf, fill_color, values[0:2], values[2])

        # Rotate square
        angle = 0.0
        if orientation == maze.Orientation.east:
            angle = -90.0
        if orientation == maze.Orientation.south:
            angle = 180
        if orientation == maze.Orientation.west:
            angle = 90
        wall_surf = pg.transform.rotate(square_surf, angle)

        surf.blit(wall_surf, maze.to_coors(x, y))


def render_overlay(x, y, fill_color, alpha_surf):
    square_surf = pg.Surface((cfg.tile_width, cfg.tile_height), pg.SRCALPHA)
    square_surf.fill(fill_color)
    alpha_surf.blit(square_surf, maze.to_coors(x, y))


def render_triangle(screen, flock, boid, boid_n, template_triangles, fill_color, contour_color):
    """ Renders a single triangle """
    # Get a rotated triangle without location
    template_triangle = template_triangles[
        np.minimum(int(np.round(np.degrees(flock.object_list[boid_n].orientation))), 359)]
    triangle_center = maze.to_coors(boid[pos_var + x_var], boid[pos_var + y_var])

    if cfg.bounding_rects_show:
        # Draw a bounding rectangle
        if flock.object_list[boid_n].collided:
            rect_color = cfg.maze_boid_inside_color
        else:
            rect_color = cfg.maze_boid_oustide_color

        triangle_pos = template_triangle.get_triangle_top_left()
        triangle_surf = template_triangle.surf.copy()
        triangle_surf.fill(rect_color, template_triangle.rect)
        screen.blit(triangle_surf, [triangle_center[x_var] + triangle_pos[x_var],
                                    triangle_center[y_var] + triangle_pos[y_var]])

    tuples = template_triangle.get_tuples(offset=triangle_center)
    pg.draw.polygon(screen, fill_color, tuples)
    # Antialiasing
    pg.draw.aalines(screen, contour_color, True, tuples, 1)

    return triangle_center


def render_triangle_label(triangle_center, boid_n, label_font, surf):
    """ Renders boid number near its boid's triangle """
    label = label_font.render(str(boid_n), True, cfg.boid_label_color)
    surf.blit(label, [triangle_center[x_var] + cfg.tile_width * 0.25,
                      triangle_center[y_var] + cfg.tile_height * 0.1])


def render_agent_pheromone_level(triangle_center, pheromone_level, label_font, surf):
    """ Renders agent's pheromone level near its boid's triangle """
    if pheromone_level >= 1000:
        pheromone_level_text = "999"
    else:
        if pheromone_level < 100:
            pheromone_level_text = "%4.1f" % pheromone_level
        else:
            pheromone_level_text = "%.0f" % pheromone_level
    label = label_font.render(pheromone_level_text, True, cfg.boid_label_color)
    surf.blit(label, [triangle_center[x_var] - cfg.tile_width * 0.6,
                      triangle_center[y_var] - cfg.tile_height * 0.8])


def render_node_pheromone_level(x, y, pheromone_level, label_font, surf):
    """ Renders pheromone level in the center of a square (stigmergy principle) """
    if pheromone_level >= 1000:
        pheromone_level_text = "999"
    else:
        if pheromone_level < 100:
            pheromone_level_text = "%4.1f" % pheromone_level
        else:
            pheromone_level_text = "%.0f" % pheromone_level
    label = label_font.render(pheromone_level_text, True, cfg.node_label_color)
    surf.blit(label, maze.to_coors(x + 0.05, y + 0.25))
