"""
maze_solver.py contains Dijkstra's algorithm implementation which is
run on each iteration to find a solution path via the mapped areas of the
maze. Dijkstra's algorithm is used if cfg.solver == "CPU".
"""
import configs as cfg
import maze

UNVISITED = False
VISITED = True


def solve_maze(global_map_inst, amaze, starting_node, goal_node):
    """
    Dijkstra's algorithm
    """
    starting_node = reuse_swarm_progress(global_map_inst, starting_node)
    goal_node = reuse_swarm_progress(global_map_inst, goal_node)
    unvisited = {}
    visited = {}
    unvisited[starting_node] = 0
    for x in range(cfg.maze_width):
        for y in range(cfg.maze_height):
            if ((x, y) != starting_node and
               isinstance(amaze.matrix[x][y].shape, maze.Passage) and
               global_map_inst[x][y][cfg.NODE_IS_EXPLORED_VAR]):
                unvisited[(x, y)] = float('Inf')

    current_node = starting_node
    while True:
        neighbours = get_neighbours(current_node)
        for neighbour in neighbours:
            if neighbour in unvisited:
                if unvisited[current_node] + 1 < unvisited[neighbour]:
                    unvisited[neighbour] = unvisited[current_node] + 1
        visited[current_node] = unvisited[current_node]
        del unvisited[current_node]
        smallest_dist = float('Inf')
        closest_node = None
        for node in unvisited:
            if unvisited[node] < smallest_dist:
                smallest_dist = unvisited[node]
                closest_node = node
        if current_node == goal_node:
            return get_backwards_path(visited,
                                      starting_node,
                                      goal_node)
        if smallest_dist == float('Inf'):
            return None
        else:
            current_node = closest_node


def reuse_swarm_progress(global_map_inst, root_node):
    """
    Use the partial solutions created by agents via propagation.
    """
    current_node = root_node
    if (global_map_inst[current_node[0]]
                       [current_node[1]]
                       [cfg.NODE_IS_GOAL_VAR]):
        previous_node = None
        path_ended = False
        while not path_ended:
            neighbours = get_neighbours(current_node)
            for neighbour in neighbours:
                path_ended = True
                if (global_map_inst[neighbour[0]]
                                   [neighbour[1]]
                                   [cfg.NODE_IS_GOAL_VAR] and
                   neighbour != previous_node):
                    previous_node = current_node
                    current_node = neighbour
                    path_ended = False
                    break
    return current_node


def get_neighbours(node):
    x = node[0]
    y = node[1]
    result = []
    if y - 1 >= 0:
        # Top center
        result.append((x, y - 1))

    # Middle left
    if x - 1 >= 0:
        result.append((x - 1, y))
    # Middle right
    if x + 1 <= cfg.maze_width - 1:
        result.append((x + 1, y))

    if y + 1 <= cfg.maze_height - 1:
        # Bottom center
        result.append((x, y + 1))
    return result


def get_backwards_path(nodes, starting_node, goal_node):
    current_node = goal_node
    sequence = []
    while current_node != starting_node:
        sequence.append(current_node)
        neighbours = get_neighbours(current_node)
        smallest_distance = float('Inf')
        for neighbour in neighbours:
            if neighbour in nodes and\
               nodes[neighbour] < smallest_distance:
                smallest_distance = nodes[neighbour]
                closest_node = neighbour
        for neighbour in neighbours:
            if neighbour in nodes and \
               neighbour != closest_node:
                del nodes[neighbour]
        del nodes[current_node]
        current_node = closest_node
    sequence.append(starting_node)
    return sequence