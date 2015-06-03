import numpy as np


def centerofpos(flock, dimensions, pos):
    """Calculate the 'center of mass' for the flock of boids"""
    np.seterr(all="print")
    com = np.zeros(dimensions)
    for dim in range(dimensions):
        for boid in flock:
            com[dim] = com[dim] + boid[pos][dim]
        com[dim] /= len(flock)
    return com


def plot_raw(fig):
    """Set up the basic plotting paramaters"""
    ax = fig.add_subplot(111, projection='3d')
    ax.set_autoscale_on(False)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_xlim3d([0.0, 100.0])
    ax.set_ylim3d([0.0, 100.0])
    ax.set_zlim3d([0.0, 100.0])
    return ax


def plot_update(timestep, fig, arrays, number_of_boids, number_of_predators, dimensions, pos):
    """This plot updates the plot with the new positions of the boids
    and predators
    """
    fig.clf()
    ax = plot_raw(fig)

    # Execute a calculation of the next set of moves
    # mainloop(timestep)

    # Plots the boids in space using matplotlib
    # http://matplotlib.org/examples/mplot3d/scatter3d_demo.html
    boidpos = np.zeros((number_of_boids, dimensions))
    predpos = np.zeros((number_of_predators, dimensions))

    # Plot center of mass as a green dot
    com = centerofpos(arrays["generated_flocks"][timestep])
    ax.scatter(com[0], com[1], com[2], color='green')

    # Plot boids as blue dots
    for index, boid in enumerate(arrays["generated_flocks"][timestep]):
        boidpos[index] = boid[pos]
    bx = boidpos[:, 0]
    by = boidpos[:, 1]
    bz = boidpos[:, 2]
    ax.scatter(bx, by, bz, c='b')

    # Predators are red dots
    if number_of_predators > 0:
        for index, predator in enumerate(arrays["generated_predators"][timestep]):
            predpos[index] = predator[pos]
        px = predpos[:, 0]
        py = predpos[:, 1]
        pz = predpos[:, 2]
        ax.scatter(px, py, pz, s=100, color='red')
