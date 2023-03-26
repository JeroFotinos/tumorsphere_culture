import matplotlib.animation as animation

from cells import *
from culture import *

def simulate_and_animate_growth(culture, num_steps, filename):
    # initialize the figure and the plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-20, 20)

    def update_plot(frame):
        # simulate the growth for one step
        culture.simulate(1)

        # plot the cells as spheres
        ax.clear()
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_zlim(-20, 20)
        for cell in culture.cells:
            x, y, z = cell.position
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            sphere_x = x + np.cos(u) * np.sin(v) * cell.radius
            sphere_y = y + np.sin(u) * np.sin(v) * cell.radius
            sphere_z = z + np.cos(v) * cell.radius
            ax.plot_surface(sphere_x, sphere_y, sphere_z, color=cell.color, alpha=0.2)

    # create the animation
    anim = animation.FuncAnimation(fig, update_plot, frames=num_steps, repeat=False)

    # save the animation to file
    anim.save(filename, writer="ffmpeg")


culture = Culture(cell_max_repro_attempts=100)
simulate_and_animate_growth(culture, num_steps=10, filename="growth.mp4")
