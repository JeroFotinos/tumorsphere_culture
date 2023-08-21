import sqlite3
import os

import networkx as nx
from networkx import Graph

import plotly.express as px
import plotly.graph_objects as go

import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# ----------------- Graph generation functions -----------------


def generate_graph_at_fixed_time(
    db_path: str, culture_id: int, time: int, path_to_save: str
) -> None:
    """Generate a directed graph from a database for a given culture_id at a
    fixed time and save it to a GraphML file.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database containing the graph data.
    culture_id : int
        Identifier for the specific culture within the database.
    time : int
        Specific simulation time for which to generate the graph.
    path_to_save : str
        Path where the GraphML file should be saved.

    Returns
    -------
    None
        The function saves the graph to a file and does not return any value.

    Examples
    --------
    >>> db_path = (
        "/home/nate/Devel/tumorsphere_culture/examples/playground/merged.db"
    )
    >>> culture_id = 1
    >>> time = 5
    >>> path_to_save = "/home/nate/Devel/tumorsphere_culture/examples/playground/"
    >>> generate_graph_at_fixed_time(db_path, culture_id, time, path_to_save)

    >>> generate_graph_at_fixed_time("path/to/database.db", 42, 10, "path/to/save")
    Graph saved to /home/nate/Devel/tumorsphere_culture/examples/playground/graph_culture_id=1_time=5.graphml
    """

    # Create a directed graph
    G = nx.DiGraph()

    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Query to get the cells for the given culture_id, including 3D positions
        cells_query = """
            SELECT cell_id, _index, parent_index, position_x, position_y, position_z, t_deactivation
            FROM Cells
            WHERE culture_id = ?
        """
        cursor.execute(cells_query, (culture_id,))

        # Add nodes with 3D position attributes and directed edges to the graph
        for row in cursor.fetchall():
            (
                cell_id,
                _index,
                parent_index,
                position_x,
                position_y,
                position_z,
                t_deactivation,
            ) = row
            # Determine active status
            active = (
                True
                if t_deactivation is None or t_deactivation > time
                else False
            )

            # Find stemness status from the StemChanges table
            stem_query = """
                SELECT is_stem
                FROM StemChanges
                WHERE cell_id = ? AND t_change <= ?
                ORDER BY t_change DESC
                LIMIT 1
            """
            cursor.execute(stem_query, (cell_id, time))
            stem_result = cursor.fetchone()
            is_stem = stem_result[0] if stem_result else False

            # Add the node with attributes
            G.add_node(
                cell_id,
                _index=_index,
                position_x=position_x,
                position_y=position_y,
                position_z=position_z,
                active=active,
                stem=is_stem,
            )
            if parent_index is not None:
                # Find the cell_id of the parent
                parent_query = """
                    SELECT cell_id
                    FROM Cells
                    WHERE _index = ? AND culture_id = ?
                """
                cursor.execute(parent_query, (parent_index, culture_id))
                parent_cell_id = cursor.fetchone()[0]
                G.add_edge(parent_cell_id, cell_id)

    # Form the full path for saving the GraphML file
    file_name = f"graph_culture_id={culture_id}_time={time}.graphml"
    full_path = os.path.join(path_to_save, file_name)

    # Write to GraphML
    nx.write_graphml(G, full_path)
    print(f"Graph saved to {full_path}")


# ======= NOT WORKING =======
def generate_graph_evolution(
    db_path: str, culture_id: int, path_to_save: str
) -> None:
    # Create a list to store graph snapshots
    graphs = []

    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Query to get the maximum creation time
        max_time_query = """
            SELECT MAX(t_creation)
            FROM Cells
            WHERE culture_id = ?
        """
        cursor.execute(max_time_query, (culture_id,))
        max_time = cursor.fetchone()[0]

        # Iterate through time
        for time in range(max_time + 1):
            # Create a directed graph for the current time
            G_time = nx.DiGraph()

            # Query to get the cells for the given culture_id at the given time
            cells_query = """
                SELECT cell_id, _index, parent_index, position_x, position_y, position_z, t_deactivation
                FROM Cells
                WHERE culture_id = ? AND t_creation <= ?
            """
            cursor.execute(cells_query, (culture_id, time))

            # Add nodes and edges to the current graph based on current time
            for row in cursor.fetchall():
                (
                    cell_id,
                    _index,
                    parent_index,
                    position_x,
                    position_y,
                    position_z,
                    t_deactivation,
                ) = row

                # Determine active status
                active = (
                    True
                    if t_deactivation is None or t_deactivation > time
                    else False
                )

                # Find stemness status from the StemChanges table
                stem_query = """
                    SELECT is_stem
                    FROM StemChanges
                    WHERE cell_id = ? AND t_change <= ?
                    ORDER BY t_change DESC
                    LIMIT 1
                """
                cursor.execute(stem_query, (cell_id, time))
                stem_result = cursor.fetchone()
                is_stem = stem_result[0] if stem_result else False

                # Add or update the node with attributes
                G_time.add_node(
                    cell_id,
                    _index=_index,
                    position_x=position_x,
                    position_y=position_y,
                    position_z=position_z,
                    active=active,
                    stem=is_stem,
                    time=time,  # Include time attribute here
                )

                if parent_index is not None:
                    # Find the cell_id of the parent
                    parent_query = """
                        SELECT cell_id
                        FROM Cells
                        WHERE _index = ? AND culture_id = ?
                    """
                    cursor.execute(parent_query, (parent_index, culture_id))
                    parent_cell_id = cursor.fetchone()[0]
                    G_time.add_edge(parent_cell_id, cell_id, time=time)

            # Add the graph to the list of snapshots
            graphs.append(G_time)

    # Combine snapshots into the final graph
    G_final = nx.DiGraph()
    for G_time in graphs:
        G_final = nx.compose(G_final, G_time)

    # Form the full path for saving the GEXF file
    file_name = f"graph_evolution_culture_id={culture_id}.gexf"
    full_path = os.path.join(path_to_save, file_name)

    # Write to GEXF
    nx.write_gexf(G_final, full_path)


# ----------------- Graph plotting functions -----------------


def plot_static_graph_3D(
    G: Graph,
    spheres: bool = True,
    sphere_radius: float = 1,
    sphere_opacity: float = 0.15,
) -> None:
    """Plots in 3D the culture paternity graph (corresponding to a single
    time).

    The nodes are plotted as colored markers, with the colors determined by
    the 'active' and 'stem' attributes of the nodes:
    - active stem cells are red;
    - active non-stem cells are blue;
    - inactive stem cells are pink;
    - inactive non-stem cells are light blue.
    The edges are plotted as arrows with configurable arrowhead angles and
    lengths. If `spheres` is set to True, a sphere is plotted around each node
    with a customizable radius and opacity.

    Parameters
    ----------
    G : NetworkX Graph
        A NetworkX graph object containing the nodes and edges to be plotted.
        The nodes must have 'position_x', 'position_y', 'position_z',
        'active', and 'stem' attributes. The nodes represent cells, and
        directed edges represent parent-child relationships.
    spheres : bool, optional
        Determines whether to plot spheres centered on each node.
        The default is True.
    sphere_radius : float, optional
        The radius of the spheres centered on each node.
        The default value is 1.
    sphere_opacity : float, optional
        The opacity of the spheres centered on each node. The default value
        is 0.15, which is a relatively transparent appearance that allows for
        a clear sight of the edges.

    Examples
    --------
    >>> import networkx as nx
    >>> path_of_saved_graph = '/home/nate/Devel/tumorsphere_culture/examples/playground/'
    >>> culture_id = 1
    >>> time = 5
    >>> G = nx.read_graphml(os.path.join(path_to_save, f"graph_culture_id={culture_id}_time={time}.graphml"))
    >>> plot_static_graph_3D(G, spheres=False, sphere_radius=1, sphere_opacity=0.15)
    """

    # Lists to hold node coordinates and colors
    x_nodes = []
    y_nodes = []
    z_nodes = []
    node_colors = []

    # Iterate through nodes and add their 3D positions to the plot
    for node, data in G.nodes(data=True):
        x = data["position_x"]
        y = data["position_y"]
        z = data["position_z"]
        x_nodes.append(x)
        y_nodes.append(y)
        z_nodes.append(z)

        # Determine the color based on active and stem status
        if data["active"]:
            color = "red" if data["stem"] else "blue"
        else:
            color = "pink" if data["stem"] else "lightblue"
        node_colors.append(color)

    # Create a scatter plot for nodes
    nodes_trace = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode="markers",
        marker=dict(size=6, color=node_colors),
    )

    # Lists to hold edge coordinates and lines for arrow directions
    x_edges = []
    y_edges = []
    z_edges = []

    # Iterate through edges and add their 3D positions to the plot
    for edge in G.edges():
        x0, y0, z0 = (
            G.nodes[edge[0]]["position_x"],
            G.nodes[edge[0]]["position_y"],
            G.nodes[edge[0]]["position_z"],
        )
        x1, y1, z1 = (
            G.nodes[edge[1]]["position_x"],
            G.nodes[edge[1]]["position_y"],
            G.nodes[edge[1]]["position_z"],
        )

        # Main line
        x_edges.extend([x0, x1, None])
        y_edges.extend([y0, y1, None])
        z_edges.extend([z0, z1, None])

        # Arrow direction lines
        arrow_length = 0.2  # Customize this as needed
        arrow_angle = 30  # degrees

        direction = np.array([x1 - x0, y1 - y0, z1 - z0])
        direction /= np.linalg.norm(direction)  # Normalize

        for angle in [-arrow_angle, arrow_angle]:
            rotation_matrix = R.from_euler("z", angle, degrees=True)
            arrow_direction = rotation_matrix.apply(direction) * arrow_length

            x_arrow = x1 - arrow_direction[0]
            y_arrow = y1 - arrow_direction[1]
            z_arrow = z1 - arrow_direction[2]

            x_edges.extend([x1, x_arrow, None])
            y_edges.extend([y1, y_arrow, None])
            z_edges.extend([z1, z_arrow, None])

    # Create a line plot for edges with increased width
    edges_trace = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode="lines",
        line=dict(width=3, color="black"),
    )

    if spheres:
        # List to hold the spheres
        sphere_traces = []

        # Iterate through nodes and add a sphere around each node
        for node, data in G.nodes(data=True):
            x, y, z = (
                data["position_x"],
                data["position_y"],
                data["position_z"],
            )

            # Create mesh of a sphere
            phi, theta = np.mgrid[0.0 : 2.0 * np.pi : 100j, 0.0 : np.pi : 50j]
            x_sphere = sphere_radius * np.sin(theta) * np.cos(phi) + x
            y_sphere = sphere_radius * np.sin(theta) * np.sin(phi) + y
            z_sphere = sphere_radius * np.cos(theta) + z

            # Determine the color based on active and stem status
            if data["active"]:
                color = "red" if data["stem"] else "blue"
            else:
                color = "pink" if data["stem"] else "lightblue"

            # Create a scatter plot for spheres
            sphere_trace = go.Mesh3d(
                x=x_sphere.flatten(),
                y=y_sphere.flatten(),
                z=z_sphere.flatten(),
                alphahull=0,  # Convex hull to create a sphere
                opacity=sphere_opacity,
                color=color,
            )

            sphere_traces.append(sphere_trace)

    # Combine the plots and render
    if spheres:
        fig = go.Figure(data=[edges_trace, nodes_trace] + sphere_traces)
    else:
        fig = go.Figure(data=[edges_trace, nodes_trace])
    fig.show()


# ======= NOT WORKING =======
def plot_graph_from_gexf(file_path: str) -> None:
    # Read the graph from the GEXF file
    G = nx.read_gexf(file_path)

    # Determine the time range manually
    min_time = 0
    max_time = 5  # Please replace this with the correct value

    # Function to convert 3D coordinates to 2D radial coordinates
    def to_radial_coordinates(x, y, z):
        rho = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)
        return rho * np.cos(phi), rho * np.sin(phi)

    # Create figure and axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Function to plot the graph for a given time
    def plot_graph(time):
        # Clear current axes
        ax.clear()

        # Determine active nodes and edges
        active_nodes = [
            node for node, data in G.nodes(data=True) if data["time"] == time
        ]
        active_edges = [
            (u, v) for u, v, d in G.edges(data=True) if d["time"] == time
        ]

        pos = {
            node: to_radial_coordinates(
                data["position_x"], data["position_y"], data["position_z"]
            )
            for node, data in G.nodes(data=True)
            if node in active_nodes
        }

        nx.draw(
            G.subgraph(active_nodes),
            pos,
            edgelist=active_edges,
            node_size=20,
            with_labels=False,
            ax=ax,
        )
        ax.set_title(f"Graph at Time {time}")

    # Initial plot
    plot_graph(min_time)

    # Add a slider for time control
    ax_slider = plt.axes([0.25, 0.01, 0.50, 0.03])
    slider = Slider(
        ax_slider, "Time", min_time, max_time, valinit=min_time, valstep=1
    )

    # Update the plot when the slider is changed
    def update(val):
        time = int(slider.val)
        plot_graph(time)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


# ----------------- Module execution -----------------

if __name__ == "__main__":
    db_path = (
        "/home/nate/Devel/tumorsphere_culture/examples/playground/merged.db"
    )
    culture_id = 1
    time = 5
    path_to_save = "/home/nate/Devel/tumorsphere_culture/examples/playground/"
    path_to_gexf = "/home/nate/Devel/tumorsphere_culture/examples/playground/graph_evolution_culture_id=1.gexf"

    # # Generating the graph `.graphml` file
    # generate_graph_at_fixed_time(db_path, culture_id, time, path_to_save)

    # Plotting the graph
    # load the graph from the `.graphml` file
    G = nx.read_graphml(
        os.path.join(
            path_to_save, f"graph_culture_id={culture_id}_time={time}.graphml"
        )
    )
    # Plotting in 3D space according to coordinates
    plot_static_graph_3D(
        G, spheres=False, sphere_radius=1, sphere_opacity=0.15
    )

    # ======= NOT WORKING =======
    # # Generating the graph evolution `.gexf` file
    # generate_graph_evolution(db_path, culture_id, path_to_save)
    # # Plotting the graph evolution
    # plot_graph_from_gexf(path_to_gexf)
