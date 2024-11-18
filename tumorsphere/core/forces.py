from abc import ABC, abstractmethod

import numpy as np


class Force(ABC):
    """
    The force or model used to calculate the interaction between 1 cell and all its 
    neighbors.
    """

    @abstractmethod
    def calculate_interaction(
        self,
        cells,
        phies,
        cell_index,
        delta_t,
        area,
    ):
        """
        Given the force/model, it returns the change in the position and in the
        orientation of the cell because of the force or torque exerted.
        """
        pass


class No_Forces(Force):
    """
    There are no forces in the system.
    """

    def __init__(self):
        pass

    def calculate_interaction(
        self,
        cells,
        phies,
        cell_index,
        delta_t,
        area,
    ):
        cell = cells[cell_index]
        # there is no change in the orientation and no force so the only change in
        # position is because of the intrinsic velocity
        dif_phi = 0
        dif_position = (cell.velocity())*delta_t
        return dif_position, dif_phi


class Spring_Force(Force):
    """
    The force used is a spring force. When two cells collide, they "bounce" on the
    opposite direction.
    """

    def __init__(
        self,
        k_spring_force: float = 0.5,
    ):
        self.k_spring_force = k_spring_force

    def calculate_interaction(
        self,
        cells,
        phies,
        cell_index,
        delta_t,
        area,
    ):
        cell = cells[cell_index]
        # initialization of the parameters of interaction
        dif_phi = 0
        dif_velocity = np.zeros(3)
        # Calculate interaction with filtered neighbors
        for neighbor_index, data in cell.neighbors_data.items():
            relative_pos = data["relative_pos"]
            overlap = data["overlap"]
            relative_pos_x, relative_pos_y = relative_pos
            # we first calculate the force
            fx = -self.k_spring_force * (-relative_pos_x)  # OJO SIGNO
            fy = -self.k_spring_force * (-relative_pos_y)
            # Calculate change in velocity given by the force model
            dif_velocity_2 = np.array([fx, fy, 0])
            # Accumulate changes in velocity
            dif_velocity += dif_velocity_2
        # In this model the change in the velocity is equal to the force
        dif_position = (cell.velocity() + dif_velocity)*delta_t
        # and the change in the orientation is given by the new velocity
        dif_phi = (
            np.arctan2((cell.velocity()[1] + dif_velocity[1]), (cell.velocity()[0] + dif_velocity[0]))
            - phies[cell_index]
        )

        return dif_position, dif_phi


class Vicsek(Force):
    """
    The cells move using the Vicsek model: if they are close enough (if they touch),
    their orientations allign.
    """

    def __init__(self):
        pass

    def calculate_interaction(
        self,
        cells,
        phies,
        cell_index,
        

        delta_t,
        area,
    ):
        # In this model there is no change in the velocity but in the orientation
        cell = cells[cell_index]
        # initialization of the parameters of interaction
        dif_phi = 0
        # Calculate interaction with filtered neighbors
        for neighbor_index, data in cell.neighbors_data.items():
            relative_pos = data["relative_pos"]
            overlap = data["overlap"]
            alpha = np.arctan2(
                np.sin(phies[cell_index]) + np.sin(phies[neighbor_index]),
                np.cos(phies[cell_index]) + np.cos(phies[neighbor_index]),
            )
            dif_phi_2 = alpha - phies[cell_index]
            dif_phi += dif_phi_2
        dif_position = cell.velocity()*delta_t
        return dif_position, dif_phi


class Vicsek_and_Spring_Force(Force):
    """
    Vicsek and Spring Force combined. They allign and bounce.
    """

    def __init__(
        self,
        k_spring_force: float = 0.5,
    ):
        self.k_spring_force = k_spring_force

    def calculate_interaction(
        self,
        cells,
        phies,
        cell_index,
        
        delta_t,
        area,
    ):
        cell = cells[cell_index]
        # initialization of the parameters of interaction
        dif_phi = 0
        dif_velocity = np.zeros(3)
        # Calculate interaction with filtered neighbors
        for neighbor_index, data in cell.neighbors_data.items():
            relative_pos = data["relative_pos"]
            overlap = data["overlap"]
            relative_pos_x, relative_pos_y = relative_pos
            # We first calculate the force
            fx = -self.k_spring_force * (-relative_pos_x)  # OJO SIGNO
            fy = -self.k_spring_force * (-relative_pos_y)
            # Calculate change in velocity given by the force model
            dif_velocity_2 = np.array([fx, fy, 0])
            # Accumulate changes in velocity
            dif_velocity += dif_velocity_2
            # and the change in the orientation is given by Vicsek
            alpha = np.arctan2(
                np.sin(phies[cell_index]) + np.sin(phies[neighbor_index]),
                np.cos(phies[cell_index]) + np.cos(phies[neighbor_index]),
            )
            dif_phi_2 = alpha - phies[cell_index]
            dif_phi += dif_phi_2
        # In this model the change in velocity is equal to the force
        dif_position = (cell.velocity() + np.array([fx, fy, 0]))*delta_t
        return dif_position, dif_phi


class Grosmann(Force):
    """
    The model is the given by Grosmann paper.
    """

    def __init__(
        self,
        kRep: float = 10,
        bExp: float = 3,
    ):
        self.kRep = kRep
        self.bExp = bExp

    def calculate_interaction(
        self,
        cells,
        phies,
        cell_index,
        
        delta_t,
        area,
    ):
        # First of all we are going to calculate the force and the torque
        cell = cells[cell_index]
        # We calculate the parameters of the cell
        # nematic matrix
        Q_cell = np.array(
            [
                [
                    np.cos(2 * phies[cell_index]),
                    np.sin(2 * phies[cell_index]),
                    0,
                ],
                [
                    np.sin(2 * phies[cell_index]),
                    -np.cos(2 * phies[cell_index]),
                    0,
                ],
                [0, 0, 0],
            ]
        )
        # anisotropy
        eps = (cell.aspect_ratio**2 - 1) / (cell.aspect_ratio**2 + 1)
        # diagonal squared
        diag2 = (area / np.pi) * (cell.aspect_ratio + 1 / cell.aspect_ratio)
        # longitudinal & transversal mobility
        if np.isclose(cell.aspect_ratio, 1):
            mP = 1 / np.sqrt((area * cell.aspect_ratio) / np.pi)
            mS = 1 / np.sqrt((area * cell.aspect_ratio) / np.pi)
        else:
            mP = (
                1
                / np.sqrt((area * cell.aspect_ratio) / np.pi)
                * (3 * cell.aspect_ratio / 4.0)
                * (
                    (cell.aspect_ratio) / (1 - cell.aspect_ratio**2)
                    + (2.0 * cell.aspect_ratio**2 - 1.0)
                    / np.power(cell.aspect_ratio**2 - 1.0, 1.5)
                    * np.log(
                        cell.aspect_ratio + np.sqrt(cell.aspect_ratio**2 - 1.0)
                    )
                )
            )
            mS = (
                1
                / np.sqrt((area * cell.aspect_ratio) / np.pi)
                * (3 * cell.aspect_ratio / 8.0)
                * (
                    (cell.aspect_ratio) / (cell.aspect_ratio**2 - 1.0)
                    + (2.0 * cell.aspect_ratio**2 - 3.0)
                    / np.power(cell.aspect_ratio**2 - 1.0, 1.5)
                    * np.log(
                        cell.aspect_ratio + np.sqrt(cell.aspect_ratio**2 - 1.0)
                    )
                )
            )

        # rotational mobility
        mR = (
            3
            / (
                2
                * (area / np.pi)
                * (cell.aspect_ratio + 1 / cell.aspect_ratio)
            )
            * mP
        )
        # initialization of the parameters of interaction
        # dif_phi = 0
        torque = 0
        # dif_velocity = np.zeros(3)
        force = np.zeros(3)
        # Calculate interaction with filtered neighbors
        for neighbor_index, data in cell.neighbors_data.items():
            relative_pos = data["relative_pos"]
            overlap = data["overlap"]
            relative_pos_x, relative_pos_y = relative_pos
            # Calculate change in velocity and orientation given by the force model
            # First we calculate some parameters of the neighbor cell
            # nematic matrix
            Q_neighbor = np.array(
                [
                    [
                        np.cos(2 * phies[neighbor_index]),
                        np.sin(2 * phies[neighbor_index]),
                        0,
                    ],
                    [
                        np.sin(2 * phies[neighbor_index]),
                        -np.cos(2 * phies[neighbor_index]),
                        0,
                    ],
                    [0, 0, 0],
                ]
            )
            # and some parameters useful for the force
            # mean nematic matrix
            mean_nematic = (1 / 2) * (Q_cell + Q_neighbor)
            # relative position and angle
            relative_pos = np.array([relative_pos_x, relative_pos_y, 0])
            relative_angle = phies[cell_index] - phies[neighbor_index]
            # and now we can calculate xi
            xi = np.exp(
                -1
                * np.matmul(
                    relative_pos,
                    (np.matmul(np.identity(3) - eps * mean_nematic, relative_pos)),
                )
                / (2 * (1 - eps**2 * (np.cos(relative_angle)) ** 2) * diag2)
            )

            # the kernel is: (k_rep = k, b_exp=gamma (from the paper))
            kernel = (self.kRep * self.bExp * xi**self.bExp) / (
                diag2 * (1 - eps**2 * (np.cos(relative_angle)) ** 2)
            )

            # finally we can calculate the force:
            force_2 = kernel * np.matmul(
                np.identity(3) - eps * mean_nematic, relative_pos
            )

            # On the other way, we calculate the torque
            # we introduce the theta = angle of r_kj
            theta = np.arctan2(relative_pos_y, relative_pos_x)
            torque_2 = (kernel / 2) * (
                eps
                * np.linalg.norm(relative_pos)
                ** 2  # (relative_pos_x**2 + relative_pos_y**2)
                * np.sin(2 * (phies[cell_index] - theta))
                + eps**2
                * (
                    np.matmul(
                        relative_pos,
                        (
                            np.matmul(
                                np.identity(3) - eps * mean_nematic, relative_pos
                            )
                        ),
                    )
                    * np.sin(2 * (-relative_angle))
                    / (1 - eps**2 * (np.cos(relative_angle)) ** 2)
                )
            )
            #dif_velocity_2 = np.array([0, 0, 0])
            #dif_phi_2 = 0
            # Accumulate changes in force and torque
            force += force_2
            torque += torque_2

        # Now that we have the force and torque we can calculate the change in velocity
        # and orientation as it is done in the paper.
        # then the change in the velocity is given by:
        dif_velocity = np.matmul(
            ((mP + mS) / 2) * np.identity(3) + ((mP - mS) / 2) * Q_cell,
            force,
        )
        # we calculate the change in the position of the cell, given all the neighbors.
        # Remember that the intrinsic velocity is already multiplied by the mobility
        # (Like in Grosmann paper).
        dif_position = (cell.velocity()+dif_velocity)*delta_t
        # and the change in the orientation:
        dif_phi = mR * torque * delta_t
        return dif_position, dif_phi


class Anisotropic_Grosmann(Force):
    """
    The model is the given by the generalization of Grosmann paper.
    """

    def __init__(
        self,
        kRep: float = 10,
        bExp: float = 3,
        eta: float = 0.1,
    ):
        self.kRep = kRep
        self.bExp = bExp
        self.eta = eta

    def calculate_interaction(
        self,
        cells,
        phies,
        cell_index,
        delta_t,
        area,
    ):
        # First of all we are going to calculate the force and torque and then
        # we see how these change the velocity and orientation
        cell = cells[cell_index]
        # We calculate the parameters of the cell
        # nematic matrix
        Q_cell = np.array(
            [
                [
                    np.cos(2 * phies[cell_index]),
                    np.sin(2 * phies[cell_index]),
                    0,
                ],
                [
                    np.sin(2 * phies[cell_index]),
                    -np.cos(2 * phies[cell_index]),
                    0,
                ],
                [0, 0, 0],
            ]
        )
        # anisotropy
        eps_cell = (cell.aspect_ratio**2 - 1) / (cell.aspect_ratio**2 + 1)
        # diagonal squared (what we call alpha)
        alpha_cell = (area / np.pi) * (
            cell.aspect_ratio + 1 / cell.aspect_ratio
        )
        # longitudinal & transversal mobility
        if np.isclose(cell.aspect_ratio, 1):
            mP = 1 / np.sqrt((area * cell.aspect_ratio) / np.pi)
            mS = 1 / np.sqrt((area * cell.aspect_ratio) / np.pi)
        else:
            mP = (
                1
                / np.sqrt((area * cell.aspect_ratio) / np.pi)
                * (3 * cell.aspect_ratio / 4.0)
                * (
                    (cell.aspect_ratio) / (1 - cell.aspect_ratio**2)
                    + (2.0 * cell.aspect_ratio**2 - 1.0)
                    / np.power(cell.aspect_ratio**2 - 1.0, 1.5)
                    * np.log(
                        cell.aspect_ratio + np.sqrt(cell.aspect_ratio**2 - 1.0)
                    )
                )
            )
            mS = (
                1
                / np.sqrt((area * cell.aspect_ratio) / np.pi)
                * (3 * cell.aspect_ratio / 8.0)
                * (
                    (cell.aspect_ratio) / (cell.aspect_ratio**2 - 1.0)
                    + (2.0 * cell.aspect_ratio**2 - 3.0)
                    / np.power(cell.aspect_ratio**2 - 1.0, 1.5)
                    * np.log(
                        cell.aspect_ratio + np.sqrt(cell.aspect_ratio**2 - 1.0)
                    )
                )
            )

        # rotational mobility
        mR = (
            3
            / (
                2
                * (area / np.pi)
                * (cell.aspect_ratio + 1 / cell.aspect_ratio)
            )
            * mP
        )
        # initialization of the parameters of interaction
        # dif_phi = 0
        torque = 0
        # dif_velocity = np.zeros(3)
        force = np.zeros(3)
        # Calculate interaction with filtered neighbors
        #for neighbor_index, relative_pos in significant_neighbors:
        #for neighbor_index, relative_pos, overlap in cell.neighbors_data:
        for neighbor_index, data in cell.neighbors_data.items():
            relative_pos = data["relative_pos"]
            overlap = data["overlap"]
            relative_pos_x, relative_pos_y = relative_pos
            # Calculate change in velocity and orientation given by the force model
            neighbor = cells[neighbor_index]
            # First we calculate some parameters of the neighbor cell
            # nematic matrix
            Q_neighbor = np.array(
                [
                    [
                        np.cos(2 * phies[neighbor_index]),
                        np.sin(2 * phies[neighbor_index]),
                        0,
                    ],
                    [
                        np.sin(2 * phies[neighbor_index]),
                        -np.cos(2 * phies[neighbor_index]),
                        0,
                    ],
                    [0, 0, 0],
                ]
            )
            # anisotropy
            eps_neighbor = (neighbor.aspect_ratio**2 - 1) / (
                neighbor.aspect_ratio**2 + 1
            )
            # diagonal squared (what we call alpha)
            alpha_neighbor = (area / np.pi) * (
                neighbor.aspect_ratio + 1 / neighbor.aspect_ratio
            )
            # and now some parameters of the cell and its neighbor
            # relative position and angle
            relative_pos = np.array([relative_pos_x, relative_pos_y, 0])
            relative_angle = phies[cell_index] - phies[neighbor_index]
            
            # we now calculate the mean nematic matrix (different than before) (the matrix M)
            matrix_M = (
                alpha_cell * eps_cell * Q_cell
                + alpha_neighbor * eps_neighbor * Q_neighbor
            ) / (alpha_cell + alpha_neighbor)

            # now we introduce the constant beta introduced by us in the TF
            beta = (
                (alpha_cell + alpha_neighbor) ** 2
                - (alpha_cell * eps_cell - alpha_neighbor * eps_neighbor) ** 2
                - 4
                * alpha_cell
                * eps_cell
                * alpha_neighbor
                * eps_neighbor
                * (np.cos(relative_angle)) ** 2
            )

            # calculate the kernel, using f[ξ]=ξ**gamma. ξ=xi calculated
            # and now we can calculate xi
            xi = overlap/(4 * area**2 / (np.pi * np.sqrt(beta)))

            # the kernel is: (k_rep = k, b_exp=gamma (from the paper))
            kernel = (
                2
                * self.kRep
                * self.bExp
                * xi**self.bExp
                * ((alpha_cell + alpha_neighbor) / beta)
            )

            # finally we can calculate the force:
            force_2 = kernel * np.matmul(np.identity(3) - matrix_M, relative_pos)

            # On the other way, we calculate the torque
            # we introduce the theta=angle of r_kj
            theta = np.arctan2(relative_pos_y, relative_pos_x)
            torque_2 = kernel * (
                (
                    (
                        2
                        * alpha_cell
                        * eps_cell
                        * alpha_neighbor
                        * eps_neighbor
                        * np.sin(-2 * relative_angle)
                    )
                    / beta
                )
                * np.matmul(
                    relative_pos,
                    np.matmul(np.identity(3) - matrix_M, relative_pos),
                )
                + (alpha_cell * eps_cell / (alpha_cell + alpha_neighbor))
                * np.linalg.norm(relative_pos) ** 2
                * np.sin(2 * (phies[cell_index] - theta))
            )

            # Accumulate changes in force and torque
            force += force_2
            torque += torque_2
            # We introduce some matrix/vectors/parameters that we need
        
        # then the change in the velocity is given by:
        dif_velocity = np.matmul(
            ((mP + mS) / 2) * np.identity(3) + ((mP - mS) / 2) * Q_cell,
            force,
        )
        # we calculate the change in the position of the cell, given all the neighbors.
        # Remember that the intrinsic velocity is already multiplied by the mobility
        # (Like in Grosmann paper).
        dif_position = (cell.velocity()+dif_velocity)*delta_t
        # and the change in the orientation:
        dif_phi = mR * torque * delta_t
        # we also add tha noise in the position:
        # we need the direction vectors
        direction_vector = np.array(
            [
                np.cos(phies[cell_index]),
                np.sin(phies[cell_index]),
                0,
            ])
        perpendicular_vector = np.array(
            [
                np.cos(phies[cell_index]+np.pi/2),
                np.sin(phies[cell_index]+np.pi/2),
                0,
            ])
        # and the noise
        s_nP = self.eta*np.sqrt(mP*delta_t)
        s_nS = self.eta*np.sqrt(mS*delta_t)

        #nP = s_nP*np.random.normal(0, 1)
        nP = s_nP*cell.culture.rng.normal(0, 1)
        #nS = s_nS*np.random.normal(0, 1)
        nS = s_nS*cell.culture.rng.normal(0, 1)
        noise = nP*direction_vector+nS*perpendicular_vector
        return dif_position+noise, dif_phi
