from abc import ABC, abstractmethod

import numpy as np


class Force(ABC):
    """
    The force or model used to calculate the interaction between 2 cells.
    """
    @abstractmethod
    def calculate_interaction(
        self,
        cells,
        phies,
        cell_index,
        neighbor_index,
        relative_pos_x,
        relative_pos_y,
        delta_t,
        area,
    ):
        """
        Given the force/model, it returns the change in the velocity and in the 
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
        neighbor_index,
        relative_pos_x,
        relative_pos_y,
        delta_t,
        area,
    ):
        dif_velocity = np.array([0, 0, 0])
        dif_phi = 0 
        return dif_velocity, dif_phi


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
        neighbor_index,
        relative_pos_x,
        relative_pos_y,
        delta_t,
        area,
    ):
        cell = cells[cell_index]
        # we first calculate the force
        fx = -self.k_spring_force * (-relative_pos_x)  # OJO SIGNO
        fy = -self.k_spring_force * (-relative_pos_y)
        # In this model the change in the velocity is equal to the force
        dif_velocity = np.array([fx, fy, 0])
        # and the change in the orientation is given by the new velocity
        dif_phi = (
            np.arctan2((cell.velocity()[1] + fy), (cell.velocity()[0] + fx))
            - phies[cell_index]
        )

        return dif_velocity, dif_phi


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
        neighbor_index,
        relative_pos_x,
        relative_pos_y,
        delta_t,
        area,
    ):
        # In this model there is no change in the velocity but in the orientation
        dif_velocity = np.array([0, 0, 0])
        alpha = np.arctan2(
            np.sin(phies[cell_index]) + np.sin(phies[neighbor_index]),
            np.cos(phies[cell_index]) + np.cos(phies[neighbor_index]),
        )
        dif_phi = alpha - phies[cell_index]
        return dif_velocity, dif_phi


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
        neighbor_index,
        relative_pos_x,
        relative_pos_y,
        delta_t,
        area,
    ):
        # We first calculate the force
        fx = -self.k_spring_force * (-relative_pos_x)  # OJO SIGNO
        fy = -self.k_spring_force * (-relative_pos_y)
        # In this model the change in velocity is equal to the force
        dif_velocity = np.array([fx, fy, 0])
        # and the change in the orientation is given by Vicsek
        alpha = np.arctan2(
            np.sin(phies[cell_index]) + np.sin(phies[neighbor_index]),
            np.cos(phies[cell_index]) + np.cos(phies[neighbor_index]),
        )
        dif_phi = alpha - phies[cell_index]
        return dif_velocity, dif_phi


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
        neighbor_index,
        relative_pos_x,
        relative_pos_y,
        delta_t,
        area,
    ):
        # First of all we are going to calculate the force and the torque
        cell = cells[cell_index]

        # we first calculate the kernel, using f[ξ]=ξ**gamma. ξ=xi from the paper
        # We introduce some matrix/vectors/parameters that we need

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

        # mean nematic matrix
        mean_nematic = (1 / 2) * (Q_cell + Q_neighbor)
        # relative position and angle
        relative_pos = np.array([relative_pos_x, relative_pos_y, 0])
        relative_angle = phies[cell_index] - phies[neighbor_index]

        # anisotropy
        eps = (cell.aspect_ratio**2 - 1) / (cell.aspect_ratio**2 + 1)
        # diagonal squared
        diag2 = (area / np.pi) * (cell.aspect_ratio + 1 / cell.aspect_ratio)

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
        force = kernel * np.matmul(
            np.identity(3) - eps * mean_nematic, relative_pos
        )

        # On the other way, we calculate the torque
        # we introduce the theta = angle of r_kj
        theta = np.arctan2(relative_pos_y, relative_pos_x)
        torque = (kernel / 2) * (
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

        # Now that we have the force and torque we can calculate the change in velocity
        # and orientation as it is done in the paper. Becuase of this, we need the Q 
        # matrix (already calculated) and the mobilities

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

        # then the change in the velocity is given by:
        dif_velocity = np.matmul(
            ((mP + mS) / 2) * np.identity(3) + ((mP - mS) / 2) * Q_cell,
            force,
        )
        # and the change in the orientation:
        dif_phi = mR * torque * delta_t
        return dif_velocity, dif_phi


class Anisotropic_Grosmann(Force):
    """
    The model is the given by the generalization of Grosmann paper.
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
        neighbor_index,
        relative_pos_x,
        relative_pos_y,
        delta_t,
        area,
    ):
        # First of all we are going to calculate the force and torque and then
        # we see how these change the velocity and orientation
        cell = cells[cell_index]
        neighbor = cells[neighbor_index]
        # we first calculate the kernel, using f[ξ]=ξ**gamma. ξ=xi calculated
        # We introduce some matrix/vectors/parameters that we need

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

        # relative position and angle
        relative_pos = np.array([relative_pos_x, relative_pos_y, 0])
        relative_angle = phies[cell_index] - phies[neighbor_index]

        # anisotropy
        eps_cell = (cell.aspect_ratio**2 - 1) / (cell.aspect_ratio**2 + 1)
        eps_neighbor = (neighbor.aspect_ratio**2 - 1) / (
            neighbor.aspect_ratio**2 + 1
        )
        # diagonal squared (what whe call alpha)
        alpha_cell = (area / np.pi) * (
            cell.aspect_ratio + 1 / cell.aspect_ratio
        )
        alpha_neighbor = (area / np.pi) * (
            neighbor.aspect_ratio + 1 / neighbor.aspect_ratio
        )

        # we now calculate the mean nematic matrix (different than before)
        mean_nematic = (
            alpha_cell * eps_cell * Q_cell
            + alpha_neighbor * eps_neighbor * Q_neighbor
        ) / (alpha_cell + alpha_neighbor)

        # and now we can calculate xi
        xi = np.exp(
            -1
            * (
                (alpha_cell + alpha_neighbor)
                / (
                    (alpha_cell + alpha_neighbor) ** 2
                    - (alpha_cell * eps_cell + alpha_neighbor * eps_neighbor)
                    ** 2
                    - 4
                    * alpha_cell
                    * eps_cell
                    * alpha_neighbor
                    * eps_neighbor
                    * np.cos(relative_angle) ** 2
                )
            )
            * np.matmul(
                relative_pos,
                np.matmul(np.identity(3) - mean_nematic, relative_pos),
            )
        )
        # the kernel is: (k_rep = k, b_exp=gamma (from the paper))
        kernel = (
            2
            * self.kRep
            * self.bExp
            * xi**self.bExp
            * (
                (alpha_cell + alpha_neighbor)
                / (
                    (alpha_cell + alpha_neighbor) ** 2
                    - (alpha_cell * eps_cell + alpha_neighbor * eps_neighbor)
                    ** 2
                    - 4
                    * alpha_cell
                    * eps_cell
                    * alpha_neighbor
                    * eps_neighbor
                    * np.cos(relative_angle) ** 2
                )
            )
        )

        # finally we can calculate the force:
        force = kernel * np.matmul(np.identity(3) - mean_nematic, relative_pos)

        # On the other way, we calculate the torque
        # we introduce the theta=angle of r_kj
        theta = np.arctan2(relative_pos_y, relative_pos_x)
        torque = kernel * (
            (
                (
                    2
                    * alpha_cell
                    * eps_cell
                    * alpha_neighbor
                    * eps_neighbor
                    * np.sin(-2 * relative_angle)
                )
                / (
                    (alpha_cell + alpha_neighbor) ** 2
                    - (alpha_cell * eps_cell + alpha_neighbor * eps_neighbor)
                    ** 2
                    - 4
                    * alpha_cell
                    * eps_cell
                    * alpha_neighbor
                    * eps_neighbor
                    * np.cos(relative_angle) ** 2
                )
            )
            * np.matmul(
                relative_pos,
                np.matmul(np.identity(3) - mean_nematic, relative_pos),
            )
            + (alpha_cell * eps_cell / (alpha_cell + alpha_neighbor))
            * np.linalg.norm(relative_pos) ** 2
            * np.sin(2 * (phies[cell_index] - theta))
        )

        # Now that we have the force and torque we can calculate the change in velocity
        # and orientation as it is done in the paper. Becuase of this, we need the Q 
        # matrix (already calculated) and the mobilities

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

        # then the change in the velocity is given by:
        dif_velocity = np.matmul(
            ((mP + mS) / 2) * np.identity(3) + ((mP - mS) / 2) * Q_cell,
            force,
        )
        # and the change in the orientation:
        dif_phi = mR * torque * delta_t
        return dif_velocity, dif_phi