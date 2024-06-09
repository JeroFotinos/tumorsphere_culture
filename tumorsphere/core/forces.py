from abc import ABC, abstractmethod

import numpy as np


class Force(ABC):
    @abstractmethod
    def calculate_force(
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
        pass


class No_Forces(Force):
    def __init__(self):
        pass

    def calculate_force(
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

        return np.array([0, 0, 0]), 0


class Spring_Force(Force):
    def __init__(
        self,
        k_spring_force: float = 0.5,
    ):
        self.k_spring_force = k_spring_force

    def calculate_force(
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

        fx = -self.k_spring_force * (-relative_pos_x)  # OJO SIGNO
        fy = -self.k_spring_force * (-relative_pos_y)

        dphi2 = (
            np.arctan2((cell.velocity()[1] + fy), (cell.velocity()[0] + fx))
            - phies[cell_index]
        )

        return np.array([fx, fy, 0]), dphi2


class Vicsek(Force):
    def __init__(self):
        pass

    def calculate_force(
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
        fx = 0
        fy = 0
        alpha = np.arctan2(
            np.sin(phies[cell_index]) + np.sin(phies[neighbor_index]),
            np.cos(phies[cell_index]) + np.cos(phies[neighbor_index]),
        )
        dphi2 = alpha - phies[cell_index]
        return np.array([fx, fy, 0]), dphi2


class Vicsek_and_Spring_Force(Force):
    def __init__(
        self,
        k_spring_force: float = 0.5,
    ):
        self.k_spring_force = k_spring_force

    def calculate_force(
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
        fx = -self.k_spring_force * (-relative_pos_x)  # OJO SIGNO
        fy = -self.k_spring_force * (-relative_pos_y)
        alpha = np.arctan2(
            np.sin(phies[cell_index]) + np.sin(phies[neighbor_index]),
            np.cos(phies[cell_index]) + np.cos(phies[neighbor_index]),
        )
        dphi2 = alpha - phies[cell_index]
        return np.array([fx, fy, 0]), dphi2


class Grossman(Force):
    def __init__(
        self,
        kRep: float = 10,
        bExp: float = 3,
    ):
        self.kRep = kRep
        self.bExp = bExp

    def calculate_force(
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

        # we first calculate the kernel, using f[ξ]=ξ**lambda. ξ=xi from the paper
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

        # the kernel is: (k_rep = k, b_exp=lambda (from the paper))
        kernel = (self.kRep * self.bExp * xi**self.bExp) / (
            diag2 * (1 - eps**2 * (np.cos(relative_angle)) ** 2)
        )

        # finally we can calculate the force:
        force = kernel * np.matmul(
            np.identity(3) - eps * mean_nematic, relative_pos
        )

        # On the other way, we calculate the torque
        # we introduce the alpha=angle of r_kj
        alpha = np.arctan2(relative_pos_y, relative_pos_x)
        torque = (kernel / 2) * (
            eps
            * (relative_pos_x**2 + relative_pos_y**2)
            * np.sin(2 * (phies[cell_index] - alpha))
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
        dphi = torque * delta_t
        return force, dphi


class Grossman_LB_Code(Grossman):
    """Grossman force with the extra term of the LB code"""

    def __init__(
        self,
        kRep: float = 10,
        bExp: float = 3,
    ):
        super().__init__(kRep, bExp)

    def calculate_force(
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
        force_paper, dphi_paper = super().calculate_force(
            cells,
            phies,
            cell_index,
            neighbor_index,
            relative_pos_x,
            relative_pos_y,
            delta_t,
            area,
        )

        cell = cells[cell_index]

        # we calculate the mobilities
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

        # and the matrix Q
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

        # and finally the force and dphi:
        force = np.matmul(
            ((mP + mS) / 2) * np.identity(3) + ((mP - mS) / 2) * Q_cell,
            force_paper,
        )
        dphi = mR * dphi_paper
        return force, dphi


class Anisotropic_Grossman(Force):
    """Anisotropic Grossman force"""

    pass


class Anisotropic_Grossman_LB_Code(Anisotropic_Grossman):
    pass
