from abc import ABC, abstractmethod

import numpy as np


class Forces(ABC):
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
    ):
        pass

class No_Forces(Forces):
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
    ):
        
        return np.array([0, 0, 0]), 0
    
class Spring_Force(Forces):
    def __init__(self,
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
    ):
        cell = cells[cell_index]
        
        fx = -self.k_spring_force * (-relative_pos_x)   # OJO SIGNO
        fy = -self.k_spring_force * (-relative_pos_y)
        
        dphi2 = np.arctan2((cell.velocity()[1]+fy),(cell.velocity()[0]+fx))-phies[cell_index]
        
        return np.array([fx, fy, 0]), dphi2

class Vicsek(Forces):
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
    ):  
        fx = 0
        fy = 0
        alpha = np.arctan2(np.sin(phies[cell_index])+np.sin(phies[neighbor_index]),np.cos(phies[cell_index])+np.cos(phies[neighbor_index]))
        dphi2 = alpha-phies[cell_index]
        return np.array([fx, fy, 0]), dphi2

class Vicsek_and_Spring_Force(Forces):
    def __init__(self,
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
    ):  
        fx = -self.k_spring_force * (-relative_pos_x)   # OJO SIGNO
        fy = -self.k_spring_force * (-relative_pos_y)
        alpha = np.arctan2(np.sin(phies[cell_index])+np.sin(phies[neighbor_index]),np.cos(phies[cell_index])+np.cos(phies[neighbor_index]))
        dphi2 = alpha-phies[cell_index]
        return np.array([fx, fy, 0]), dphi2
    
class Grossman(Forces):
    def __init__(self,
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
    ):
              
        cell = cells[cell_index]

        # we calculate parameter derived from the cell's attributes
        s_Rot, s_Rep, s_DmPmS, s_SmPmS, s_epsA05, s_epsA2, s_diag2 = (
            cell.derived_parameters(self.kRep, self.bExp)
        )

        alpha = np.arctan2(relative_pos_y, relative_pos_x)
        # relative distance
        distance = np.sqrt(relative_pos_x**2+relative_pos_y**2)

        # angular relation
        dP = phies[cell_index] - phies[neighbor_index]
        c2dP = np.cos(dP) * np.cos(dP)

        # abbreviate some symmetric factors
        g_Lij = 1.0 / (1.0 - s_epsA2 * c2dP)
        g_Sij = (
            0.5
            * g_Lij
            * (distance**2 / s_diag2)
            * (
                1.0
                - s_epsA05
                * (
                    np.cos(2.0 * (phies[cell_index] - alpha))
                    + np.cos(2.0 * (phies[neighbor_index] - alpha))
                )
            )
        )
        g_Kij = g_Lij * np.exp(-self.bExp * g_Sij)

        # force onto i from j
        sfx = (
            g_Kij
            * distance
            * (
                np.cos(alpha)
                - s_epsA05
                * (
                    np.cos(2.0 * phies[cell_index] - alpha)
                    + np.cos(2.0 * phies[neighbor_index] - alpha)
                )
            )
        )
        sfy = (
            g_Kij
            * distance
            * (
                np.sin(alpha)
                - s_epsA05
                * (
                    np.sin(2.0 * phies[cell_index] - alpha)
                    + np.sin(2.0 * phies[neighbor_index] - alpha)
                )
            )
        )

        # torque onto i from j
        sfphi = (
            g_Kij
            * s_epsA05
            * distance**2
            * np.sin(2.0 * (phies[cell_index] - alpha))
        )
        sfphi = sfphi + g_Kij * s_epsA2 * s_diag2 * g_Sij * np.sin(
            2.0
            * (phies[neighbor_index] - phies[cell_index])
        )

        fx = s_Rep * (
            (s_SmPmS + s_DmPmS * np.cos(2 * phies[cell_index])) * sfx
            + s_DmPmS * np.sin(2 * phies[cell_index]) * sfy
        )
        fy = s_Rep * (
            (s_SmPmS - s_DmPmS * np.cos(2 * phies[cell_index])) * sfy
            + s_DmPmS * np.sin(2 * phies[cell_index]) * sfx
        )

        dphi = delta_t * s_Rot * sfphi
        return np.array([fx, fy, 0]), dphi

def choose_force(name_force):
    if name_force == "No_Forces":
        type_force = No_Forces()
    elif name_force == "Spring_Force":
        type_force = Spring_Force()
    elif name_force == "Vicsek":
        type_force = Vicsek()
    elif name_force == "Vicsek_and_Spring_Force":
        type_force = Vicsek_and_Spring_Force()
    elif name_force == "Grossman":
        type_force = Grossman()
    return type_force