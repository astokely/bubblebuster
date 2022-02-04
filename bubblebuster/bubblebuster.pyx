##############################################################################
# BubbleBuster: A Python Library for detecting water box bubbles in 
#               structural files used in molecular simulations. 
#
# Copyright (c) 2021 Andy Stokely 
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
###############################################################################


from math import ceil
from math import floor
from typing import Dict
from typing import Optional
from typing import Tuple

import mdtraj as md
import numpy as np
from mdtraj.core.trajectory import Trajectory
from numpy import ndarray

__all__ = ['water_box_properties', 'get_cube_bounds']


class WaterBoxProperties(object):
    __slots__ = (
        'box_vectors', 'box_dimensions',
        'mesh', 'cubic_partition',
        '_empty_cubes', 'bounds'
    )
    __setattr_calls__ = 0
    def __new__(
            cls,
            *args,
            **kwargs
    ):
        cls.__setattr_calls__ = 0
        return super(WaterBoxProperties, cls).__new__(cls)

    def __init__(
            self,
            box_vectors: np.ndarray,
            box_dimensions: np.ndarray,
            mesh: Tuple,
            cubic_partition: Dict,
            empty_cubes: Tuple,
            bounds: np.ndarray,
    ) -> None:
        self.box_vectors = box_vectors
        self.box_dimensions = box_dimensions
        self.mesh = mesh
        self.cubic_partition = cubic_partition
        self._empty_cubes = empty_cubes
        self.bounds = bounds

    def __setattr__(
            self,
            attr,
            val
    ):
        if self.__class__.__setattr_calls__ >= len(self.__slots__):
            return
        super().__setattr__(attr, val)
        self.__class__.__setattr_calls__ += 1

    def __get_cube_bounds(
            self,
            cube_index: int,
    ) -> np.ndarray:
        i, j, k = list(self.cubic_partition)[cube_index]
        cube_bounds = np.zeros((3, 2), dtype=np.float64)
        wx, wy, wz = self.mesh
        x_lb = self.bounds[0, 0]
        y_lb = self.bounds[1, 0]
        z_lb = self.bounds[2, 0]
        cube_bounds[0, 0] = (i * wx) + x_lb
        cube_bounds[0, 1] = (i * wx) + x_lb + wx
        cube_bounds[1, 0] = (j * wy) + y_lb
        cube_bounds[1, 1] = (j * wy) + y_lb + wy
        cube_bounds[2, 0] = (k * wz) + z_lb
        cube_bounds[2, 1] = (k * wz) + z_lb + wz
        return cube_bounds

    @property
    def empty_cubes(
            self
    ):
        return {i: self.__get_cube_bounds(i) for i in self._empty_cubes}

    @empty_cubes.setter
    def empty_cubes(
            self,
            empty_cubes
    ):
        self._empty_cubes = empty_cubes


def get_box_vectors(
        trajectory: Trajectory,
) -> np.ndarray:
    return trajectory.unitcell_vectors[0]

def get_periodic_box_volume(
        box_vectors: np.ndarray
) -> np.ndarray:
    a, b, c = box_vectors
    return abs(np.dot(a, np.cross(b, c)))

def get_box_dimensions(
        box_vectors: np.ndarray,
) -> np.ndarray:
    a, b, c = md.utils.box_vectors_to_lengths_and_angles(
        *box_vectors
    )[:3]
    return np.array([a, b, c])

def get_coordinates(
        trajectory: Trajectory,
) -> ndarray:
    npacoords = np.empty((trajectory.n_atoms, 3), dtype=np.float64)
    for i in range(trajectory.n_atoms):
        npacoords[i][0] = trajectory.xyz[0, i, 0]
        npacoords[i][1] = trajectory.xyz[0, i, 1]
        npacoords[i][2] = trajectory.xyz[0, i, 2]
    coords = [
        atom_coordinates for atom_coordinates
        in trajectory.xyz[0]
    ]
    coords = np.array(coords)
    return coords

def pbc_scale_factor(
        box_dimensions: np.ndarray,
        atom_coordinates: np.ndarray,
        coordinate_component_index: int,
) -> float:
    return floor(
        atom_coordinates[coordinate_component_index]
        / box_dimensions[coordinate_component_index]
    )

def apply_pbc_to_triclinic_coordinates(
        box_dimensions: np.ndarray,
        atom_coordinates: np.ndarray,
        box_vectors: np.ndarray,
) -> np.ndarray:
    for i in reversed(range(3)):
        for j in range(3):
            scale_factor = pbc_scale_factor(
                box_dimensions,
                atom_coordinates, i
            )
            atom_coordinates[j] -= \
                scale_factor * box_vectors[i][j]
    return atom_coordinates

def apply_pbc_to_cubic_coordinates(
        box_dimensions: np.ndarray,
        atom_coordinates: np.ndarray,
        box_vectors: np.ndarray,
) -> np.ndarray:
    for i in range(3):
        scale_factor = pbc_scale_factor(
            box_dimensions,
            atom_coordinates, i
        )
        atom_coordinates[i] -= \
            scale_factor * box_vectors[i][i]
    return atom_coordinates

def apply_pbc_to_coordinates(
        box_dimensions: np.ndarray,
        atom_coordinates: np.ndarray,
        box_vectors: np.ndarray,
        box_type: str,
) -> np.ndarray:
    if box_type == "triclinic":
        return apply_pbc_to_triclinic_coordinates(
            box_dimensions,
            atom_coordinates,
            box_vectors,
        )
    elif box_type == "cubic":
        return apply_pbc_to_cubic_coordinates(
            box_dimensions,
            atom_coordinates,
            box_vectors,
        )

def apply_pbc(
        coordinates: np.ndarray,
        box_vectors: np.ndarray,
        box_dimensions: np.ndarray,
        box_type: str,
) -> np.ndarray:
    box_dimensions = get_box_dimensions(
        box_vectors
    )
    for coords in coordinates:
        apply_pbc_to_coordinates(
            box_dimensions,
            coords,
            box_vectors,
            box_type
        )
    return coordinates

def wrap_structure(
        coordinates: np.ndarray,
        box_vectors: np.ndarray,
        box_dimensions: np.ndarray,
        box_type: str,
) -> np.ndarray:
    apply_pbc(coordinates, box_vectors, box_dimensions, box_type)
    bounds = coordinate_bounds(coordinates)
    if (round(bounds[0, 0], 2) == 0.0
            and round(bounds[1, 0], 2) == 0.0
            and round(bounds[2, 0], 2) == 0.0
    ):
        return coordinates
    return wrap_structure(coordinates, box_type)

def coordinate_bounds(
        coordinates: np.ndarray,
) -> np.ndarray:
    coordinates_transpose = np.array(coordinates).T
    bounds = np.zeros((3, 2), dtype=np.float64)
    for i in range(3):
        bounds[i, 0] = min(coordinates_transpose[i])
        bounds[i, 1] = max(coordinates_transpose[i])
    return bounds

def get_num_subintervals(
        lower_bound: float,
        upper_bound: float,
        mesh: Tuple,
) -> int:
    num_subintervals = ceil(
        (upper_bound - lower_bound) / mesh[0]
    )
    return num_subintervals

def get_mesh(
        bounds: np.ndarray,
        num_subintervals: Tuple,
) -> Tuple:
    mesh = []
    for i in range(3):
        mesh.append((bounds[i, 1] - bounds[i, 0]) / num_subintervals[i])
    return mesh

def get_cubic_partition(
        coordinate_bounds: np.ndarray,
        num_x_subintervals: int,
        num_y_subintervals: int,
        num_z_subintervals: int,
) -> Dict:
    cube_dict = {}
    for i in range(num_x_subintervals):
        for j in range(num_y_subintervals):
            for k in range(num_z_subintervals):
                cube_dict[(i, j, k)] = 0
    return cube_dict

def hash_widths(
        bounds: np.ndarray,
        delta: Optional[float] = 1e-7,
)-> Tuple:
    return (
        (bounds[0, 1] - bounds[0, 0]) + delta,
        (bounds[1, 1] - bounds[1, 0]) + delta,
        (bounds[2, 1] - bounds[2, 0]) + delta
    )

def get_cubic_partition_occupancy(
        coordinates: np.ndarray,
        num_x_subintervals: int,
        num_y_subintervals: int,
        num_z_subintervals: int,
        hash_widths: Tuple,
        bounds: np.ndarray,
        cubic_partition: Dict,
) -> Dict:
    for x, y, z in coordinates:
        i = int(
            num_x_subintervals * ((x - bounds[0, 0]) / hash_widths[0])
        )
        j = int(
            num_y_subintervals * ((y - bounds[1, 0]) / hash_widths[1])
        )
        k = int(
            num_z_subintervals * ((z - bounds[2, 0]) / hash_widths[2])
        )
        cubic_partition[(i, j, k)] += 1
    return cubic_partition

def get_empty_cubes(
        cubic_partition: Dict,
) -> Tuple:
    empty_cubes = []
    cube_index = 0
    for k, v in cubic_partition.items():
        if v == 0:
            empty_cubes.append(cube_index)
        cube_index += 1
    return tuple(empty_cubes)

def get_cube_bounds(
        water_box_properties: WaterBoxProperties,
        cube_index: int,
) -> np.ndarray:
    i, j, k = list(water_box_properties.cubic_partition)[cube_index]
    cube_bounds = np.zeros((3, 2), dtype=np.float64)
    wx, wy, wz = water_box_properties.mesh
    x_lb = water_box_properties.bounds[0, 0]
    y_lb = water_box_properties.bounds[1, 0]
    z_lb = water_box_properties.bounds[2, 0]
    cube_bounds[0, 0] = (i * wx) + x_lb
    cube_bounds[0, 1] = (i * wx) + x_lb + wx
    cube_bounds[1, 0] = (j * wy) + y_lb
    cube_bounds[1, 1] = (j * wy) + y_lb + wy
    cube_bounds[2, 0] = (k * wz) + z_lb
    cube_bounds[2, 1] = (k * wz) + z_lb + wz
    return cube_bounds

def get_box_type(
        box_vectors: np.ndarray,
) -> str:
    off_diagonals = [
        int(box_vectors[i][j]) for i in range(3)
        for j in range(3) if i != j
    ]
    if sum(off_diagonals) == 0:
        return "cubic"
    return "triclinic"

def load_mdtraj_trajectory(
        structure_file: str,
        topology_file: Optional[str] = '',
) -> Trajectory:
    if topology_file == '':
        return md.load(structure_file)
    return md.load(structure_file, top=topology_file)

def water_box_properties(
        structure_file: str,
        box_vectors: Optional[np.ndarray] = False,
        mesh: Optional[Tuple[float]] = False,
        cutoff: Optional[float] = 0.5,
        topology_file: Optional[str] = '',
        wrapped_structure_filename: Optional[str] = '',
) -> WaterBoxProperties:
    trajectory = load_mdtraj_trajectory(
        structure_file,
        topology_file=topology_file
    )
    if not box_vectors:
        box_vectors = get_box_vectors(trajectory)
    box_dimensions = get_box_dimensions(box_vectors)
    if not mesh:
        mesh = (
            box_dimensions[0] / 10.0,
            box_dimensions[1] / 10.0,
            box_dimensions[2] / 10.0,
        )

    box_type = get_box_type(box_vectors)
    trajectory.center_coordinates()
    unwrapped_coordinates = get_coordinates(trajectory)
    wrapped_coordinates = wrap_structure(
        unwrapped_coordinates,
        box_vectors,
        box_dimensions,
        box_type
    )
    trajectory.xyz[0, :, :] = wrapped_coordinates[:, :]
    topology = trajectory.topology
    coordinates = get_coordinates(trajectory)
    if wrapped_structure_filename:
        trajectory.save_pdb(wrapped_structure_filename)
    bounds = coordinate_bounds(coordinates)
    nx, ny, nz = (
        get_num_subintervals(
            lower_bound=bounds[0, 0], upper_bound=bounds[0, 1],
            mesh=mesh
        ),
        get_num_subintervals(
            lower_bound=bounds[1, 0], upper_bound=bounds[1, 1],
            mesh=mesh
        ),
        get_num_subintervals(
            lower_bound=bounds[2, 0], upper_bound=bounds[2, 1],
            mesh=mesh
        ),
    )
    cubic_partition = get_cubic_partition(
        bounds, nx, ny, nz
    )
    cubic_partition = get_cubic_partition_occupancy(
        coordinates=coordinates,
        num_x_subintervals=nx,
        num_y_subintervals=ny,
        num_z_subintervals=nz,
        hash_widths=hash_widths(bounds),
        bounds=bounds,
        cubic_partition=cubic_partition,
    )
    empty_cubes = get_empty_cubes(cubic_partition)
    mesh = get_mesh(bounds, (nx, ny, nz))
    properties = WaterBoxProperties(
        box_vectors,
        box_dimensions,
        mesh,
        cubic_partition,
        empty_cubes,
        bounds,
    )
    return properties
