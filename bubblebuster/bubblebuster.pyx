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

from math import floor, \
    ceil
from typing import Dict, Optional, Tuple

import mdtraj as md
import numpy as np
from mdtraj.core.trajectory import Trajectory
from numpy import ndarray


class BoxInfo(object):
    """Holds the water box properties pertinent to checking
    the box for bubbles.

    Parameters
    ----------
    box_type : str, optional
        The type of box the water box is. If the water box is a
        cubic box, box_type is set to 'cubic'. Else,
        box_type is set to triclinic.
    box_vectors : ndarray, optional
        Vectors that define the shape and size of the water box
        in nanometers. To convert to angstroms, multiply each vector
        by 10.
    dimensions : ndarray, optional
        1D numpy array containing the magnitudes of the box's
        unit vectors. The first elment is the magnitude of the
        unit vector a, which lies on the x-plane. The second
        element is the magnitude of the unit vector b, which lies
        on the y-plane. The third element is the magnitude of the
        unit vector c, which lies on the z-plane.
    volume : float, optional
        Volume of the water box.
    cubic_partition : dict
        Dictionary of the number of atoms in each cube. If a cube has zero
        atoms, then the cube is part of a water bubble.
    atom_density : float, optional
        Water box's atom density, which calculated by dividing the
        total number of atoms in the box by it's volume.
    number_of_atoms : int, optional
        Total number of atoms in the water box.
    cubic_partition_bounds : ndarray, optional
        Numpy array containing the x, y, and z bounds of each cube in
        the box's cubic partition.
    mesh : float, optional
        Length of cube's sides in the box's cubic partition. A smaller
        mesh will increase the bubble calculation's confidence at the
        price of significantly decreasing the calculation's speed. Due
        to this accuracy/performance tradeoff, the mesh value must be
        chosen carefully, especially if a large number of water
        box's have to be checked for bubbles.

    Attributes
    ----------
    box_type : str
        The type of box the water box is. If the water box is a
        cubic box, box_type is set to 'cubic'. Else,
        box_type is set to triclinic.
    box_vectors : ndarray
        Vectors that define the shape and size of the water box
        in nanometers. To convert to angstroms, multiply each vector
        by 10.
    dimensions : ndarray
        1D numpy array containing the magnitudes of the box's
        unit vectors. The first elment is the magnitude of the
        unit vector a, which lies on the x-plane. The second
        element is the magnitude of the unit vector b, which lies
        on the y-plane. The third element is the magnitude of the
        unit vector c, which lies on the z-plane.
    volume : float
        Volume of the water box.
    atom_density : float
        Water box's atom density, which calculated by dividing the
        total number of atoms in the box by it's volume.
    number_of_atoms : int
        Total number of atoms in the water box.
    cubic_partition : dict
        Dictionary of the number of atoms in each cube. If a cube has zero
        atoms, then the cube is part of a water bubble.
    cubic_partition_bounds : ndarray, optional
        Numpy array containing the x, y, and z bounds of each cube in
        the box's cubic partition.
    mesh : float, optional
        Length of cube's sides in the box's cubic partition. A smaller
        mesh will increase the bubble calculation's confidence and the
        price of significantly decreasing the calculation's speed. Due
        to this accuracy/performance tradeoff, the mesh value must be
        choosen vary carefully, especially if a large number of water
        box's have to be checked for bubbles.

    """
    def __init__(
            self,
            box_type: Optional[str] = None,
            box_vectors: Optional[np.ndarray] = None,
            dimensions: Optional[np.ndarray] = None,
            volume: Optional[float] = None,
            atom_density: Optional[float] = None,
            number_of_atoms: Optional[int] = None,
            cubic_partition: Optional[Dict] = None,
            cubic_partition_bounds: Optional[np.ndarray] = None,
            mesh: Optional[float] = 1.0,
    ) -> None:
        self.box_type = box_type
        self.box_vectors = box_vectors
        self.dimensions = dimensions
        self.volume = volume
        self.atom_density = atom_density
        self.number_of_atoms = number_of_atoms
        self.cubic_partition = cubic_partition
        self.cubic_partition_bounds = \
            cubic_partition_bounds
        self.mesh = mesh


def get_box_vectors(
        trajectory: Trajectory,
) -> np.ndarray:
    """
    Returns the box vectors of a mdtraj trajectory's water box.

    Parameters
    ----------
    trajectory : mdtraj.core.trajectory
        mdtraj trajectory object that holds the molecular system of
        interest's structural information.

    Returns
    -------
    : ndarray
        Trajectory's water box box vectors.

    """
    return trajectory.unitcell_vectors[0]

def get_periodic_box_volume(
        box_vectors: np.ndarray
) -> np.ndarray:
    """
    Returns the volume of the system's periodic box.

    Parameters
    ----------
    box_vectors : ndarray
        Box vector matrix.

    Returns
    -------
    : ndarray
        Volume of the system's periodic box.
    """
    a, b, c = box_vectors
    return abs(np.dot(a, np.cross(b, c)))

def get_box_vector_dimensions(
        box_vectors: np.ndarray,
) -> np.ndarray:
    """
    Returns the magnitude of the trajectory's period box unit
    vectors: a, b, and c.

    Parameters
    ----------
    box_vectors : ndarray
        Box vector matrix.

    Returns
    -------
    : ndarray
        Magnitude of the trajectory's period box unit vectors: a, b,
        and c.

    """
    a, b, c = md.utils.box_vectors_to_lengths_and_angles(
        *box_vectors
    )[:3]
    return np.array([a, b, c])

def get_coordinates(
        trajectory: Trajectory,
) -> ndarray:
    """
    Returns the cartesian coordinates of each atom in the trajectory's
    first frame.

    Parameters
    ----------
    trajectory : mdtraj.core.trajectory
        mdtraj trajectory object that holds the molecular system of
        interest's structural information.

    Returns
    -------
    : ndarray
        Cartesian coordinates of each atom in the trajectory's
        first frame.

    """
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
    #coords = list(coords)
    return coords

def pbc_scale_factor(
        box_vector_dimensions: np.ndarray,
        atom_coordinates: np.ndarray,
        coordinate_component_index: int,
) -> float:
    """
    Factor used to scale the each coordinate component of each atom
    in a molecular system with defined periodic boundary conditions.
    The periodic boundary conditions scale factor is calculated by
    taking the floor of the quotient obtained by dividing each
    component of an atom's coordinate (x, y, or z) by the magnitude of
    the periodic box's corresponding unit cell vector. The
    scale factor will evaluate to: 0 if the coordinate component is
    inside the periodic box, 1 if the coordinate component is
    outside the box by a distance less then or equal to two times
    the magnitude of the corresponding periodic box unit cell vector,
    2 if the coordinate component is outside the box by a distance
    greater then the previously mentioned quantity and outside of the
    box by a distance less then or equal to three times the
    magnitude of
    the corresponding periodic box unit cell vector, etc ...

    Parameters
    ----------
    box_vector_dimensions : ndarray

    atomic_coordinates : ndarray

    coordinate_component_index : int

    Returns
    -------
    : float
    Period boundary conditions scale factor used to scale an atomic
    coordinate component according to the system's periodic boundary
    conditions.

    """
    return floor(
        atom_coordinates[coordinate_component_index]
        / box_vector_dimensions[coordinate_component_index]
    )

def apply_pbc_to_triclinic_coordinates(
        box_vector_dimensions: np.ndarray,
        atom_coordinates: np.ndarray,
        box_vectors: np.ndarray,
) -> np.ndarray:
    """
    Wraps the coordinates of an atom in a system with a triclinic
    water box. Each coordinate component is wrapped according to the
    below equation.

    coord[i]_pbc = coord[i] - [(floor(coord[2] / ||c||) * V[2][i]
                              + (floor(coord[1] / ||b||) * V[1][i]
                              + (floor(coord[0] / ||a||) * V[0][i]]

    for i in range(3), where a, b, and c are the magnitudes of the
    periodic box's unit vectors, V is the periodic box vector matrix,
    and coord[i] is the i'th component of the starting atomic
    coordinates.

    Parameters
    ----------
    box_vector_dimensions : ndarray
        Magnitudes of the periodic box unit vectors.
    atom_coordinates : ndarray
        Cartesian coordinates of the atom being wrapped.
    box_vectors : ndarray
        Box vectors of the system's water box.

    Returns
    -------
    : ndarray
        Wrapped version of the input atom coordinates.

    """
    for i in reversed(range(3)):
        for j in range(3):
            scale_factor = pbc_scale_factor(
                box_vector_dimensions,
                atom_coordinates, i
            )
            atom_coordinates[j] -= \
                scale_factor * box_vectors[i][j]
    return atom_coordinates

def apply_pbc_to_cubic_coordinates(
        box_vector_dimensions: np.ndarray,
        atom_coordinates: np.ndarray,
        box_vectors: np.ndarray,
) -> np.ndarray:
    """
    Wraps the coordinates of an atom in a system with a cubic
    water box. Each coordinate component is wrapped according to the
    below equation.

    coord[0]_pbc = coord[0] - [(floor(coord[0] / ||a||) * V[0][0]
    coord[1]_pbc = coord[1] - [(floor(coord[1] / ||b||) * V[1][1]
    coord[2]_pbc = coord[2] - [(floor(coord[2] / ||c||) * V[2][2]

    where a, b, and c are the magnitudes of the
    periodic box's unit vectors, V is the periodic box vector matrix,
    and coord[i] is the i'th component of the starting atomic
    coordinates.

    Parameters
    ----------
    box_vector_dimensions : ndarray
        Magnitudes of the periodic box unit vectors.
    atom_coordinates : ndarray
        Cartesian coordinates of the atom being wrapped.
    box_vectors : ndarray
        Box vectors of the system's water box.

    Returns
    -------
    : ndarray
        Wrapped version of the input atom coordinates.

    """
    for i in range(3):
        scale_factor = pbc_scale_factor(
            box_vector_dimensions,
            atom_coordinates, i
        )
        atom_coordinates[i] -= \
            scale_factor * box_vectors[i][i]
    return atom_coordinates

def apply_pbc_to_coordinates(
        box_vector_dimensions: np.ndarray,
        atom_coordinates: np.ndarray,
        box_vectors: np.ndarray,
        box_type: str,
) -> np.ndarray:
    """
    Wraps the coordinates of an atom in a system with periodic
    boundary conditions. If the system has a cubic water box,
    the atom's coordinates are wrapped by the
    apply_pbc_to_cubic_coordinates function, and if the system
    has a triclinic water box the atom's coordinates are wrapped by
    the apply_pbc_to_cubic_coordinates function.

    Parameters
    ----------
    box_vector_dimensions : ndarray
        Magnitudes of the periodic box unit vectors.
    atom_coordinates : ndarray
        Cartesian coordinates of the atom being wrapped.
    box_vectors : ndarray
        Box vectors of the system's water box.

    Returns
    -------
    : ndarray
        Wrapped version of the input atom coordinates.

    """
    if box_type == "triclinic":
        return apply_pbc_to_triclinic_coordinates(
            box_vector_dimensions,
            atom_coordinates,
            box_vectors,
        )
    elif box_type == "cubic":
        return apply_pbc_to_cubic_coordinates(
            box_vector_dimensions,
            atom_coordinates,
            box_vectors,
        )

def apply_pbc(
        coordinates: ndarray,
        box_info: BoxInfo,
) -> Trajectory:
    """
    Wraps all atoms in the provided trajectory according to the
    systems periodic boundary conditions.

    Parameters
    ----------
    trajectory : mdtraj.core.trajectory
        mdtraj trajectory object that holds the molecular system of
        interest's structural information.
    box_info : BoxInfo
        BoxInfo class object that holds the properties of the system's
        water box.

    Returns
    -------
    trajectory : mdtraj.core.trajectory
        mdtraj trajectory object with atomic coordinates
        wrapped according to the system's periodic boundary
        conditions.

    """

    box_vectors = box_info.box_vectors
    box_vector_dimensions = get_box_vector_dimensions(
        box_vectors
    )
    box_info.dimensions = box_vector_dimensions
    box_info.volume = get_periodic_box_volume(box_vectors)
    for atom_coordinates in coordinates:
        apply_pbc_to_coordinates(
            box_vector_dimensions,
            atom_coordinates,
            box_vectors,
            box_info.box_type
        )
    return coordinates

def wrap_structure(
        coordinates: ndarray,
        box_info: BoxInfo,
) -> Trajectory:
    """
    Wraps all atoms in the provided trajectory according to the
    systems periodic boundary conditions and ensures that all of
    the atoms are wrapped into the same periodic image. This function
    handles special cases where atoms are multiple periodic images away
    from the periodic box. To check to see if all wrapped atoms are
    in the same periodic image, the wrapped system's x, y and z
    coordinate minimums are calculated. If each of these minimums equal
    zero, all of the wrapped atomic coordinates are in the same
    periodic
    image. wrap_structure is called recursively until the above
    condition is
    met.

    Parameters
    ----------
    trajectory : mdtraj.core.trajectory
        mdtraj trajectory object that holds the molecular system of
        interest's structural information.
    box_info : BoxInfo
        BoxInfo class object that holds the properties of the system's
        water box.

    Returns
    -------
    trajectory : mdtraj.core.trajectory
        mdtraj trajectory object with atomic coordinates
        wrapped according to the system's periodic boundary
        conditions.

    """
    apply_pbc(coordinates, box_info)
    bounds = coordinate_bounds(coordinates)
    if (round(bounds[0, 0], 2) == 0.0
            and round(bounds[1, 0], 2) == 0.0
            and round(bounds[2, 0], 2) == 0.0
    ):
        return coordinates
    return wrap_structure(coordinates, box_info)

def coordinate_bounds(
        coordinates: np.ndarray,
) -> np.ndarray:
    """
    x, y, and z upper and lower bounds for an array of cartesian
    coordinate vectors. This function is used to calculate the wrapped
    structure's x, y, and z coordinate bounds.

    Parameters
    ----------
    coordinates : ndarray
        2D numpy array containing the cartesian coordinates of the
        system's atoms.

    Returns
    -------
    : ndarray
        Returns a numpy 2D array of the system's minimum and maximum
        x, y, and z atomic coordinate values.
        Eg.) [[xmin, xmax], [ymin, ymax], [zmin, zmax]].

    """
    coordinates_transpose = np.array(coordinates).T
    bounds = np.zeros((3, 2), dtype=np.float64)
    for i in range(3):
        bounds[i, 0] = min(coordinates_transpose[i])
        bounds[i, 1] = max(coordinates_transpose[i])
    return bounds

def get_num_subintervals(
        box_info: BoxInfo(),
        lower_bound: float,
        upper_bound: float,
) -> int:
    """
    Calculates the number of sub-intervals used to construct a
    partition
    with bounds defined by the lower_bound and upper_bound parameters
    and a mesh defined by the BoxInfo object's mesh attribute.

    Parameters
    ----------
    box_info : BoxInfo
        BoxInfo class object that holds the properties of the system's
        water box. The BoxInfo mesh attribute is used in this function
        defines the uniform sub-interval length.
    lower_bound : float
        Lower bound of the partition.
    upper_bound : float
        Upper bound of the partition.

    Returns
    -------
    : int
        The number of sub-intervals used to construct a partition
        with bounds defined by the lower_bound and upper_bound
        parameters
        and a mesh defined by the BoxInfo object's mesh attribute.
    """
    num_subintervals = ceil(
        (upper_bound - lower_bound) / box_info.mesh[0]
    )
    return num_subintervals

def get_cubic_partition(
        coordinate_bounds: np.ndarray,
        num_x_subintervals: int,
        num_y_subintervals: int,
        num_z_subintervals: int,
        box_info: BoxInfo,
) -> Dict:
    """
    Constructs the cubic partition that is used to calculate local atom
    densities throughout the system. The properties of each cube in the
    partition are stored in a unique CubeInfo object.

    Parameters
    ----------
    coordinate_bounds : np.ndarray
        Numpy array with the minimums and maximums for the
        system's atoms x, y, and z coordinates.

    num_x_subintervals : int
        Number of x-axis subintervals.

    num_y_subintervals : int
        Number of y-axis subintervals.

    num_z_subintervals : int
        Number of z-axis subintervals.

    box_info : BoxInfo
        BoxInfo class object that holds the properties of the system's
        water box. The BoxInfo mesh attribute is used in this function
        defines the uniform sub-interval length.

    Returns
    -------
    : dict

    """
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
    """
    :param bounds: Numpy array of the x, y, z
        coordinate upper and lower bounds
    :type:np.ndarray

    :param delta:
    :type:float, optional

    :return:
    :rtype: tuple
    """
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
        i = int(num_x_subintervals * ((x - bounds[0, 0]) / hash_widths[0]))
        j = int(num_y_subintervals * ((y - bounds[1, 0]) / hash_widths[1]))
        k = int(num_z_subintervals * ((z - bounds[2, 0]) / hash_widths[2]))
        cubic_partition[(i, j, k)] += 1
    return cubic_partition

def get_box_type(
        box_vectors: np.ndarray,
) -> str:
    """
    Returns the system's type of periodic box. The type of
    periodic box can be determined by examining the box vector
    matrix. Box vectors for cubic boxes will form a diagonal
    matrix, while box vectors for triclinic boxes will not.
    Using this property, the box type is determined by looking
    at all of the box vector matrix's off diagonal elements. If
    any of them are non-zero, then the box is triclinic. Otherwise,
    the system's periodic box is cubic.

    Parameters
    ----------
    box_vectors : ndarray
        Box vectors of the system's water box.

    Returns
    -------
    : str
        System's type of periodic box.

    """
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

def periodic_box_properties(
        structure_file: str,
        box_vectors: Optional[np.ndarray] = None,
        mesh: Optional[Tuple[float]] = (1.5, 1.5, 1.5),
        cutoff: Optional[float] = 0.5,
        topology_file: Optional[str] = '',
        wrapped_structure_filename: Optional[str] = '',
) -> BoxInfo:
    """
    Core function in module. Returns a filled out BoxInfo object,
    which holds the system's total number of atoms, atom density,
    periodic box volume, periodic box unit vector magnitudes, and
    a boolean that is set to True if the system's water box likely
    has a bubble. To determine if the system's water box has a bubble,
    the structure provided by the structure_file parameter (and
    possibly
    the topology_file parameter) is wrapped into it's periodic image.
    The wrapped structure is partitioned into cubes and the local atom
    density is calculated for each cube. If any of these densities is
    less then the system's mean density multiplied by the cutoff, the
    system's water box likely has a bubble.

    Parameters
    ----------
    structure_file : str
        Name (including the path) of the system's structure file. Using
        this file, the system's structure is loaded as a mdtraj
        trajectory. If this file does not contain topology information,
        a topology file must be provided. h5, lh5, and pdb formats
        contain topology information and do not need to be accompanied
        by a topology file.

    mesh : tuple, optional
        Three element tuple of floats that defines the length's of the
        cubes' sides in the cubic partition. Decreasing
        this value will lead to increased accuracy, but will
        significantly increase the calculation time.
        Defaults to (1.5, 1.5, 1.5).

    box_vectors : ndarray, optional
        Box vectors that define the system's periodic box. If they are
        not provided, the box vectors are extracted from the system's
        mdtraj Trajectory object.

    cutoff : float, optional
        Cutoff value used to determine if the system's water box has a
        bubble. If any cube in the partition has a atom density less
        than this value multiplied by the system's atom density then
        the system is deemed to likely have a bubble. Increasing the
        cutoff value increases the chances of this calculation yielding
        a false negative, while having to low of a cutoff value
        increases the chances of getting a false positive. Defaults to
        0.5.

    topology_file : str, optional
        Name (and path) of system's topology file. A topology file is
        required if structure_file does not contain topology
        information.

    wrapped_structure_filename : str, optional
        If provided, the wrapped structure will be saved as PDB with
        this name. If left blank, the wrapped structure will not be
        saved.

    Returns
    -------
    box_info : BoxInfo
        BoxInfo class object that holds the properties of the system's
        water box.

    Notes
    -----
    The unit for all numerical float values is nanometers(nm) unless
    otherwise noted.

    """
    box_info = BoxInfo(
        mesh=mesh,
    )
    trajectory = load_mdtraj_trajectory(
        structure_file,
        topology_file=topology_file
    )
    if box_vectors is not None:
        box_info.box_vectors = box_vectors
    else:
        box_info.box_vectors = get_box_vectors(trajectory)
    box_info.box_type = get_box_type(box_info.box_vectors)
    trajectory.center_coordinates()
    unwrapped_coordinates = get_coordinates(trajectory)
    wrapped_coordinates = wrap_structure(unwrapped_coordinates,
                                         box_info)
    trajectory.xyz[0, :, :] = wrapped_coordinates[:, :]
    topology = trajectory.topology
    coordinates = get_coordinates(trajectory)
    box_info.number_of_atoms = len(coordinates)
    box_info.atom_density = box_info.number_of_atoms \
                            / box_info.volume
    if wrapped_structure_filename:
        trajectory.save_pdb(wrapped_structure_filename)
    bounds = coordinate_bounds(coordinates)
    box_info.cubic_partition_bounds = bounds
    nx, ny, nz = (
        get_num_subintervals(box_info, lower_bound=bounds[0, 0], upper_bound=bounds[0, 1]),
        get_num_subintervals(box_info, lower_bound=bounds[1, 0], upper_bound=bounds[1, 1]),
        get_num_subintervals(box_info, lower_bound=bounds[2, 0], upper_bound=bounds[2, 1]),
    )
    cubic_partition = get_cubic_partition(
        bounds, nx, ny, nz, box_info,
    )
    box_info.cubic_partition = get_cubic_partition_occupancy(
        coordinates=coordinates,
        num_x_subintervals=nx,
        num_y_subintervals=ny,
        num_z_subintervals=nz,
        hash_widths=hash_widths(bounds),
        bounds=bounds,
        cubic_partition=cubic_partition,
    )
    return box_info
