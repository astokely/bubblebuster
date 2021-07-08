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

from math import fabs
from math import floor
from math import ceil 
import numpy as np
import mdtraj
import mdtraj as md
from mdtraj.core.trajectory import Trajectory
from mdtraj.core.topology import Atom 
import mdtraj.core.topology as Topology 
from collections import namedtuple
from itertools import product 
from typing import List, Dict, Union, Any, Tuple, Optional, NamedTuple

class CubeInfo(object):
    """
    Stores properties of individual cubes belonging to a water box's 
    cubic partition.

    Parameters
    ----------
    index : int
        Cube's cubic partition index.
    bounds : ndarray, optional
        2D numpy array that holds the cube's x, y, and z coordinate
        bounds. The first array contains the x bounds, second 
        array contains the y bounds, and third array contains the z
        bounds. The first element in each array is the coordinate's lower
        bound, while the second element is the coordinate's upper bound. 
    volume : float, optional 
        Volume of the cube.  
    atom_density : float, optional 
        The cube's atom density, which is calculated by dividing
        the total number of atoms in the cube by the cube's volume.
    number_of_atoms : int, optional 
        Total number of atoms in the cube.
    atom_indices : list, optional 
        Indices of atoms in cube with respect to the system's mdtraj 
        Topology object.
    elements : list, optional 
        Element symbols for atoms cube.
    atom_names : list, optional
        Names of atoms in cube.
    residue_names : list, optional
        Names of residues that atoms in cube are part of.
    
    Attributes
    ----------
    index : int
        Cube's cubic partition index.
    bounds : ndarray, optional
        2D numpy array that holds the cube's x, y, and z coordinate
        bounds. The first array contains the x bounds, second 
        array contains the y bounds, and third array contains the z
        bounds. The first element in each array is the coordinate's lower
        bound, while the second element is the coordinate's upper bound. 
    volume : float, optional 
        Volume of the cube.  
    atom_density : float, optional 
        The cube's atom density, which is calculated by dividing
        the total number of atoms in the cube by the cube's volume.
    number_of_atoms : list, optional 
        Number of atoms in cube.
    atom_indices : list, optional 
        Indices of atoms in cube with respect to the system's mdtraj 
        Topology object.
    elements : list, optional 
        Element symbols for atoms cube.
    atom_names : list, optional
        Names of atoms in cube.
    residue_names : list, optional
        Names of residues that atoms in cube are part of.
        Total number of atoms in the cube.

    """
    def __init__(
            self,
            index: Optional[int] = None,
            bounds: Optional[np.ndarray] = None,
            volume: Optional[float] = None,
            atom_density: Optional[float] = None,
            number_of_atoms: Optional[int] = None,
            atom_indices: Optional[List] = None,
            elements: Optional[List[str]] = None,
            atom_names: Optional[List] = None,
            residue_names: Optional[List] = None,
            ) -> None:
        self.index = index
        self.bounds = bounds
        self.volume = volume
        self.number_of_atoms = number_of_atoms
        self.atom_density = atom_density
        self.atom_indices = atom_indices
        self.elements = elements
        self.atom_names = atom_names
        self.residue_names = residue_names


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
    atom_density : float, optional
        Water box's atom density, which calculated by dividing the
        total number of atoms in the box by it's volume.
    number_of_atoms : int, optional
        Total number of atoms in the water box.
    cubic_partition : list, optional
        To determine if the water box contains a bubble, the box is
        partitioned into cubes and the atom density for each cube is
        calculated. If the water box has a bubble, then the density of
        at least one cube will be significantly less then the atom 
        density of the water box as a whole. The properties of each
        cube in the partition are stored a CubeInfo object, and the
        water box's entire cubic partition is represented by a list
        of these CubeInfo objects. 
    mean_cube_atom_density : float, optional
        Mean atom density of the individual cubic partition cubes, which
        is calculated by dividing the summation of the individual 
        cube's atom densities by the total number of cubes in the 
        partition.    
    cubic_partition_bounds : ndarray, optional
        Numpy array containing the x, y, and z bounds of each cube in
        the box's cubic partition.
    has_bubble : bool, optional
        Set to True of the water box has a bubble. The water box is
        said to contain at least one bubble if any cube has a density
        less then the box's atom density multiplied by the cutoff value.
    cutoff : float, optional
        Value used to determine if the water box has a bubble. The
        context in which it is used, is explained in the has_bubble 
        parameter's description.
    mesh : float, optional
        Length of cube's sides in the box's cubic partition. A smaller
        mesh will increase the bubble calculation's confidence and the
        price of significantly decreasing the calculation's speed. Due
        to this accuracy/performance tradeoff, the mesh value must be 
        choosen vary carefully, especially if a large number of water
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
    cubic_partition : list
        To determine if the water box contains a bubble, the box is
        partitioned into cubes and the atom density for each cube is
        calculated. If the water box has a bubble, then the density of
        at least one cube will be significantly less then the atom 
        density of the water box as a whole. The properties of each
        cube in the partition are stored a CubeInfo object, and the
        water box's entire cubic partition is represented by a list
        of these CubeInfo objects. 
    mean_cube_atom_density : float, optional
        Mean atom density of the individual cubic partition cubes, which
        is calculated by dividing the summation of the individual 
        cube's atom densities by the total number of cubes in the 
        partition.    
    cubic_partition_bounds : ndarray, optional
        Numpy array containing the x, y, and z bounds of each cube in
        the box's cubic partition.
    has_bubble : bool, optional
        Set to True of the water box has a bubble. The water box is
        said to contain at least one bubble if any cube has a density
        less then the box's atom density multiplied by the cutoff value.
    cutoff : float, optional
        Value used to determine if the water box has a bubble. The
        context in which it is used, is explained in the has_bubble 
        parameter's description.
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
            cubic_partition: Optional[List[CubeInfo]] = None,
            mean_cube_atom_density: Optional[float] = None,
            cubic_partition_bounds: Optional[np.ndarray] = None,
            has_bubble: Optional[bool] = False,
            cutoff: Optional[float] = 0.5,
            mesh: Optional[float] = 1.0, 
            ) -> None:
        self.box_type = box_type
        self.box_vectors = box_vectors
        self.dimensions = dimensions
        self.volume = volume
        self.atom_density = atom_density
        self.number_of_atoms = number_of_atoms
        self.cubic_partition = cubic_partition
        self.mean_cube_atom_density = \
            mean_cube_atom_density
        self.cubic_partition_bounds = \
            cubic_partition_bounds
        self.has_bubble = has_bubble
        self.cutoff = cutoff
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
        Magnitude of the trajectory's period box unit vectors: a, b, and c.

    """
    a, b, c = md.utils.box_vectors_to_lengths_and_angles(
        *box_vectors
    )[:3]
    return np.array([a, b, c])

def get_coordinates(
        trajectory: Trajectory,
        ) -> np.ndarray:
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
    return [
        atom_coordinates for atom_coordinates 
        in trajectory.xyz[0]
    ]

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
    box by a distance less then or equal to three times the magnitude of
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
    return floor(atom_coordinates[coordinate_component_index] 
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
        trajectory: Trajectory,
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
        wrapped according to the system's periodic boundary conditions. 

    """

    box_vectors = box_info.box_vectors
    box_vector_dimensions = get_box_vector_dimensions(
        box_vectors
    )
    box_info.dimensions = box_vector_dimensions
    box_info.volume = get_periodic_box_volume(box_vectors) 
    coordinates = get_coordinates(trajectory)
    for atom_coordinates in coordinates:
        apply_pbc_to_coordinates(
            box_vector_dimensions, 
            atom_coordinates, 
            box_vectors,
            box_info.box_type
        )
    return trajectory

def wrap_structure(
        trajectory: Trajectory,
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
    zero, all of the wrapped atomic coordinates are in the same periodic
    image. wrap_structure is called recursively until the above condition is
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
        wrapped according to the system's periodic boundary conditions. 

    """
    apply_pbc(trajectory, box_info)
    coordinates = get_coordinates(trajectory)
    bounds = coordinate_bounds(coordinates)
    if (round(bounds.xmin, 2) == 0.0
            and round(bounds.ymin, 2) == 0.0 
            and round(bounds.zmin, 2) == 0.0
        ):
        return trajectory
    return wrap_structure(trajectory, box_info)

def coordinate_bounds(
        coordinates: np.ndarray,
        ) -> NamedTuple:
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
    : namedtuple
        Returns a namedtuple with the minimums and maximums for the 
        system's atoms x, y, and z coordinates. The namedtuple keys
        and values are shown in the below table.

        |=============================|
        |xmin    x coordinate minimum |
        |-----------------------------|
        |xmax    x coordinate maximum |
        |-----------------------------|
        |ymin    y coordinate minimum |
        |-----------------------------|
        |ymax    y coordinate maximum | 
        |-----------------------------|
        |zmin    z coordinate minimum |
        |-----------------------------|
        |zmax    z coordinate maximum | 
        |=============================|

    """
    bounds = namedtuple(
        'bounds', 
        [
            'xmin', 'xmax',
            'ymin', 'ymax',
            'zmin', 'zmax'
        ]
    )
    return bounds(
        *[bound for upper_lower in [
            (
                lambda coordinate: [min(coordinate), max(coordinate)]
            )
            (coordinate) for coordinate in list(zip(*coordinates))
        ] 
        for bound in upper_lower]
    ) 

def construct_partition(
        minimum: float,
        maximum: float,
        num_subintervals: int,
        ) -> np.ndarray:
    """
    Constructs an interval partition by dividing the interval with 
    bounds defined by the minimum and maxium parameters into evenly
    spaced sub-intervals. The number of sub-intervals is defined by 
    the num_subintervals parameter.
    
    Parameters
    ----------
    minimum : float
        Minimum value included in the partition.
    maximum : float
        Maximum value included in the partition.
    num_subintervals : int
        Number of sub-intervals used to construct the partition.

    Returns
    -------
    : ndarray
        Numpy array of the partition's sub-interval's upper and lower
        bounds.

    """
    delta = (abs(minimum) + abs(maximum)) / num_subintervals
    subinterval_bounds =  np.array([
        minimum + index * delta for index
        in range(num_subintervals + 1)
    ])
    return np.array([
        np.array([
            subinterval_bounds[index], 
            subinterval_bounds[index + 1]
        ]) 
        for index in range(num_subintervals)
    ])  

def num_subintervals(
        box_info: BoxInfo(),
        lower_bound: float,
        upper_bound: float,
        ) -> int:
    """
    Calculates the number of sub-intervals used to construct a partition
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
        with bounds defined by the lower_bound and upper_bound parameters
        and a mesh defined by the BoxInfo object's mesh attribute.
    """
    return ceil(
        (upper_bound - lower_bound) / box_info.mesh
    )
    

def construct_cubic_partition(
        coordinate_bounds: NamedTuple,
        box_info: BoxInfo,
        ) -> np.ndarray:
    """
    Constructs the cubic partition that is used to calculate local atom
    densities throughout the system. The properties of each cube in the
    partition are stored in a unique CubeInfo object.     

    Parameters
    ----------
    coordinate_bounds : namedtuple
        Namedtuple with the minimums and maximums for the 
        system's atoms x, y, and z coordinates.         

    box_info : BoxInfo
        BoxInfo class object that holds the properties of the system's
        water box. The BoxInfo mesh attribute is used in this function
        defines the uniform sub-interval length.

    Returns
    -------
    : ndarray
        3D numpy array that contains the x, y, and z upper and lower
        bounds that are used to define the indiviual cubes in the 
        partition.

    """
    num_x_subintervals = num_subintervals(
        box_info, 
        coordinate_bounds.xmin, 
        coordinate_bounds.xmax
    )
    num_y_subintervals = num_subintervals(
        box_info, 
        coordinate_bounds.ymin, 
        coordinate_bounds.ymax
    )
    num_z_subintervals = num_subintervals(
        box_info, 
        coordinate_bounds.zmin, 
        coordinate_bounds.zmax
    )
    xpart = construct_partition(
        coordinate_bounds.xmin, 
        coordinate_bounds.xmax, 
        num_x_subintervals
    )
    ypart = construct_partition(
        coordinate_bounds.ymin, 
        coordinate_bounds.ymax, 
        num_y_subintervals
    )
    zpart = construct_partition(
        coordinate_bounds.zmin, 
        coordinate_bounds.zmax, 
        num_z_subintervals
    )
    
    cubic_partition = np.array(list(product(
        *np.array([xpart, ypart, zpart], dtype=object)
    )))
    cube_info_objs_list = [
        CubeInfo(index=index, bounds=cube, atom_indices = []) for index, cube
        in enumerate(cubic_partition)
    ]
    setattr(box_info, "cubic_partition", cube_info_objs_list)
    return cubic_partition
    
def in_subinterval(
        val: float,
        subinterval_bounds: np.ndarray,
        ) -> bool:
    """
    Checks to see if an individual component of an atom's coordinate
    vector is in a sub-interval with bounds defined by the 
    subinterval_bounds parameter.

    Parameters
    ----------
    val : float
        Component of an atom's coordinate vector.
    subinterval_bounds : ndarray
        Numpy array that contains the sub-interval's upper and lower
        bounds

    Returns
    -------
    : bool
        True if the coordinate component is greater then or equal to
        the sub-intervals lower bound and less then or equal to the 
        sub-intervals upper bound. False otherwise.
        
    """
    lower_bound, upper_bound = subinterval_bounds 
    if val >= lower_bound:
        if val <= upper_bound:
            return True
    return False

def atom_in_cube(
        atom_coordinates: np.ndarray,
        cube: np.ndarray,
        ) -> bool:
    """
    Checks to see if an atom's coordinates are in the cube specified
    by the cube parameter. This is done by checking if the 
    individual x, y, and z components of the atom's coordinates lie 
    within the cube's x, y, and z bounds. If this is true, then the 
    atom's coordinates are in the provided cube. 

    Parameters
    ----------
    atom_coordinates : ndarray
        Cartesian coordinates of the atom.
    cube : ndarray
        2D numpy array with the cartesian coordinate upper and 
        lower bounds that define the cube.
    
    Returns
    ------- 
    : bool
        True if the atom x, y, and z coordinate components are greater
        then or equal to the cube's x, y and z lower bounds and less 
        then or equal to the cube's x, y and z upper bounds. 
        False otherwise.

    """
    x, y, z = atom_coordinates
    x_bounds, y_bounds, z_bounds = cube
    in_cube = sum(
        [
            in_subinterval(x, x_bounds),
            in_subinterval(y, y_bounds),
            in_subinterval(z, z_bounds)
        ]
    )
    if in_cube == 3:
        return True
    return False

def set_cubic_partition_atom_properties(
        cubic_partition: np.ndarray,
        coordinates: np.ndarray,
        topology: mdtraj.core.Topology 
        ) -> None:
    """
    Sets the number_of_atoms, elements, atom_names,
    residue_names, and atom_indices attributes for a 
    CubeInfo object

    Parameters
    ----------
    cubic_partition : list
        cubic_partition attribute of the BoxInfo class
        which is a list of CubeInfo objects.
    coordinates : ndarray
        Numpy array of the system's atomic coordinates.

    topology  : mdtraj.core.topology
        mdtraj topology object which holds important
        information about the system's atoms, residues,
        chains, and elements.
    Returns
    -------
    : None

    """
    for cube in cubic_partition:
        cube.number_of_atoms = 0
        cube.atom_indices = []
        cube.elements = []
        cube.atom_names = []
        cube.residue_names = []
    atoms = [atom for atom in topology.atoms]
    for index, atom_coordinates in enumerate(coordinates):
        for cube in cubic_partition:
            if atom_in_cube(atom_coordinates, cube.bounds):
                cube.atom_indices.append(index)
                cube.elements.append(
                    atoms[index].element.symbol
                )
                cube.atom_names.append(
                    atoms[index].name
                )
                cube.atom_names.append(
                    atoms[index].name
                )
                cube.residue_names.append(
                    atoms[index].residue.name
                )
                cube.number_of_atoms += 1    

def cube_volumes(
        cubic_partition: np.ndarray,
        ) -> np.ndarray:
    """
    Returns a 1D numpy array containing the volume of each cube in the 
    cubic partition.

    Parameters
    ----------
    cubic_partition : ndarray
        3D numpy array of the x, y, and z bounds that define the cubes
        in the periodic box cubic partition. The array format is shown
        below. 

        (((cube 0 x lower bound, cube 0 x upper bound),
          (cube 0 y lower bound, cube 0 y upper bound), 
          (cube 0 z lower bound, cube 0 z upper bound))
             .  . .   .     .     .   . .   .     .
             .  . .   .     .     .   . .   .     .
             .  . .   .     .     .   . .   .     .
         ((cube n x lower bound, cube n x upper bound),
          (cube n y lower bound, cube n y upper bound), 
          (cube n z lower bound, cube n z upper bound)))

        , where n is the number of cubes in the partition
    
    Returns
    -------
    : ndarray
        1D numpy array containing the volume of each cube in the cubic
        partition.
    """
    return np.array([
        np.prod([ 
            fabs(cube[0][0] - cube[0][1]), 
            fabs(cube[1][0] - cube[1][1]), 
            fabs(cube[2][0] - cube[2][1])
        ])
        for cube in cubic_partition
    ])

def atom_density_per_cube(
        cubic_partition: np.ndarray,
        coordinates: np.ndarray,
        box_info: BoxInfo,
        topology: mdtraj.core.Topology
        ) -> Dict:
    """
    Returns a dictionary containing the atom density for each cube.
    The dictionary keys are the cube names ("cube" + str(index))
    and the values are the per cube densities.

    Parameters
    ----------
    cubic_partition : ndarray
        3D numpy array of the x, y, and z bounds that define the cubes
        in the periodic box cubic partition. The array format is shown
        below. 

        (((cube 0 x lower bound, cube 0 x upper bound),
          (cube 0 y lower bound, cube 0 y upper bound), 
          (cube 0 z lower bound, cube 0 z upper bound))
             .  . .   .     .     .   . .   .     .
             .  . .   .     .     .   . .   .     .
             .  . .   .     .     .   . .   .     .
         ((cube n x lower bound, cube n x upper bound),
          (cube n y lower bound, cube n y upper bound), 
          (cube n z lower bound, cube n z upper bound)))

        , where n is the number of cubes in the partition

    coordinates : ndarray
        Numpy array of the system's atomic coordinates.

    box_info : BoxInfo
        BoxInfo class object that holds the properties of the system's
        water box. This function sets the atom_density, volume, and 
        number_of_atoms attributes for each CubeInfo object in BoxInfo's
        cubic_partition attribute, which is a list of CubeInfo objects. 

    topology  : mdtraj.core.topology
        mdtraj topology object which holds important
        information about the system's atoms, residues,
        chains, and elements.

    Returns
    -------
    atom_density_per_cube_dict : dict
        Dictionary containing the atom density for each cube.


    """
    set_cubic_partition_atom_properties(
        box_info.cubic_partition, 
        coordinates, 
        topology
    )
    volumes = cube_volumes(cubic_partition)
    atom_density_per_cube_dict = {}
    for cube in box_info.cubic_partition:
        box_info.cubic_partition[cube.index].atom_density = \
            atom_density_per_cube_dict[cube.index] = \
                cube.number_of_atoms / volumes[cube.index]
        box_info.cubic_partition[cube.index].volume = \
            volumes[cube.index]
    return atom_density_per_cube_dict

def mean_atom_density_per_cube(
        atom_density_per_cube_dict: Dict, 
        ) -> float:
    """
    Returns a 1D numpy array of per cube densities.

    Parameters
    ----------
    atom_density_per_cube : dict
        Dictionary containing the atom density for each cube.
        The dictionary keys are the cube names ("cube" + str(index))
        and the values are the per cube densities.

    Returns
    -------
    : float
        Returns a 1D numpy array of per cube densities.
    """
    return np.mean(
        np.array(list(atom_density_per_cube_dict.values()))
    )

def check_for_bubbles(
        box_info: BoxInfo,
        ) -> bool:
    """
    Checks for bubbles in the system's water box. If any of the
    individual cube atom densities is less then the system's mean atom
    density multiplied by the BoxInfo cutoff attribute the water box
    likely contains a bubble. The accuracy of this calculation depends
    on the number of cubes in the cubic partition, which can be controlled
    by adjusting the mesh parameter in the check_water_box_for_bubbles 
    function. 

    Parameters
    ----------
    box_info : BoxInfo
        BoxInfo class object that holds the properties of the system's
        water box. This function sets the atom_density, volume, and 
        number_of_atoms attributes for each CubeInfo object in BoxInfo's
        cubic_partition attribute, which is a list of CubeInfo objects. 

    Returns
    -------
    : bool
        True if any individual cube atom density is < then the system's
        mean atom density * box_info.cutoff.

    """
    cube_atom_densities = [
        cube.atom_density for cube in box_info.cubic_partition
    ]
    cube_atom_densities_below_cutoff = [
        density for density in cube_atom_densities 
            if density < (box_info.atom_density * box_info.cutoff)
    ]
    if cube_atom_densities_below_cutoff:
        box_info.has_bubble = True
        return True
    else:
        return False

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
        mesh: Optional[float] = 1.5,
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
    the structure provided by the structure_file parameter (and possibly
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

    mesh : float, optional
        Legnth of the cubes' sides in the cubic partition. Decreasing 
        this value will lead to increased accuracy, but will
        significantly increase the calculation time. Defaults to 1.5.

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
        cutoff=cutoff,
        mesh=mesh,
    )
    trajectory = load_mdtraj_trajectory(
        structure_file, 
        topology_file = topology_file
    )
    if box_vectors is not None:
        box_info.box_vectors = box_vectors
    else:
        box_info.box_vectors = get_box_vectors(trajectory) 
    box_info.box_type = get_box_type(box_info.box_vectors)
    trajectory.center_coordinates()
    wrap_structure(trajectory, box_info)
    topology = trajectory.topology
    coordinates = get_coordinates(trajectory)
    box_info.number_of_atoms = len(coordinates)
    box_info.atom_density = box_info.number_of_atoms \
        / box_info.volume 
    if wrapped_structure_filename:
        trajectory.save_pdb(wrapped_structure_filename)
    bounds = coordinate_bounds(coordinates)
    box_info.cubic_partition_bounds = bounds
    cubic_partition = construct_cubic_partition(bounds, box_info)
    atom_density = atom_density_per_cube(
        cubic_partition,
        coordinates, 
        box_info, 
        topology
    )
    mean_atom_density = mean_atom_density_per_cube(atom_density) 
    box_info.mean_cube_atom_density = mean_atom_density
    check_for_bubbles(box_info)
    return box_info 


















        

        
    
    
    
    
    
        
        
        









