import os
import numpy as np
import pytest
import bubblebuster

@pytest.mark.tryp_ben
def test_tryp_ben_num_atoms(
        tryp_ben,
        tryp_ben_num_atoms 
    ):
    assert tryp_ben.number_of_atoms == \
        tryp_ben_num_atoms 

@pytest.mark.tryp_ben
def test_tryp_ben_volume(
        tryp_ben,
        tryp_ben_volume
    ):
    assert round(tryp_ben.volume, 2) == \
        round(tryp_ben_volume, 2)
         

@pytest.mark.tryp_ben
def test_tryp_ben_dimensions(
        tryp_ben,
        tryp_ben_dimensions
    ):
    assert np.array_equal(
        tryp_ben.dimensions,
        tryp_ben_dimensions) == True

@pytest.mark.tryp_ben
def test_tryp_ben_box_type(
        tryp_ben,
        tryp_ben_box_type
    ):
    assert tryp_ben.box_type == \
        tryp_ben_box_type
         

@pytest.mark.tryp_ben
def test_tryp_ben_box_vectors(
        tryp_ben,
        tryp_ben_box_vectors
    ):
    assert np.array_equal(
        tryp_ben.box_vectors,
        tryp_ben_box_vectors) == True

@pytest.mark.tryp_ben
def test_tryp_ben_mesh(
        tryp_ben,
        tryp_ben_mesh
    ):
    assert tryp_ben.mesh == \
        tryp_ben_mesh
    
@pytest.mark.tryp_ben
def test_tryp_ben_cutoff(
        tryp_ben,
        tryp_ben_cutoff
    ):
    assert tryp_ben.cutoff == \
        tryp_ben_cutoff

@pytest.mark.tryp_ben
def test_tryp_ben_has_bubble(
        tryp_ben
    ):
    assert tryp_ben.has_bubble == False 

@pytest.mark.tryp_ben
def test_tryp_ben_cubic_partition_bounds(
        tryp_ben,
        tryp_ben_cubic_partition_bounds
    ):
    assert tryp_ben.cubic_partition_bounds._asdict() \
        == tryp_ben_cubic_partition_bounds

@pytest.mark.tryp_ben
def test_tryp_ben_atom_density(
        tryp_ben,
        tryp_ben_atom_density
    ):
    assert round(tryp_ben.atom_density, 2) == \
        round(tryp_ben_atom_density, 2)

@pytest.mark.tryp_ben
def test_tryp_ben_cube_atom_densities(
        tryp_ben,
        tryp_ben_cube_atom_densities
        ):
    cube_atom_densities = [
        cube.atom_density for cube 
        in tryp_ben.cubic_partition
    ]
    assert tryp_ben_cube_atom_densities == \
        cube_atom_densities 

@pytest.mark.tryp_ben
def test_tryp_ben_cube_volumes(
        tryp_ben,
        tryp_ben_cube_volumes
        ):
    cube_volumes = [
        cube.volume for cube 
        in tryp_ben.cubic_partition
    ]
    assert tryp_ben_cube_volumes == \
        cube_volumes
  
@pytest.mark.tryp_ben
def test_tryp_ben_mean_cube_atom_density(
        tryp_ben,
        tryp_ben_mean_cube_atom_density
        ):
    assert tryp_ben.mean_cube_atom_density == \
        tryp_ben_mean_cube_atom_density

@pytest.mark.structure_with_topology_file
def test_tryp_ben_with_topology_fie():
    assert bubblebuster.periodic_box_properties(
        "trypsin_benzamidine.inpcrd",
        topology_file="trypsin_benzamidine_topology_test.pdb"
    )

@pytest.mark.tryp_ben_with_box_vectors
def test_tryp_ben_with_box_vectors_volume(
        tryp_ben_with_box_vectors,
        tryp_ben_volume
        ):
    assert round(tryp_ben_with_box_vectors.volume, 2) == \
        round(tryp_ben_volume, 2)

@pytest.mark.tryp_ben_with_box_vectors
def test_tryp_ben_with_box_vectors_atom_density(
        tryp_ben_with_box_vectors,
        tryp_ben_atom_density
        ):
    assert round(tryp_ben_with_box_vectors.atom_density, 2) == \
        round(tryp_ben_atom_density, 2)

@pytest.mark.tryp_ben_with_box_vectors
def test_tryp_ben_with_box_vectors_cube_volumes(
        tryp_ben_with_box_vectors,
        tryp_ben_cube_volumes
        ):
    cube_volumes = [
        cube.volume for cube 
        in tryp_ben_with_box_vectors.cubic_partition
    ]
    assert tryp_ben_cube_volumes == \
        cube_volumes

@pytest.mark.tryp_ben_with_box_vectors
def test_tryp_ben_with_box_vectors_cube_atom_densities(
        tryp_ben_with_box_vectors,
        tryp_ben_cube_atom_densities
        ):
    cube_atom_densities = [
        cube.atom_density for cube 
        in tryp_ben_with_box_vectors.cubic_partition
    ]
    assert tryp_ben_cube_atom_densities == \
        cube_atom_densities 

@pytest.mark.tryp_ben_with_box_vectors
def test_tryp_ben_with_box_vectors_box_type(
        tryp_ben_with_box_vectors,
        tryp_ben_box_type
    ):
    assert tryp_ben_with_box_vectors.box_type == \
        tryp_ben_box_type

@pytest.mark.tryp_ben_with_box_vectors
def test_tryp_ben_with_box_vectors_has_bubble(
        tryp_ben_with_box_vectors,
    ):
    assert tryp_ben_with_box_vectors.has_bubble == \
        False

@pytest.mark.save_wrapped_structure
def test_save_wrapped_structure(
        tryp_ben_save_wrapped_structure,
        tmp_test_files_dir,
    ):
    assert os.path.exists(
        str(tmp_test_files_dir / 
        "trypsin_benzamidine_wrapped.pdb") == True
    )

@pytest.mark.tryp_ben_bubble
def test_tryp_ben_bubble_cube_atom_densities(
        tryp_ben_bubble,
        tryp_ben_bubble_cube_atom_densities
        ):
    cube_atom_densities = [
        cube.atom_density for cube 
        in tryp_ben_bubble.cubic_partition
    ]
    assert tryp_ben_bubble_cube_atom_densities == \
        cube_atom_densities 

@pytest.mark.tryp_ben_bubble
def test_tryp_ben_bubble_cube_volumes(
        tryp_ben_bubble,
        tryp_ben_bubble_cube_volumes
        ):
    cube_atom_volumes = [
        cube.volume for cube 
        in tryp_ben_bubble.cubic_partition
    ]
    assert tryp_ben_bubble_cube_volumes == \
        cube_atom_volumes 

@pytest.mark.tryp_ben_bubble
def test_tryp_ben_bubble_has_bubble(
        tryp_ben_bubble,
        ):
    assert tryp_ben_bubble.has_bubble == True

@pytest.mark.tryp_ben_bubble
def test_tryp_ben_bubble_with_box_vectors_has_bubble(
        tryp_ben_bubble_with_box_vectors,
        ):
    assert tryp_ben_bubble_with_box_vectors.has_bubble == True

@pytest.mark.large_mesh_bubble
def test_tryp_ben_bubble_mesh2_cube_volumes(
        tryp_ben_bubble_mesh2,
        tryp_ben_bubble_mesh2_cube_volumes
        ):
    cube_volumes = [
        cube.volume for cube 
        in tryp_ben_bubble_mesh2.cubic_partition
    ]
    assert tryp_ben_bubble_mesh2_cube_volumes == \
        cube_volumes


@pytest.mark.large_mesh_bubble
def test_tryp_ben_bubble_mesh2_cube_atom_densities(
        tryp_ben_bubble_mesh2,
        tryp_ben_bubble_mesh2_cube_atom_densities
        ):
    cube_atom_densities = [
        cube.atom_density for cube 
        in tryp_ben_bubble_mesh2.cubic_partition
    ]
    assert tryp_ben_bubble_mesh2_cube_atom_densities == \
        cube_atom_densities 

@pytest.mark.large_mesh_bubble
def test_tryp_ben_bubble_mesh2_has_bubble(
        tryp_ben_bubble_mesh2,
        ):
    assert tryp_ben_bubble_mesh2.has_bubble == False 

@pytest.mark.cubic_box
def test_cyclodextrin_cube_volumes(
        cyclodextrin,
        cyclodextrin_cube_volumes,
        ):
    cube_volumes = [
        cube.volume for cube 
        in cyclodextrin.cubic_partition
    ]
    assert cyclodextrin_cube_volumes == cube_volumes 

@pytest.mark.cubic_box
def test_cyclodextrin_cube_atom_densities(
        cyclodextrin,
        cyclodextrin_cube_atom_densities,
        ):
    cube_atom_densities = [
        cube.atom_density for cube 
        in cyclodextrin.cubic_partition
    ]
    assert cyclodextrin_cube_atom_densities == \
        cube_atom_densities 

@pytest.mark.cubic_box
def test_cyclodextrin_has_bubble(
        cyclodextrin,
        ):
    assert cyclodextrin.has_bubble == False

@pytest.mark.cubic_box
def test_cyclodextrin_box_type(
        cyclodextrin_with_box_vectors,
        cyclodextrin_box_type
    ):
    assert cyclodextrin_with_box_vectors.box_type == \
        cyclodextrin_box_type

@pytest.mark.cubic_box
def test_cyclodextrin_with_box_vectors_cube_volumes(
        cyclodextrin_with_box_vectors,
        cyclodextrin_cube_volumes
        ):
    cube_volumes = [
        cube.volume for cube 
        in cyclodextrin_with_box_vectors.cubic_partition
    ]
    assert cyclodextrin_cube_volumes == \
        cube_volumes

@pytest.mark.cubic_box
def test_cyclodextrin_with_box_vectors_cube_atom_densities(
        cyclodextrin_with_box_vectors,
        cyclodextrin_cube_atom_densities
        ):
    cube_atom_densities = [
        cube.atom_density for cube 
        in cyclodextrin_with_box_vectors.cubic_partition
    ]
    assert cyclodextrin_cube_atom_densities == \
        cube_atom_densities 

@pytest.mark.cubic_box
def test_cyclodextrin_with_box_vectors_box_type(
        cyclodextrin_with_box_vectors,
        cyclodextrin_box_type
    ):
    assert cyclodextrin_with_box_vectors.box_type == \
        cyclodextrin_box_type

@pytest.mark.cubic_box
def test_cyclodextrin_with_box_vectors_has_bubble(
        cyclodextrin_with_box_vectors,
    ):
    assert cyclodextrin_with_box_vectors.has_bubble == \
        False
