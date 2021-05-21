import pytest
import shutil
import os
import bubblebuster
import numpy as np

@pytest.fixture(scope="session")
def tmp_test_files_dir(
        tmpdir_factory
        ):
    test_files_tmpdir_factory = \
        tmpdir_factory.mktemp('test_files')
    yield test_files_tmpdir_factory
    shutil.rmtree(str(test_files_tmpdir_factory))

@pytest.fixture(scope="session")
def tryp_ben():
    default = bubblebuster.periodic_box_properties(
        "trypsin_benzamidine.pdb",
    )
    return default

@pytest.fixture(scope="session")
def tryp_ben_with_box_vectors(
        tryp_ben_box_vectors
        ):
    box_vectors = bubblebuster.periodic_box_properties(
        "trypsin_benzamidine.pdb",
        box_vectors=tryp_ben_box_vectors
    )
    return box_vectors

@pytest.fixture(scope="session")
def tryp_ben_bubble():
    bubble = bubblebuster.periodic_box_properties(
        "trypsin_benzamidine_bubble.pdb",
    )
    return bubble 

@pytest.fixture(scope="session")
def tryp_ben_bubble_with_box_vectors(
        tryp_ben_bubble,
        ):
    bubble_box_vectors = bubblebuster.periodic_box_properties(
        "trypsin_benzamidine_bubble.pdb",
        box_vectors=tryp_ben_bubble.box_vectors
    )
    return bubble_box_vectors 

@pytest.fixture(scope="session")
def tryp_ben_bubble_mesh2():
    bubble = bubblebuster.periodic_box_properties(
        "trypsin_benzamidine_bubble.pdb",
        mesh=2.0
    )
    return bubble 

@pytest.fixture(scope="session")
def cyclodextrin():
    cyclodex = bubblebuster.periodic_box_properties(
        "1-butanol.pdb",
    )
    return cyclodex 

@pytest.fixture(scope="session")
def cyclodextrin_with_box_vectors(cyclodextrin):
    box_vectors = bubblebuster.periodic_box_properties(
        "1-butanol.pdb",
        box_vectors=cyclodextrin.box_vectors
    )
    return box_vectors 


@pytest.fixture
def tryp_ben_volume():
    return np.float32(210.56819)

@pytest.fixture
def tryp_ben_mean_cube_atom_density():
    return np.float64(107.28573505515618)

@pytest.fixture
def tryp_ben_atom_density():
    return np.float64(107.28116073011799)

@pytest.fixture
def tryp_ben_box_type():
    return "triclinic" 

@pytest.fixture
def tryp_ben_cubic_partition_bounds():
    return {
        "xmin" : np.float32(0.00028252602), 
        "xmax" : np.float32(6.4904194),
        "ymin" : np.float32(0.00019741058),
        "ymax" : np.float32(6.1201), 
        "zmin" : np.float32(0.0009970516),
        "zmax" : np.float32(5.2994204)
    } 

@pytest.fixture
def tryp_ben_num_atoms():
    return 22590

@pytest.fixture
def tryp_ben_mesh():
    return 1.5 

@pytest.fixture
def tryp_ben_cutoff():
    return 0.5 

@pytest.fixture
def tryp_ben_box_vectors():
    return np.array([
        [6.4913, 0., 0.],
        [-2.1636367, 6.1201024, 0.],
        [-2.1636367, -3.059775, 5.3003235]
    ], dtype=np.float32)

@pytest.fixture
def tryp_ben_dimensions():
    return np.array([6.4913, 6.4913, 6.4913], dtype=np.float32)

@pytest.fixture(scope="session")
def tryp_ben_volume():
    return np.float32(210.56819)

@pytest.fixture
def tryp_ben_mean_cube_atom_density():
    return np.float64(
        107.28573141591592
    )

@pytest.fixture
def tryp_ben_atom_density():
    return np.float64(107.28116073011799)

@pytest.fixture
def tryp_ben_box_type():
    return "triclinic" 

@pytest.fixture
def tryp_ben_cubic_partition_bounds():
    return {
        "xmin" : np.float32(0.00028252602), 
        "xmax" : np.float32(6.4904194),
        "ymin" : np.float32(0.00019741058),
        "ymax" : np.float32(6.1201), 
        "zmin" : np.float32(0.0009970516),
        "zmax" : np.float32(5.2994204)
    } 

@pytest.fixture
def tryp_ben_num_atoms():
    return 22590

@pytest.fixture
def tryp_ben_mesh():
    return 1.5 

@pytest.fixture
def tryp_ben_cutoff():
    return 0.5 

@pytest.fixture(scope='session')
def tryp_ben_box_vectors():
    return np.array([
        [6.4913, 0., 0.],
        [-2.1636367, 6.1201024, 0.],
        [-2.1636367, -3.059775, 5.3003235]
    ], dtype=np.float32)

@pytest.fixture
def tryp_ben_dimensions():
    return np.array([6.4913, 6.4913, 6.4913], dtype=np.float32)


@pytest.fixture(scope="session")
def tryp_ben_cube_atom_densities():
    return [
        np.float64(107.80814976278407),
        np.float64(86.4364901181793),
        np.float64(96.40993128566153),
        np.float64(117.30666516038619),
        np.float64(88.81111896757983),
        np.float64(107.80814976278407),
        np.float64(95.46007974590131),
        np.float64(103.05889206398301),
        np.float64(116.83173939050609),
        np.float64(111.13263015194481),
        np.float64(109.23292707242439),
        np.float64(107.33322399290395),
        np.float64(108.75800130254427),
        np.float64(103.53381783386311),
        np.float64(120.63114554954693),
        np.float64(96.88485705554163),
        np.float64(117.30666516038613),
        np.float64(105.90844668326359),
        np.float64(113.98218477122539),
        np.float64(94.03530243626095),
        np.float64(97.83470859530183),
        np.float64(120.63114554954693),
        np.float64(108.75800130254427),
        np.float64(122.05592285918725),
        np.float64(104.95859514350343),
        np.float64(106.85829822302385),
        np.float64(108.28307553266417),
        np.float64(90.23589627722015),
        np.float64(119.20636823990661),
        np.float64(106.38337245314375),
        np.float64(113.03233323146523),
        np.float64(113.03233323146523),
        np.float64(101.15918898446259),
        np.float64(110.18277861218459),
        np.float64(100.20933744470237),
        np.float64(105.43352091338353),
        np.float64(104.48366937362329),
        np.float64(116.35681362062593),
        np.float64(114.93203631098561),
        np.float64(94.98515397602117),
        np.float64(118.73144247002651),
        np.float64(89.76097050734003),
        np.float64(113.50725900134533),
        np.float64(108.75800130254427),
        np.float64(102.5839662941029),
        np.float64(114.93203631098565),
        np.float64(126.3302547881082),
        np.float64(94.5102282061411),
        np.float64(109.70785284230449),
        np.float64(113.03233323146523),
        np.float64(113.98218477122545),
        np.float64(83.11200972901855),
        np.float64(96.88485705554163),
        np.float64(103.53381783386311),
        np.float64(122.53084862906735),
        np.float64(95.46007974590131),
        np.float64(100.68426321458243),
        np.float64(96.88485705554159),
        np.float64(112.55740746158507),
        np.float64(106.3833724531437),
        np.float64(93.56037666638089),
        np.float64(128.70488363750874),
        np.float64(90.23589627722015),
        np.float64(108.28307553266417),
        np.float64(118.73144247002651),
        np.float64(112.08248169170503),
        np.float64(120.63114554954693),
        np.float64(125.38040324834799),
        np.float64(113.03233323146523),
        np.float64(92.61052512662067),
        np.float64(126.80518055798831),
        np.float64(113.98218477122545),
        np.float64(88.81111896757983),
        np.float64(110.18277861218459),
        np.float64(107.33322399290395),
        np.float64(122.05592285918725),
        np.float64(109.23292707242433),
        np.float64(102.10904052422275),
        np.float64(104.95859514350339),
        np.float64(112.08248169170497),
        np.float64(93.5603766663809),
        np.float64(107.33322399290398),
        np.float64(110.65770438206472),
        np.float64(105.90844668326366),
        np.float64(96.88485705554164),
        np.float64(133.45414133630982),
        np.float64(106.85829822302388),
        np.float64(107.33322399290398),
        np.float64(108.7580013025443),
        np.float64(99.25948590494218),
        np.float64(103.05889206398302),
        np.float64(85.4866385784191),
        np.float64(116.356813620626),
        np.float64(126.80518055798834),
        np.float64(101.1591889844626),
        np.float64(113.03233323146526),
        np.float64(95.93500551578141),
        np.float64(120.15621977966683),
        np.float64(105.90844668326365),
        np.float64(98.30963436518195),
    ]

@pytest.fixture(scope="session")
def tryp_ben_cube_volumes():
    return [
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077852),
        np.float64(2.105592207077852),
        np.float64(2.105592207077852),
        np.float64(2.105592207077852),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077852),
        np.float64(2.105592207077852),
        np.float64(2.105592207077852),
        np.float64(2.105592207077852),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077852),
        np.float64(2.105592207077852),
        np.float64(2.105592207077852),
        np.float64(2.105592207077852),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077852),
        np.float64(2.105592207077852),
        np.float64(2.105592207077852),
        np.float64(2.105592207077852),
        np.float64(2.1055922070778506),
        np.float64(2.1055922070778506),
        np.float64(2.1055922070778506),
        np.float64(2.1055922070778506),
        np.float64(2.1055922070778506),
        np.float64(2.1055922070778506),
        np.float64(2.1055922070778506),
        np.float64(2.1055922070778506),
        np.float64(2.1055922070778506),
        np.float64(2.1055922070778506),
        np.float64(2.1055922070778506),
        np.float64(2.1055922070778506),
        np.float64(2.1055922070778506),
        np.float64(2.1055922070778506),
        np.float64(2.1055922070778506),
        np.float64(2.1055922070778506),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
        np.float64(2.105592207077851),
    ]

@pytest.fixture
def tryp_ben_save_wrapped_structure(
        tmp_test_files_dir,
        ):
    save_wrapped_structure = \
        bubblebuster.periodic_box_properties(
            "trypsin_benzamidine.pdb",
            wrapped_structure_filename = \
            str(tmp_test_files_dir 
                / "trypsin_benzamidine_wrapped.pdb"
            ) 
    )
    yield save_wrapped_structure
    os.remove(
        str(tmp_test_files_dir 
            / "trypsin_benzamidine_wrapped.pdb"
        )
    )


@pytest.fixture
def tryp_ben_bubble_cube_atom_densities():
    return [
        np.float64(109.72765413508743),
        np.float64(106.87758519651373),
        np.float64(81.70197623911272),
        np.float64(121.60294137914453),
        np.float64(118.75287244057083),
        np.float64(83.60202219816186),
        np.float64(102.6024817886532),
        np.float64(135.85328607201302),
        np.float64(132.528205643677),
        np.float64(81.22696474935043),
        np.float64(101.17744731936632),
        np.float64(130.6281596846279),
        np.float64(127.30307925629195),
        np.float64(85.02705666744873),
        np.float64(96.90234391150581),
        np.float64(134.4282516027262),
        np.float64(145.35351586725872),
        np.float64(82.17698772887502),
        np.float64(96.42733242174353),
        np.float64(129.2031252153411),
        np.float64(132.05319415391475),
        np.float64(86.45209113673556),
        np.float64(87.40211411626012),
        np.float64(123.02797584843137),
        np.float64(130.62815968462792),
        np.float64(89.77717156507154),
        np.float64(97.85236689103036),
        np.float64(127.7780907460542),
        np.float64(129.20312521534103),
        np.float64(79.32691879030129),
        np.float64(96.90234391150577),
        np.float64(106.87758519651372),
        np.float64(126.82806776652967),
        np.float64(92.62724050364525),
        np.float64(90.72719454459613),
        np.float64(123.0279758484314),
        np.float64(129.67813670510336),
        np.float64(93.10225199340755),
        np.float64(97.37735540126809),
        np.float64(130.62815968462795),
        np.float64(131.10317117439018),
        np.float64(83.60202219816186),
        np.float64(97.85236689103036),
        np.float64(131.10317117439018),
        np.float64(142.0284354389227),
        np.float64(76.00183836196533),
        np.float64(108.77763115556287),
        np.float64(134.42825160272616),
        np.float64(135.85328607201302),
        np.float64(78.37689581077674),
        np.float64(101.65245880912862),
        np.float64(121.60294137914453),
        np.float64(136.80330905153764),
        np.float64(90.72719454459614),
        np.float64(80.27694176982591),
        np.float64(129.20312521534112),
        np.float64(105.45255072722694),
        np.float64(106.87758519651379),
        np.float64(74.10179240291622),
        np.float64(123.02797584843142),
        np.float64(123.50298733819363),
        np.float64(97.85236689103034),
        np.float64(99.75241285007947),
        np.float64(121.12792988938222),
        np.float64(126.82806776652961),
        np.float64(90.72719454459609),
        np.float64(98.32737838079262),
        np.float64(139.65337799011127),
        np.float64(129.67813670510327),
        np.float64(91.20220603435835),
        np.float64(97.85236689103031),
        np.float64(131.10317117439013),
        np.float64(124.9280218074805),
        np.float64(88.82714858554698),
        np.float64(76.47684985172761),
        np.float64(136.80330905153758),
        np.float64(116.85282648152169),
        np.float64(109.25264264532515),
        np.float64(53.20128685337573),
        np.float64(136.80330905153758),
        np.float64(121.6029413791445),
        np.float64(96.42733242174349),
        np.float64(95.9523209319812),
        np.float64(128.25310223581647),
        np.float64(132.05319415391472),
        np.float64(87.87712560602239),
        np.float64(110.2026656248497),
        np.float64(135.3782745822507),
        np.float64(123.97799882795589),
        np.float64(79.80193028006356),
        np.float64(97.37735540126803),
        np.float64(131.10317117439013),
        np.float64(131.57818266415248),
        np.float64(90.7271945445961),
        np.float64(106.87758519651373),
        np.float64(129.20312521534106),
        np.float64(121.60294137914453),
        np.float64(75.52682687220305),
        np.float64(62.22650515885911),
        np.float64(130.62815968462792)
    ]

@pytest.fixture
def tryp_ben_bubble_cube_volumes():
    return [
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723454),
        np.float64(2.1052122349723454),
        np.float64(2.1052122349723454),
        np.float64(2.1052122349723454),
        np.float64(2.1052122349723454),
        np.float64(2.1052122349723454),
        np.float64(2.1052122349723454),
        np.float64(2.1052122349723454),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723454),
        np.float64(2.1052122349723454),
        np.float64(2.1052122349723454),
        np.float64(2.1052122349723454),
        np.float64(2.1052122349723454),
        np.float64(2.1052122349723454),
        np.float64(2.1052122349723454),
        np.float64(2.1052122349723454),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972345),
        np.float64(2.105212234972345),
        np.float64(2.105212234972345),
        np.float64(2.105212234972345),
        np.float64(2.105212234972345),
        np.float64(2.105212234972345),
        np.float64(2.105212234972345),
        np.float64(2.105212234972345),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723468),
        np.float64(2.1052122349723468),
        np.float64(2.1052122349723468),
        np.float64(2.1052122349723468),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723463),
        np.float64(2.1052122349723468),
        np.float64(2.1052122349723468),
        np.float64(2.1052122349723468),
        np.float64(2.1052122349723468),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
        np.float64(2.105212234972346),
    ]

@pytest.fixture
def tryp_ben_bubble_cubic_partition_bounds():
    return {
        "xmin" : np.float32(0.0001077652), 
        "xmax" : np.float32(6.490708),
        "ymin" : np.float32(1.9550323e-05),
        "ymax" : np.float32(6.119722), 
        "zmin" : np.float32(6.2942505e-05),
        "zmax" : np.float32(5.2997866)
    } 

@pytest.fixture
def tryp_ben_bubble_mesh2_cube_atom_densities():
    return [
        np.float64(115.14278511837746),
        np.float64(72.50575379731491),
        np.float64(130.87516565930432),
        np.float64(133.61123184033502),
        np.float64(74.10179240291619),
        np.float64(126.99907190284407),
        np.float64(128.59511050844532),
        np.float64(69.54168210119826),
        np.float64(132.69920977999146),
        np.float64(136.1192925062799),
        np.float64(71.59373173697132),
        np.float64(128.82311602353124),
        np.float64(132.92721529507733),
        np.float64(69.54168210119826),
        np.float64(129.96314359896073),
        np.float64(132.69920977999143),
        np.float64(74.10179240291619),
        np.float64(123.80699469164152),
        np.float64(127.91109396318762),
        np.float64(71.82173725205722),
        np.float64(121.07092851061077),
        np.float64(123.8069946916415),
        np.float64(73.4177758576585),
        np.float64(127.45508293301586),
        np.float64(134.06724287050682),
        np.float64(72.50575379731491),
        np.float64(123.57898917655562),
        np.float64(133.15522081016323),
        np.float64(77.06586409903284),
        np.float64(129.73513808387483),
        np.float64(131.33117668947608),
        np.float64(62.701516648621386),
        np.float64(132.24319874981967),
        np.float64(113.31874099769028),
        np.float64(68.1736490106829),
        np.float64(130.41915462913252),
        np.float64(118.33486232957999),
        np.float64(81.85397991583665),
        np.float64(134.29524838559274),
        np.float64(130.4191546291325),
        np.float64(74.55780343308797),
        np.float64(134.97926493085043),
        np.float64(131.10317117439018),
        np.float64(77.29386961411873),
        np.float64(129.27912705370304),
        np.float64(123.8069946916415),
        np.float64(54.72132362061503),
        np.float64(124.26300572181331),
    ]

@pytest.fixture
def tryp_ben_bubble_mesh2_cube_volumes():
    return [
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
        np.float64(4.3858588228590545),
        np.float64(4.3858588228590545),
        np.float64(4.385858822859054),
    ]

@pytest.fixture(scope="session")
def cyclodextrin_cube_volumes():
    return [
        np.float64(1.8936947499841996),
        np.float64(1.8936947499841996),
        np.float64(1.8936947499842),
        np.float64(1.8936947499841996),
        np.float64(1.8936947499841996),
        np.float64(1.8936947499842),
        np.float64(1.8936947499841994),
        np.float64(1.8936947499841994),
        np.float64(1.8936947499841998),
        np.float64(1.8936947499841996),
        np.float64(1.8936947499841996),
        np.float64(1.8936947499842),
        np.float64(1.8936947499841996),
        np.float64(1.8936947499841996),
        np.float64(1.8936947499842),
        np.float64(1.8936947499841994),
        np.float64(1.8936947499841994),
        np.float64(1.8936947499841998),
        np.float64(1.8936947499841998),
        np.float64(1.8936947499841998),
        np.float64(1.8936947499842003),
        np.float64(1.8936947499841998),
        np.float64(1.8936947499841998),
        np.float64(1.8936947499842003),
        np.float64(1.8936947499841996),
        np.float64(1.8936947499841996),
        np.float64(1.8936947499842),
    ]

@pytest.fixture(scope="session")
def cyclodextrin_cube_atom_densities():
    return [
        np.float64(95.58034630529033),
        np.float64(112.47852907749636),
        np.float64(100.8610284216047),
        np.float64(111.42239265423349),
        np.float64(102.97330126813047),
        np.float64(106.14171053791908),
        np.float64(99.80489199834186),
        np.float64(102.44523305649905),
        np.float64(106.1417105379191),
        np.float64(108.25398338444485),
        np.float64(107.19784696118198),
        np.float64(98.74875557507895),
        np.float64(105.61364232628766),
        np.float64(124.62409794501944),
        np.float64(108.25398338444484),
        np.float64(105.08557411465624),
        np.float64(113.53466550075925),
        np.float64(106.1417105379191),
        np.float64(91.88386882387026),
        np.float64(106.1417105379191),
        np.float64(100.86102842160469),
        np.float64(113.00659728912778),
        np.float64(109.31011980770772),
        np.float64(99.8048919983418),
        np.float64(92.94000524713314),
        np.float64(103.5013694797619),
        np.float64(96.63648272855319),
    ]

@pytest.fixture(scope="session")
def cyclodextrin_box_type():
    return "cubic"
