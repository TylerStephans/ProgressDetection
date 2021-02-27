import open3d.open3d as o3d
import numpy as np
import pickle
import os
import errno

import voxel_labelling as vl

np.random.seed(0)

# Inputs
pn = r"./corner_n_CMU_10/"  # directory for instructions
fn_instructions = r"corner_n_CMU_stest.txt"
fn_setup = r"./autoplanner_setups/corner_n_CMU_total_setup.pkl"
fn_asBuilt = r"/home/tbs5111/IRoCS_BIM2Robot/SfM/corner_n_CMU_s6_10/dense/fused.ply"
fn_asPlanned = r"/home/tbs5111/IRoCS_BIM2Robot/Meshes/corner_n_CMU_total.obj"
fn_cameras = r"/home/tbs5111/IRoCS_BIM2Robot/SfM/corner_n_CMU_s6_10/sparse/manual/cameras.txt"
fn_images = r"/home/tbs5111/IRoCS_BIM2Robot/SfM/corner_n_CMU_s6_10/sparse/manual/images.txt"

# Order that elements are supposed to be placed. Each item is a step, and each
# step must consist of a list of element indices.
# Brick corner only
# BIM_order = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12,
#              3, 13, 4, 14, 15, 16, 17, 18, 19]
# Build first 3 levels of brick, then CMU, then finish brick
# Each piece is a step
# BIM_order = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 3, 13, 4, 14,
#              20, 21, 22, 23, 24, 25,
#              15, 16, 17, 18, 19]
# BIM_order = [[k] for k in BIM_order]

BIM_order = [[0, 1, 2],
             [5, 6, 7],
             [8, 9, 10],
             [11, 12, 3, 13, 4],
             [14, 20, 21],
             [22, 23, 24],
             [25, 15, 16],
             [17, 18, 19]]

# Each item in materials is a material. For each material, an element is said
# to be that material if material[0] is in its name.
materials = [["Standard_Brick", "standard_brick"],
             ["CMU", "cmu"]]

# After importing/generating the BIM and other files
continue_after_import = True

voxel_size = 0.01  # width of voxel in meters
point_cloud_density = 50000  # number of points / surface area

# bounding box coordinates with about 2 cm tolerance
min_bound_coord = np.array([2.88, 7.64, -0.02])
# min_bound_coord = np.array([7, 7.55, 0.0])
max_bound_coord = np.array([7.95, 12.4, 1.0])
# max_bound_coord = np.array([8.0, 8.5, 0.4])

print "Starting import... \n"
# import as-built and as-planned models
# I should analyze this point cloud and determine how dense it is and use that
# value when making the point cloud for the as-planned model
pcd_b = o3d.io.read_point_cloud(fn_asBuilt)
print(pcd_b)

print "Importing COLMAP meta files"
cameras = vl.import_cameras_file(fn_cameras)

images = vl.import_images_file(fn_images)

# Make directory if it doesn't already exist
try:
    os.makedirs(pn)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    with open(fn_setup, 'r') as f_setup:
        print "Loading autoplanner setup file"
        (point_cloud_density, elements, voxel_reference) = pickle.load(f_setup)
except IOError:
    print "Importing BIM"
    elements, mesh_p = vl.import_BIM(fn_asPlanned, min_bound_coord,
                                     max_bound_coord, point_cloud_density)

    # Create voxel grid from element point clouds
    pcd_p = o3d.geometry.PointCloud()
    for element in elements:
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(element['point_cloud'])
        pcd_p += pcd_temp.paint_uniform_color(element['color'])

    voxelGrid_p = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd_p, voxel_size, min_bound_coord, max_bound_coord
    )
    print(voxelGrid_p)

    # Convert voxel grid into custom format to save labels
    voxel_reference = vl.voxel_label_reference(voxelGrid_p,
                                               min_bound_coord,
                                               max_bound_coord)

    # o3d.visualization.draw_geometries([voxelGrid_p, pcd_b])

    print "Labelling elements in reference voxel grid."
    voxel_reference.create_element_labels(elements)

    print "Saving autoplanner setup file"
    dump = [point_cloud_density, elements, voxel_reference]
    with open(fn_setup, 'w') as f_setup:
        pickle.dump(dump, f_setup)


if continue_after_import:
    print "Starting searching for current step...\n"
    found_all_elements = False
    step = len(BIM_order)
    # Starting with the last step, loop backwards through the steps until
    # you've found a step where you observe every element you expect to. The
    # following step must then be the current step.
    while not found_all_elements and step > 0:
        cur_elements = [k_elem for k_elems in BIM_order[:step]
                        for k_elem in k_elems]
        print "\n=========== Step %G ===========\n" % (step)
        print "Creating labelled voxel"
        voxel_labelled = voxel_reference.export_voxel_labelled(cur_elements)

        # print "Plotting labelled voxel"
        # voxel_labelled.visualize(
        #     "elements",
        #     np.array([elements[k]['color'] for k in range(len(elements))]),
        #     plot_geometry=[pcd_b]
        # )

        print "Labeling blocked voxels in as-planned model."
        voxel_labelled.create_planned_labels(cameras, images)

        # Expect to observe any element with at least one voxel not labelled
        # as blocked. This relies on there being enough cameras to reconstruct
        # every element visible from the images.
        expected_elements = np.unique(
            voxel_labelled.elements[voxel_labelled.planned != "b"])

        print "Labeling occupied voxels in as-built model."
        voxel_labelled.create_built_labels(pcd_b)

        print "Predicting what elements are present in the as-built model."
        found_elements = voxel_labelled.predict_progress()

        # voxel_labelled.visualize(
        #     "elements",
        #     np.array([elements[k]['color'] for k in range(len(elements))]))

        # voxel_labelled.visualize("planned")

        # voxel_labelled.visualize("built", plot_geometry=[pcd_b])

        found_sorted = np.sort(found_elements)
        expected_sorted = np.sort(expected_elements)

        # Making this comparison only tells me that I found everything I was
        # looking for. It doesn't tell me what I missed...
        expected_found = found_sorted == expected_sorted

        # Found a step with all elements present, so the current step must
        # have been the last one checked!
        found_all_elements = np.all(expected_found)
        step -= 1

    cur_step = step + 2

    print "Construction elements found:"
    found_names = [elements[k]['name'] for k in found_elements]
    for name in found_names:
        print name

    print "Current step is step %G." % (cur_step)
    # Missing elements is currently broken...
    # expected_names = [elements[k]['name'] for k in missing_elements]
    # for name in expected_names:
    #     print name

    voxel_labelled.visualize("built", plot_geometry=[pcd_b])

    print "Writing instructions for material placement."
    # Instructions for the robot with one list item for each object to place
    instructions = []
    for k in BIM_order[cur_step - 1]:
        cur_material = [material[1] for material in materials
                        if material[0] in elements[k]['name']]
        instructions.append(cur_material[0])

    with open(pn + fn_instructions, "w") as f:
        for instruction in instructions:
            f.write("%s\n" % instruction)


print('Done!')
