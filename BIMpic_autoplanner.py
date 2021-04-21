import open3d.open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pickle
import os
import errno

import voxel_labelling as vl

np.random.seed(0)

# Inputs
pn = r"./foundation_test/"  # directory for instructions
fn_instructions = r"testing.txt"
fn_setup = r"./autoplanner_setups/foundation_1inVoxel_setup.pkl"
fn_asBuilt = r"/home/tbs5111/IRoCS_BIM2Robot/SfM/foundation_01_courses_s04/dense/fused.ply"
fn_asPlanned = r"/home/tbs5111/IRoCS_BIM2Robot/Meshes/foundation.obj"
fn_cameras = r"/home/tbs5111/IRoCS_BIM2Robot/SfM/foundation_01_courses_s04/sparse/manual/cameras.txt"
fn_images = r"/home/tbs5111/IRoCS_BIM2Robot/SfM/foundation_01_courses_s04/sparse/manual/images.txt"
fn_plan = r"/home/tbs5111/IRoCS_BIM2Robot/ProgressPredict/linear_plans/foundation_01_courses.csv"

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

# # BIM_order = [[0, 1, 2],
# #              [5, 6, 7],
# #              [8, 9, 10],
# #              [11, 12, 3, 13, 4],
# #              [14, 20, 21],
# #              [22, 23, 24],
# #              [25, 15, 16],
# #              [17, 18, 19]]

# Each item in materials is a material. For each material, an element is said
# to be that material if material[0] is in its name.
materials = [["Standard_Brick", "standard_brick"],
             ["CMU", "cmu"]]

# After importing/generating the BIM and other files
continue_after_import = True

plot_heatmap = True

voxel_size = 0.0254  # width of voxel in meters
point_cloud_density = 10/voxel_size**2  # number of points / surface area

# bounding box coordinates with about 2 cm tolerance
# construction area bounds
# # min_bound_coord = np.array([2.88, 7.64, -0.02])
# # max_bound_coord = np.array([7.95, 12.4, 1.0])
# foundation bounds
min_bound_coord = np.array([2.88, 7.38, -0.02])
max_bound_coord = np.array([8.12, 12.58, 1.0])

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

    print "Voxelizing BIM"
    voxel_reference = vl.voxelize_BIM(elements, voxel_size, min_bound_coord, max_bound_coord)

    print "Saving autoplanner setup file"
    dump = [point_cloud_density, elements, voxel_reference]
    with open(fn_setup, 'w') as f_setup:
        pickle.dump(dump, f_setup)

print "Importing linear plan"
BIM_order = vl.import_LinearPlan(fn_plan, elements)


if continue_after_import:
    print "Starting searching for current step...\n"
    found_all_elements = False

    # # prediction_board = np.empty([len(elements), len(BIM_order)])
    # # prediction_board[:] = np.NaN
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])

    step = len(BIM_order)
    # Starting with the last step, loop backwards through the steps until
    # you've found a step where you observe every element you expect to. The
    # following step must then be the current step.
    while not found_all_elements and step > 0:
        cur_elements = [k_elem for k_elems in BIM_order[:step]
                        for k_elem in k_elems]
        k_elements = BIM_order[step-1]
        print "\n=========== Step %G ===========\n" % (step)
        print "Creating labelled voxel"
        voxel_labelled = voxel_reference.export_voxel_labelled(cur_elements)

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
        found_elements, element_Ps = voxel_labelled.predict_progress()

        # # elem_order = np.unique(voxel_labelled.elements)
        # # prediction_board[elem_order, step-1] = element_Ps

        # # voxel_labelled.visualize(
        # #     "elements",
        # #     np.array([elements[k]['color'] for k in range(len(elements))]))

        # # voxel_labelled.visualize("planned")

        # # voxel_labelled.visualize("built", plot_geometry=[pcd_b])

        # # # found_sorted = np.sort(found_elements)
        # # # expected_sorted = np.sort(expected_elements)

        # Making this comparison only tells me that I found everything I was
        # looking for. It doesn't tell me what I missed...
        # # # expected_found = found_sorted == expected_sorted

        # Found a step with all elements present, so the current step must
        # have been the last one checked!
        # # # found_all_elements = np.all(expected_found)
        if not np.all(np.isin(k_elements, expected_elements)):
            print "WARNING: Some expected elements are blocked..."
            # I think I should place a continue here?
        # Found a step where all elements for the kth step were found, so the
        # current step must have been the last one checked!
        found_all_elements = np.all(np.isin(k_elements, found_elements))
        step -= 1

    # # plt.imshow(prediction_board, cmap=cmap, interpolation='nearest')
    # # ax = sns.heatmap(prediction_board)
    # # plt.show()

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
