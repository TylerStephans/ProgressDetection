import open3d.open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import pickle
import os
import errno

import voxel_labelling as vl

np.random.seed(0)

# Inputs
pn = r"./foundation_test/"  # directory for instructions
fn_instructions = r"testing.txt"
fn_setup = r"./autoplanner_setups/foundation_1inVoxel_setup_test.pkl"
fn_asPlanned = r"/home/tbs5111/IRoCS_BIM2Robot/Meshes/foundation.obj"  # path for as-planned model
fn_plan = r"/home/tbs5111/IRoCS_BIM2Robot/ProgressPredict/linear_plans/foundation_01_courses.csv"
asBuilt_name = "foundation_01_courses_s53"

fn_asBuilt = (
    r"/home/tbs5111/IRoCS_BIM2Robot/SfM/" +
    asBuilt_name + r"/dense/fused.ply"
)
fn_cameras = (
    r"/home/tbs5111/IRoCS_BIM2Robot/SfM/" +
    asBuilt_name + r"/sparse/manual/cameras.txt"
)
fn_images = (
    r"/home/tbs5111/IRoCS_BIM2Robot/SfM/" +
    asBuilt_name + r"/sparse/manual/images.txt"
)

# If true, then only import BIM and model data, and save to file
only_import = False
# Loop through every step in plan to populate prediction board for heatmap
plot_heatmap = False
# Save prediction board to csv file
save_prediction_board = False

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

if not only_import:
    print "Starting searching for current step...\n"
    check_next_step = True
    found_k_elements = False
    cur_step = np.NaN

    # Color map based on excel conditional formatting
    mymap = np.array([[248, 105, 107], [255, 235, 132], [99, 190, 123]])
    mymap = mymap/255.0
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", mymap)

    print "Importing linear plan"
    BIM_order = vl.import_LinearPlan(fn_plan, elements)

    print("Projecting all voxels onto images...")
    voxel_reference.project_voxels(cameras, images)

    # Array where each column represents a step, and each row a construction
    # element. The values are the probability of that element having
    # progressed on that step.
    prediction_board = np.empty([len(elements), len(BIM_order)])
    prediction_board[:] = np.NaN

    step = len(BIM_order)
    # Starting with the last step, loop backwards through the steps until
    # you've found a step where you observe every element you expect to. The
    # following step must then be the current step.
    while check_next_step and step > 0:
        # All elements that exist for current step
        cur_elements = [k_elem for k_elems in BIM_order[:step]
                        for k_elem in k_elems]
        # Elements that should have been placed during current step
        k_elements = BIM_order[step-1]
        print "\n=========== Step %G ===========\n" % (step)
        print "Creating labelled voxel"
        voxel_labelled = voxel_reference.export_voxel_labelled(cur_elements)

        print "Labeling blocked voxels in as-planned model."
        voxel_labelled.create_planned_labels(cameras, images)

        print "Labeling occupied voxels in as-built model."
        voxel_labelled.create_built_labels(pcd_b)

        print "Predicting what elements are present in the as-built model."
        elements_found, element_Ps = voxel_labelled.predict_progress()

        # Report if any of the elements placed in current step are completely
        # blocked.
        visible_elements = np.unique(
            voxel_labelled.elements[voxel_labelled.planned != "b"])
        if not np.all(np.isin(k_elements, visible_elements)):
            print "WARNING: Some expected elements are blocked..."

        # Determine indices that will sort element probabilities to be in the
        # same order as prescribed in BIM_order
        cur_elements_unordered = np.unique(voxel_labelled.elements)
        index_reorder = np.argsort(np.argsort(cur_elements))
        index_sort = np.argsort(cur_elements_unordered)

        prediction_board[:len(cur_elements), step-1] = element_Ps[index_sort][index_reorder]

        # # voxel_labelled.visualize(
        # #     "elements",
        # #     np.array([elements[k]['color'] for k in range(len(elements))]))

        # # voxel_labelled.visualize("planned")

        # # voxel_labelled.visualize("built", plot_geometry=[pcd_b])

        # Found a step where all elements for the kth step were found, so the
        # current step must have been the last one checked!
        found_k_elements = np.all(np.isin(k_elements, elements_found))
        if found_k_elements and np.isnan(cur_step):
            cur_step = step + 1
            # # voxel_labelled.visualize("built", plot_geometry=[pcd_b])

            print "Construction elements found:"
            found_names = [elements[k]['name'] for k in elements_found]
            for name in found_names:
                print name

            print "******** Current step is step %G ********" % (cur_step)

            if not plot_heatmap:
                check_next_step = False

        step -= 1

    if plot_heatmap:
        # index of last visible element in prediction board
        ind_first = np.where(~np.all(np.isnan(prediction_board), axis=1))[0][0]
        ind_last = np.where(~np.all(np.isnan(prediction_board), axis=1))[0][-1]
        # generate heatmap of element progress probabilities
        ax = sns.heatmap(prediction_board, cmap=cmap,
                         cbar_kws={'label': 'Progress Probability'})
        # add vertical lines for different steps
        ax.vlines(range(1, prediction_board.shape[1]), *ax.get_ylim(),
                  colors=[0.7]*3, linewidths=0.5)
        # label ticks by step, not array index
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.IndexFormatter(
                range(prediction_board.shape[1]+1))
        )
        plt.ylim([ind_last + 1, ind_first])
        plt.xlabel("Step")
        plt.ylabel("Construction Element")
        plt.title("Prediction Heatmap for Step " + str(cur_step-1))
        plt.savefig(asBuilt_name + "_heatmap.png")
        plt.show()

    if save_prediction_board:
        # ordered list of element indices based on BIM_order
        ordered_elements = [k_elem for k_elems in BIM_order
                            for k_elem in k_elems]
        # ordered list of steps correlating to ordered_elements
        ordered_steps = [k_step + 1 for k_step in range(len(BIM_order))
                         for k_elem in BIM_order[k_step]]

        # create data frame of element ID, material, and step for each element
        element_properties = (
            [[elements[k]['ID'], elements[k]['material']]
             for k in ordered_elements]
        )
        for k in range(len(element_properties)):
            element_properties[k].append(ordered_steps[k])
        df_properties = pd.DataFrame(data=element_properties,
                                     columns=['element ID', 'material',
                                              'step'])

        # create data frame based on prediction board
        step_headers = ['s' + str(k)
                        for k in range(1, prediction_board.shape[1]+1)]
        df_probabilities = pd.DataFrame(data=prediction_board,
                                        columns=step_headers)
        # combine data frames and export to csv
        df = pd.concat([df_properties, df_probabilities], axis=1)
        df.to_csv(pn + r"/" + asBuilt_name + ".csv", index=False)

    print "Writing instructions for material placement."
    # Instructions for the robot with one list item for each object to place
    instructions = []
    for k in BIM_order[cur_step - 1]:
        instructions.append(elements[k]['material'])

    with open(pn + fn_instructions, "w") as f:
        for instruction in instructions:
            f.write("%s\n" % instruction)


print('Done!')
