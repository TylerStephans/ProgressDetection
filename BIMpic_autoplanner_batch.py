import open3d.open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import pickle
import os
import errno
from datetime import datetime
from copy import deepcopy

import voxel_labelling as vl

# Inputs Jezero03
### fn_setup = r"./autoplanner_setups/jezero03_padnwall_1inVoxel_setup.pkl"
### fn_asPlanned = r"/home/tyler/MS_Thesis/models/landingPad_blastWall02_rotatedyfzu.obj"  # path for as-planned model
### fn_plan = r"/home/tyler/MS_Thesis/ProgressDetection/linear_plans/landingPad_blastWall_by8.csv"
fn_setup = r"./autoplanner_setups/foundation01_1inVoxel_setup.pkl"
fn_asPlanned = r"/home/tyler/MS_Thesis/models/foundation01_rotatedyfzu.obj"  # path for as-planned model
fn_plan = r"/home/tyler/MS_Thesis/ProgressDetection/linear_plans/foundation_01_courses.csv"

# Jezero03 what reconstructions to process
#camera_sets = pd.read_csv("./Jezero03_1875.csv")
# For foundation
camera_sets = pd.read_csv("./foundation01_array01_occlusions.csv")

# If true, then only import BIM and model data, and save to file
only_import = False
write_instructions = False
# Loop through every step in plan to populate prediction board for heatmap
plot_heatmap = True
visualize_voxel_labels = False
# Save prediction board to csv file
save_prediction_board = True
save_run_details = True

voxel_size = 0.0254  # width of voxel in meters
point_cloud_density = 10/voxel_size**2  # number of points / surface area

# Landing pad and blast wall bounds
### min_bound_coord = np.array([3.0, 3.0, -0.1])
### max_bound_coord = np.array([7.0, 7.0, 1.0])
# foundation01 bounds
min_bound_coord = np.array([2.88, 7.38, -0.02])
max_bound_coord = np.array([8.12, 12.58, 1.0])

for index, row in camera_sets.iterrows():
    # Reconstruct file paths for transform matrix and images
    set_setting = row['Setting']
    set_description = row['Description']
    set_prefix = row['Prefix']
    step_built = row['Step']

    # If prefix is nan change to empty string
    if not isinstance(set_prefix, basestring) and np.isnan(set_prefix):
        set_prefix = ""

    # This is here because all data sets had their seed set to 0 initially
    np.random.seed(0)

    output_name = set_setting + "_" + set_prefix + step_built
    asBuilt_path = set_setting + "_" + set_description + "/" + set_prefix + step_built
    # # fn_tf_matrix = (
    # #     r"/home/tbs5111/IRoCS_BIM2Robot/ProgressPredict/transformation_matrices/"
    # #     + location_date + "_" + set_description + "_" + reconstruction_settings
    # #     + "_to_FoamCorner.txt"
    # # )
    fn_tf_matrix = (
        r"/home/tyler/MS_Thesis/ProgressDetection/transformation_matrices/" +
        "identity.txt")
    pn = r"./" + set_setting + "_" + set_description + "/" + step_built + r"/"
    # All COLMAP files should be pulled from the dense model workspace.
    # This is because the sparse model within the dense model workspace is already
    # corrected for distortions, which this script does not.
    fn_asBuilt = (
        r"/home/tyler/MS_Thesis/reconstruction/" + 
        asBuilt_path + r"/dense/lowLgImg/fused.ply"
    )
    fn_cameras = (
        r"/home/tyler/MS_Thesis/reconstruction/" + 
        asBuilt_path + r"/sparse/high/aligned/cameras.bin"
    )
    fn_images = (
        r"/home/tyler/MS_Thesis/reconstruction/" + 
        asBuilt_path + r"/sparse/high/aligned/images.bin"
    )

    # Transformation matrix for transforming as-built model and image locations to BIM coordinates
    tf_matrix = vl.import_transformMatrix(fn_tf_matrix)

    print("Starting import... \n")
    if only_import:
        # Keep track of how much time has passed
        startTime = datetime.now()
    # import as-built and as-planned models
    # I should analyze this point cloud and determine how dense it is and use that
    # value when making the point cloud for the as-planned model
    pcd_b = o3d.io.read_point_cloud(fn_asBuilt)
    pcd_b.transform(tf_matrix)  # transform as-built model
    print(pcd_b)

    print("Importing COLMAP meta files")
    cameras = vl.import_cameras_file(fn_cameras)

    images = vl.import_images_file(fn_images, tf_matrix=tf_matrix)

    # Make directory if it doesn't already exist
    try:
        os.makedirs(pn)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    try:
        with open(fn_setup, 'r') as f_setup:
            print("Loading autoplanner setup file")
            (point_cloud_density, elements, voxel_reference) = pickle.load(f_setup)
    except IOError:
        print("Importing BIM")
        elements, mesh_p = vl.import_BIM(fn_asPlanned, min_bound_coord,
                                        max_bound_coord, point_cloud_density)

        print("Voxelizing BIM")
        voxel_reference = vl.voxelize_BIM(elements, voxel_size, min_bound_coord, max_bound_coord)

        print("Saving autoplanner setup file")
        dump = [point_cloud_density, elements, voxel_reference]
        with open(fn_setup, 'w') as f_setup:
            pickle.dump(dump, f_setup)

    if not only_import:
        # Keep track of how much time has passed
        startTime = datetime.now()

        print("Starting searching for current step...\n")
        check_next_step = True
        found_k_elements = False
        cur_step = np.NaN

        # Color map based on excel conditional formatting
        mymap = np.array([[248, 105, 107], [255, 235, 132], [99, 190, 123]])
        mymap = mymap/255.0
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", mymap)

        print("Importing linear plan")
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
            print("\n=========== Step %G ===========\n" % (step))
            print("Creating labelled voxel")
            voxel_labelled = voxel_reference.export_voxel_labelled(cur_elements)

            print("Labeling blocked voxels in as-planned model.")
            voxel_labelled.create_planned_labels(cameras, images)

            print("Labeling occupied voxels in as-built model.")
            voxel_labelled.create_built_labels(pcd_b)

            print("Predicting what elements are present in the as-built model.")
            elements_found, element_Ps = voxel_labelled.predict_progress()

            # Report if any of the elements placed in current step are completely
            # blocked.
            visible_elements = np.unique(
                voxel_labelled.elements[voxel_labelled.planned != "b"])
            if not np.all(np.isin(k_elements, visible_elements)):
                print("WARNING: Some expected elements are blocked or out of view...")

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
            # current step must have been the previous one checked!
            found_k_elements = np.all(np.isin(k_elements, elements_found))
            if found_k_elements and np.isnan(cur_step):
                cur_step = step + 1
                print("******** Current step is step %G ********" % (cur_step))
                step_found_time = datetime.now() - startTime
                print("Found after " + str(step_found_time))

                print("Construction elements found:")
                found_names = [elements[k]['name'] for k in elements_found]
                for name in found_names:
                    print(name)

                # Save current labeled voxels for plotting later
                if visualize_voxel_labels:
                    voxels_current_step = deepcopy(voxel_labelled)

                if not plot_heatmap:
                    check_next_step = False

            step -= 1

        if np.isnan(cur_step):
            print("No current step was found...")
            step_found_time = np.NaN

        plan_traversal_time = datetime.now() - startTime
        print("Total plan traversal took " + str(plan_traversal_time))

        if write_instructions:
            print("Writing instructions for material placement.")
            fn_instructions = output_name + "_NextInstr.txt"

            # Instructions for the robot with one list item for each object to place
            instructions = []
            if np.isnan(cur_step):
                instructions.append('Current step could not be determined...')
            elif cur_step > len(BIM_order):
                instructions.append('Construction was finished (no current step).')
            else:
                for k in BIM_order[cur_step - 1]:
                    instructions.append(elements[k]['material'])

            with open(pn + fn_instructions, "w") as f:
                for instruction in instructions:
                    f.write("%s\n" % instruction)

        # Save a file with information on inputs used and how long it took to move
        # through the progress detection
        if save_run_details:
            fn_details = output_name + "_details.txt"
            with open(pn + fn_details, "w") as f:
                f.write("Setup file: %s\n" % fn_setup)
                f.write("As-planned file: %s\n" % fn_asPlanned)
                f.write("Construction plan file: %s\n" % fn_plan)
                f.write("COLMAP input files:\n%s\n%s\n%s\n" %
                        (fn_asBuilt, fn_cameras, fn_images))
                f.write("Time to find current step: %s\n" % str(step_found_time))
                f.write("Time to check every step: %s\n" %
                        str(plan_traversal_time))
                f.write("Current state: %.0f\n" % (cur_step - 1))
                f.write("Current step: %.0f\n" % cur_step)

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
            df.to_csv(pn + output_name + ".csv", index=False)

        if visualize_voxel_labels and not np.isnan(cur_step):
            # For when the pcd_b is so large it makes navigating the gui difficult...
            box = o3d.utility.Vector3dVector(np.concatenate([min_bound_coord[np.newaxis, :]-10, max_bound_coord[np.newaxis, :]+10], axis=0))
            bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(box)

            # Adding coordinate frames to visualize images
            vis_images = []
            for k in range(len(images)):
                mesh_temp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                mesh_temp.rotate(images[k].dcm.T,)
                mesh_temp.translate(images[k].dcm.T.dot(-images[k].translation), relative=False)
                # mesh_temp.transform(tf_matrix)
                vis_images.append(mesh_temp)

            # plot_list = [pcd_b.crop(bounding_box)]
            plot_list = [pcd_b]
            plot_list.extend(vis_images)
            voxels_current_step.visualize("built", element_highlight=BIM_order[cur_step-2], plot_geometry=[pcd_b.crop(bounding_box)])

        if plot_heatmap:
            # index of last visible element in prediction board
            ind_first = np.where(~np.all(np.isnan(prediction_board), axis=1))[0][0]
            ind_last = np.where(~np.all(np.isnan(prediction_board), axis=1))[0][-1]
            # generate heatmap of element progress probabilities
            # NOTE: Antialiased is applied to try and address the quadmesh in
            # heatmap not being perfectly aligned with hlines and vlines. This
            # results in lines being visible between quads, which is also fixed by
            # allowing small linewidths set to their face colors.
            ax = sns.heatmap(prediction_board, cmap=cmap, vmin=0, vmax=1,
                            cbar_kws={'label': 'Progress Probability',
                                      'ticks': [0.0, 0.25, 0.5, 0.75, 1.0]},
                            antialiased=True, linecolor='face', linewidths=0.5)
            # add vertical lines for different steps
            ax.vlines(range(1, prediction_board.shape[1]), *ax.get_ylim(),
                    colors=[0.7]*3, linewidths=0.5)
            # label ticks by step, not array index
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.IndexFormatter(
                    range(prediction_board.shape[1]+1))
            )
            step_inds = np.cumsum([len(k_elems) for k_elems in BIM_order])
            ax.hlines(step_inds[:-1], *ax.get_xlim(),
                    colors=[0.5]*3, linewidths=0.5)
            plt.ylim([ind_first, ind_last + 1])
            plt.xlabel("Step")
            plt.ylabel("Construction Element")
            plt.title("Prediction Heatmap for Step " + str(cur_step-1))
            plt.savefig(pn + output_name + "_heatmap.png")
            # # # plt.show()
            plt.close()

    else:
        print("Import completed after " + str(datetime.now() - startTime))

    print('Done!')
