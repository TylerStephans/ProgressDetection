import open3d.open3d as o3d
import numpy as np
import pandas as pd
import pickle

import voxel_labelling as vl

if __name__ == '__main__':
    np.random.seed(0)

    # Inputs Physical Demo
    ### min_bound_coord = np.array([2.88, 7.38, -0.02])
    ### max_bound_coord = np.array([8.12, 12.58, 1.0])
    # Landing pad and blast wall bounds
    ### min_bound_coord = np.array([3.0, 3.0, -0.1])
    ### max_bound_coord = np.array([7.0, 7.0, 1.0])
    # foundation01 bounds
    min_bound_coord = np.array([2.5, 5.5, -0.1])
    max_bound_coord = np.array([6.5, 6.5, 2])

    # Inputs
    fn_setup = r"./autoplanner_setups/CMU_straightWall.pkl"
    fn_asPlanned = r"/home/tyler/IRoCS_BIM2Robot/models/CMU_straightWall.obj"  # path for as-planned model
    fn_plan = r"/home/tyler/IRoCS_BIM2Robot/ProgressDetection/linear_plans/CMU_straightWall.csv"

    # step = 19
    step = 20
    step_name = "11"  # name of step files to input
    pn = r"./test_20220613/" + step_name + r"/"  # directory for outputs
    output_name = "test_20220613_" + step_name + ""  # base name for output files
    # inputs from COLMAP
    asBuilt_name = "" + step_name
    fn_tf_matrix = (
        r"/home/tyler/IRoCS_BIM2Robot/ProgressDetection/transformation_matrices/" +
        "tf_11.txt")
    fn_asBuilt = (
        r"/home/tyler/IRoCS_BIM2Robot/reconstruction/test_20220613/" +
        asBuilt_name + r"/dense/high/fused.ply"
    )
    fn_cameras = (
        r"/home/tyler/IRoCS_BIM2Robot/reconstruction/test_20220613/" +
        asBuilt_name + r"/sparse/extreme/0/cameras.bin"
    )
    fn_images = (
        r"/home/tyler/IRoCS_BIM2Robot/reconstruction/test_20220613/" +
        asBuilt_name + r"/sparse/extreme/0/images.bin"
    )

    print("Starting import... \n")
    # Transformation matrix for transforming as-built model and image locations to BIM coordinates
    # for FoamBricks_corner10.obj
    tf_matrix = vl.import_transformMatrix(fn_tf_matrix)
    # import as-built and as-planned models
    # I should analyze this point cloud and determine how dense it is and use that
    # value when making the point cloud for the as-planned model
    pcd_b = o3d.io.read_point_cloud(fn_asBuilt)
    pcd_b.transform(tf_matrix)  # transform as-built model
    print(pcd_b)

    print("Importing COLMAP meta files")
    cameras = vl.import_cameras_file(fn_cameras)

    images = vl.import_images_file(fn_images, tf_matrix)

    with open(fn_setup, 'r') as f_setup:
        print("Loading autoplanner setup file")
        (point_cloud_density, elements, voxel_reference) = pickle.load(f_setup)

    print("Importing linear plan")
    BIM_order = vl.import_LinearPlan(fn_plan, elements)

    print("Projecting all voxels onto images...")
    voxel_reference.project_voxels_parallel(cameras, images)

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

    # print("Plotting labelled voxel"
    voxel_labelled.visualize(
        "elements",
        np.array([elements[k]['color'] for k in cur_elements]),
        plot_geometry=[]
    )

    # voxel_labelled.visualize("planned")
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

    # plot_list = [pcd_b]
    plot_list = [pcd_b.crop(bounding_box)]
    plot_list.extend(vis_images)
    #voxel_labelled.visualize("planned")
    #voxel_labelled.visualize("built", plot_geometry=plot_list)
    voxel_labelled.visualize("built", element_highlight=k_elements, plot_geometry=plot_list)
