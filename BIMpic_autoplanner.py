import open3d.open3d as o3d
import numpy as np
import pickle

import voxel_labelling as vl

np.random.seed(0)

# Inputs
fn_setup = "autoplanner_setup.pkl"
fn_asBuilt = r"fused.ply"
fn_asPlanned = r"corner_flat_s6.obj"
fn_cameras = r"cameras.txt"
fn_images = r"images.txt"

voxel_size = 0.01
point_cloud_density = 0.00002  # surface area / number of points

# bounding box coordinates with about 0.1 m tolerance
# min_bound_coord = np.array([2.8, 7.55, 0.0])
min_bound_coord = np.array([7, 7.55, 0.0])
# max_bound_coord = np.array([8.0 ,12.5 ,0.4])
max_bound_coord = np.array([8.0, 8.5, 0.4])

# import as-built and as-planned models
# I should analyze this point cloud and determine how dense it is and use that
# value when making the point cloud for the as-planned model
pcd_b = o3d.io.read_point_cloud(fn_asBuilt)
print(pcd_b)

try:
    # elements = []
    # cameras = vl.camera()
    # images = []
    # pcd_p = o3d.geometry.PointCloud()
    # voxelGrid_p = o3d.geometry.VoxelGrid()
    # voxel_reference =
    with open(fn_setup) as f_setup:
        print "Loading autoplanner setup file"
        (point_cloud_density, elements, cameras,
         images, pcd_p, voxelGrid_p, voxel_reference) = pickle.load(f_setup)
except IOError:
    print "Importing BIM"
    elements, mesh_p = vl.import_BIM(fn_asPlanned, min_bound_coord,
                                     max_bound_coord, point_cloud_density)

    print "Importing COLMAP meta files"
    cameras = vl.import_cameras_file(fn_cameras)

    images = vl.import_images_file(fn_images)

    # Create voxel grid from element point clouds
    pcd_p = o3d.geometry.PointCloud()
    for element in elements:
        pcd_p = pcd_p + element['point_cloud']

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
    dump = [point_cloud_density, elements, cameras,
            images, pcd_p, voxelGrid_p, voxel_reference]
    with open(fn_setup, 'w') as f_setup:
        pickle.dump(dump, f_setup)

print "Creating labelled voxel"
voxel_labelled = voxel_reference.export_voxel_labelled(range(len(elements)))

# print "Creat test voxel"
# test_elements = range(0, len(elements), 2)
# voxel_label_test = voxel_reference.export_voxel_labelled(test_elements)

# print "Plotting labelled voxel"
# voxel_labelled.visualize(
#     "elements",
#     np.array([elements[k]['color'] for k in range(len(elements))]),
#     plot_geometry=[mesh_p]
# )

# print "Plotting test labelled voxel"
# voxel_label_test.visualize(
#     "elements",
#     np.array([elements[k]['color'] for k in test_elements]),
#     plot_geometry=[]
# )


print "Labeling blocked voxels in as-planned model."
voxel_labelled.create_planned_labels(cameras, images)

print "Labeling occupied voxels in as-built model."
voxel_labelled.create_built_labels(pcd_b)

voxel_labelled.visualize("built", plot_geometry=[pcd_b])

print('done!')
