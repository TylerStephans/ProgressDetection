import open3d.open3d as o3d
import numpy as np
import copy

import open3d_extra as o3de

np.random.seed(0)

# Inputs
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

mesh_p = o3d.io.read_triangle_mesh(fn_asPlanned)
print(mesh_p)

# generate numpy array of vertices from as-planned model
vert_p = np.asarray(mesh_p.vertices)

# parse .obj file to find indices of named objects (building elements)
print "Parsing *.obj file..."
elements = []

with open(fn_asPlanned) as f_obj:
    lines_obj = f_obj.readlines()

n_elements = len([line for line in lines_obj if line.startswith('o ')])

last_tag = 'v'
k = 0
for line in lines_obj:
    if line.startswith('o '):
        elements.append(dict(name=line[2:-1]))

        last_tag = 'o'

    if line.startswith('v '):
        vert_cur = np.asarray(line[2:-1].split(' ')).astype('float')
        diff_cur = vert_cur - vert_p[k, :]
        if (np.abs(diff_cur) > 1E-5).any():
            print("Warning: vertex %G differed by %s" %
                  (k, np.array2string(diff_cur)))

        if last_tag == 'o':
            elements[-1]['v_index'] = [k]
        else:
            elements[-1]['v_index'].append(k)

        last_tag = 'v'
        k = k + 1

print "Importing COLMAP meta files"
cameras = o3de.import_cameras_file(fn_cameras)

images = o3de.import_images_file(fn_images)

print "Generating point clouds and removing unwanted elements"
# Loop through every named object from the obj in reverse order, remove onces
# which fall outside the specified bounds, and save point clouds of the ones
# that don't
for k in range(len(elements)-1, -1, -1):
    # Remove every other element from the triangle mesh by remove their
    # vertices
    to_remove = range(k) + range(k+1, len(elements))
    vert_remove = [vert for index in to_remove for vert in
                   elements[index]["v_index"]]
    mesh_temp = copy.deepcopy(mesh_p)
    mesh_temp.remove_vertices_by_index(vert_remove)

    vert_temp = np.asarray(mesh_temp.vertices)
    # If the current element is within the bounds of interest
    if ((vert_temp >= min_bound_coord).all()
            and (vert_temp <= max_bound_coord).all()):
        # Calculate the area of the desired element.
        # The area is used to ensure every element has a mostly equally dense
        # point cloud.
        n_pts = int(o3de.triangle_mesh_area(mesh_temp)/point_cloud_density)

        # Add color for visual differentiation
        elements[k]['color'] = np.random.rand(3)

        print("Saved %s as point cloud with %G points."
              % (elements[k]["name"], n_pts))
        pcd_temp = mesh_temp.sample_points_poisson_disk(n_pts)
        elements[k]['point_cloud'] = \
            pcd_temp.paint_uniform_color(elements[k]['color'])
    else:
        del elements[k]

# Create voxel grid from element point clouds
pcd_p = o3d.geometry.PointCloud()
for element in elements:
    pcd_p = pcd_p + element['point_cloud']

voxelGrid_p = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
    pcd_p, voxel_size, min_bound_coord, max_bound_coord
)
print(voxelGrid_p)
# o3d.visualization.draw_geometries([voxelGrid_p])
# o3d.visualization.draw_geometries([pcd_b, pcd_p])

# Convert voxel grid into custom format to save labels
voxel_reference = o3de.voxel_label_reference(voxelGrid_p,
                                             min_bound_coord, max_bound_coord)

# o3d.visualization.draw_geometries([voxelGrid_p, pcd_b])

print "Labelling elements in reference voxel grid."
voxel_reference.create_element_labels(elements)

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

voxel_labelled.visualize("planned")

print('done!')
