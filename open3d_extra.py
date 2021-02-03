import open3d.open3d as o3d
import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# Calculate the area of a triangle mesh generated using open3d
# using the rule that the magnitude of the vector created by the
# cross product between two vectors is the area of the rhombus between them.
def triangle_mesh_area(mesh):
    area = 0.0
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    for tri in triangles:
        pt1 = vertices[tri[0], :]
        pt2 = vertices[tri[1], :]
        pt3 = vertices[tri[2], :]

        area = area + 0.5*np.linalg.norm(np.cross(pt2 - pt1, pt3 - pt1))

    return area


class voxel_label_base:
    def __init__(self, voxelGrid, min_bound=None, max_bound=None):
        voxels = voxelGrid.get_voxels()
        n_voxels = len(voxels)
        # number of element indices to keep when recording which elements
        # have points in the voxel
        n_elements_keep = 8

        self.grid_index = np.zeros([n_voxels, 3], dtype='int32')
        for k in range(n_voxels):
            self.grid_index[k, :] = voxels[k].grid_index

        # Indices of elements which intersect a voxel, sorted from most to
        # least points in voxel across rows.
        self.elements = -np.ones([n_voxels, n_elements_keep], dtype='int32')
        # Width of voxel (in meters)
        self.voxel_size = voxelGrid.voxel_size
        # not sure yet exactly...
        self.origin = voxelGrid.origin
        self.length = n_voxels
        # boundaries used when creating the original VoxelGrid
        self.min_bound = min_bound
        self.max_bound = max_bound

    # Given an open3d voxel, determine the voxels bounding box based on the
    # grid origin, the voxel size, and voxel's grid index
    def voxel_bounds(self, index=None, grid_index=None):
        if grid_index is None:
            grid_index = self.grid_index[index, :]

        local_min = grid_index*self.voxel_size
        local_max = (grid_index + 1)*self.voxel_size
        return local_min + self.origin, local_max + self.origin


class voxel_label_reference(voxel_label_base):
    pass

    # Fill out element labels using list of elements
    # For each voxel, loop through every element and check whether any points
    # fall inside that voxel. Save the indices of the elements from most to
    # least points inside the voxel.
    # NOTE: For future reference, the order of elements should not change!
    def create_element_labels(self, elements):
        for k in range(self.length):
            voxel_min_coord, voxel_max_coord = self.voxel_bounds(index=k)
            n_inside_prior = np.zeros((self.elements.shape[1],))

            k_element = 0
            for element in elements:
                pts_cur = np.asarray(element['point_cloud'].points)

                n_inside = np.sum(np.all(np.logical_and(
                    pts_cur >= voxel_min_coord, pts_cur <= voxel_max_coord
                    ), axis=1))

                more_inside = n_inside > n_inside_prior
                if (more_inside).any():
                    more_index = np.argmax(more_inside)

                    n_inside_prior[more_index+1:] = \
                        n_inside_prior[more_index:-1]
                    n_inside_prior[more_index] = n_inside

                    self.elements[k, more_index+1:] = \
                        self.elements[k, more_index:-1]
                    self.elements[k, more_index] = k_element
                k_element = k_element + 1

    def export_voxel_labelled(self, desired_elements):
        desired_elements = np.unique(desired_elements)

        elements_bool = np.zeros(self.elements.shape)
        for element in desired_elements:
            elements_bool = elements_bool + (self.elements == element)

        n_voxels = np.sum(np.any(elements_bool, axis=1))
        index_i = np.zeros((n_voxels,), dtype='int')
        index_j = np.zeros((n_voxels,), dtype='int')

        k = 0
        ind = 0
        for element_bool in elements_bool:
            if (element_bool).any():
                index_i[k] = ind
                index_j[k] = np.argmax(element_bool)
                k = k + 1
            ind = ind + 1

        voxel_labels = \
            voxel_labelled(self.grid_index[index_i, :],
                           self.elements[index_i, index_j],
                           self.voxel_size, self.origin, n_voxels,
                           self.min_bound, self.max_bound)

        return voxel_labels


class voxel_labelled(voxel_label_base):
    def __init__(self, grid_index, elements, voxel_size, origin, length,
                 min_bound, max_bound):
        self.grid_index = grid_index
        self.elements = elements
        self.voxel_size = voxel_size
        self.origin = origin
        self.length = length
        self.min_bound = min_bound
        self.max_bound = max_bound

        # label for as-planned model
        self.planned = np.array(['o']*length)
        # label for as-built model
        self.built = np.array(['e']*length)

    def visualize(self, label_to_plot, label_colors=None, plot_geometry=[]):
        pts = self.grid_index*self.voxel_size + \
            self.voxel_size/2.0 + self.origin
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        if label_to_plot == "elements":
            label_unique = np.unique(self.elements)
            # negative label means the voxel was not matched with an element
            label_unique = label_unique[label_unique >= 0]
            labels = self.elements

        if label_colors is None:
            label_colors = np.random.rand(len(label_unique), 3)

        # initialize all colors to black
        colors = np.ones([self.length, 3])
        for k in range(self.length):
            label_bool = label_unique == labels[k]
            if (label_bool).any():
                label_index = np.argmax(label_bool)
                colors[k, :] = label_colors[label_index, :]

        pcd.colors = o3d.utility.Vector3dVector(colors)

        voxelGrid = o3d.geometry.VoxelGrid.\
            create_from_point_cloud_within_bounds(pcd, self.voxel_size,
                                                  self.min_bound,
                                                  self.max_bound)
        plot_geometry.append(voxelGrid)
        o3d.visualization.draw_geometries(plot_geometry)

        # grid_shape = np.max(self.grid_index, axis=0) + 1
        # x, y, z = np.indices(grid_shape+1)
        # x = x*self.voxel_size + self.origin[0]
        # y = y*self.voxel_size + self.origin[1]
        # z = z*self.voxel_size + self.origin[2]

        # voxels = np.zeros(grid_shape, dtype='bool')
        # colors = np.ones([grid_shape[0], grid_shape[1], grid_shape[2], 4])

        # for k in range(self.length):
        #     ind = self.grid_index[k]
        #     voxels[ind[0], ind[1], ind[2]] = True
        #     label_bool = label_unique == labels[k]
        #     if (label_bool).any():
        #         label_index = np.argmax(label_bool)
        #         colors[ind[0], ind[1], ind[2], 0:3] = \
        #             label_colors[label_index, :]
        #     else:  # set alpha to 0 (transparent)
        #         colors[ind[0], ind[1], ind[2], 3] = 0

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(x, y, z, voxels, facecolors=colors, edgecolor='k')

        # plt.show()
        print("plotted!")

    # def create_built_labels(self, camera, images):


class camera:
    def __init__(self):
        self.camera_id = -1
        self.model = []
        self.width = 0.0
        self.height = 0.0
        self.focal_length = 0.0
        self.principal_pt = np.array([0, 0])

    def import_cameras_file(self, fn):
        with open(fn) as f_obj:
            lines_obj = f_obj.readlines()

        # remove comments
        k = 0
        non_whitespace = len(lines_obj[k]) - len(lines_obj[k].lstrip())
        while lines_obj[k][non_whitespace] == "#":
            k = k + 1
            non_whitespace = len(lines_obj[k]) - len(lines_obj[k].lstrip())
        lines_obj = lines_obj[k:]

        n_cameras = int(np.ceil(len(lines_obj)/2.0))

        self.camera_id = -np.ones((n_cameras,))
        self.width = np.zeros((n_cameras,))
        self.height = np.zeros((n_cameras,))
        self.focal_length = np.zeros((n_cameras,))
        self.principal_pt = np.zeros([n_cameras, 2])

        k = 0
        for k in range(0, len(lines_obj), 2):
            params = lines_obj[k].split(" ")
            self.camera_id[k/2] = int(params[0])
            self.model.append(params[1])
            self.width[k/2] = int(params[2])
            self.height[k/2] = int(params[3])
            self.focal_length[k/2] = float(params[4])
            self.principal_pt[k/2, :] = np.asarray(params[5:], dtype='float')

            k = k + 1

    # For a given image
    # def pt2image_project(self, pts, image):


class images:
    def __init__(self):
        self.image_id = -1
        self.quaternion = np.array([1, 0, 0, 0])
        self.translation = np.array([0, 0, 0, 0])
        self.camera_id = -1
        self.name = []

    def import_images_file(self, fn):
        with open(fn) as f_obj:
            lines_obj = f_obj.readlines()

        # remove comments
        k = 0
        non_whitespace = len(lines_obj[k]) - len(lines_obj[k].lstrip())
        while lines_obj[k][non_whitespace] == "#":
            k = k + 1
            non_whitespace = len(lines_obj[k]) - len(lines_obj[k].lstrip())
        lines_obj = lines_obj[k:]

        n_images = int(np.ceil(len(lines_obj)/2.0))

        self.image_id = -np.ones((n_images,))
        self.quaternion = np.zeros([n_images, 4])
        self.translation = np.zeros([n_images, 3])
        self.camera_id = -np.ones((n_images,))

        for k in range(0, len(lines_obj), 2):
            params = lines_obj[k].split(" ")
            self.image_id[k/2] = int(params[0])
            self.quaternion[k/2, :] = np.asarray(params[1:5], dtype='float')
            self.translation[k/2, :] = np.asarray(params[5:8], dtype='float')
            self.camera_id[k/2] = int(params[8])
            self.name.append(params[9][:-2])

            k = k + 1
