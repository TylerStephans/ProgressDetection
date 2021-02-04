import open3d.open3d as o3d
import numpy as np
# from skimage.morphology import convex_hull_image
from scipy.spatial import ConvexHull, Delaunay
# import pickle

import matplotlib.pyplot as plt
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


# Import cameras.txt file from COLMAP and save a list of cameras.
# If there is only one cameras specified, then it just returns a camera.
def import_cameras_file(fn):
    with open(fn) as f_obj:
        lines_obj = f_obj.readlines()

    # remove comments
    k = 0
    non_whitespace = len(lines_obj[k]) - len(lines_obj[k].lstrip())
    while lines_obj[k][non_whitespace] == "#":
        k = k + 1
        non_whitespace = len(lines_obj[k]) - len(lines_obj[k].lstrip())
    lines_obj = lines_obj[k:]

    cameras = []

    for k in range(0, len(lines_obj), 2):
        cameras.append(camera())
        params = lines_obj[k].split(" ")
        cameras[k/2].camera_id = int(params[0])
        cameras[k/2].model = params[1]
        cameras[k/2].size = np.asarray((params[2:4]), dtype='int')
        cameras[k/2].focal_length = float(params[4])
        cameras[k/2].principal_pt = np.asarray(params[5:], dtype='float')

    if len(cameras) == 1:
        return cameras[0]
    else:
        return cameras


def convex_fill(pixels, size, only_index=False):
    hull = ConvexHull(pixels)
    deln = Delaunay(pixels[hull.vertices])

    # restrict to area bounded by max and min of convex hull
    y = np.arange(hull.min_bound[0], hull.max_bound[0])
    x = np.arange(hull.min_bound[1], hull.max_bound[1])
    idx = np.stack(np.meshgrid(y, x), axis=-1)

    # indices of pixels in convex hull
    out_idx = np.asarray(np.nonzero(deln.find_simplex(idx) + 1)).T
    out_idx = np.flip(out_idx, axis=1) + hull.min_bound.astype('int')
    if not only_index:
        out_image = np.zeros(size, dtype='u1')
        out_image[out_idx[:, 0], out_idx[:, 1]] = 1
        return out_image
    else:
        return out_idx


# Import an images.txt file from COLMAP and save a list of images
def import_images_file(fn):
    with open(fn) as f_obj:
        lines_obj = f_obj.readlines()

    # remove comments
    k = 0
    non_whitespace = len(lines_obj[k]) - len(lines_obj[k].lstrip())
    while lines_obj[k][non_whitespace] == "#":
        k = k + 1
        non_whitespace = len(lines_obj[k]) - len(lines_obj[k].lstrip())
    lines_obj = lines_obj[k:]

    images = []

    for k in range(0, len(lines_obj), 2):
        images.append(image())
        params = lines_obj[k].split(" ")
        images[k/2].image_id = int(params[0])
        images[k/2].quaternion = np.asarray(params[1:5], dtype='float')
        images[k/2].translation = np.asarray(params[5:8], dtype='float')
        images[k/2].camera_id = int(params[8])
        images[k/2].name = params[9][:-2]
        images[k/2].calc_rotation_matrix()

    return images


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
        self.planned = np.array(['b']*length)
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
        elif label_to_plot == "planned":
            label_colors = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]])
            label_unique = np.array(["e", "o", "b"])
            labels = self.planned

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

    # To speed things up, I could have a check that if every voxel has been
    # labelled as visible, then stop loop
    def create_planned_labels(self, camera, images):
        # number of pixels which must be changed on the marker_board in order
        # for a voxel to be considered observed by an image
        threshold = 0

        grid_offset = np.array([[0, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0],
                                [0, 1, 1],
                                [1, 0, 0],
                                [1, 0, 1],
                                [1, 1, 0],
                                [1, 1, 1]])

        pts = self.grid_index*self.voxel_size + \
            self.voxel_size/2.0 + self.origin

        for image in images:
            print "Looking at %s" % image.name

            marking_board = np.zeros(np.flip(camera.size), dtype='u1')
            image_xyz = image.dcm.T.dot(-image.translation)

            # Before doing this, eventually I should check what voxels
            # actually can be seen in this image. This way I won't waist
            # checking voxels I can't see.
            voxel_distance = np.linalg.norm(pts - image_xyz, axis=1)

            indices = np.argsort(voxel_distance)

            # For saving workspace for debugging
            # dump = [camera, image, observe_threshold, grid_offset, marking_board, indices, self]
            # with open('create_planned_labels.pkl', 'w') as f:  # Python 3: open(..., 'wb')
            #     pickle.dump(dump, f)

            for index in indices:
                vertices = ((grid_offset + self.grid_index[index, :])
                            * self.voxel_size + self.origin)

                pixels = np.floor(camera.project_pt2image(vertices,
                                                          image)).astype('int')

                inside_image = (np.all(pixels >= 0, axis=1) &
                                np.all(pixels < np.flip(camera.size), axis=1))
                if np.all(inside_image):
                    # insert new method here
                    # board_cur[pixels[:, 1], pixels[:, 0]] = 1
                    # indices of projection
                    ind_pj = convex_fill(pixels, np.flip(camera.size),
                                         only_index=True)
                elif np.sum(inside_image) >= 1:
                    min_pixels = np.min(
                        np.concatenate(
                            (np.zeros((1, 2), dtype='int'), pixels),
                            axis=0), axis=0)
                    max_pixels = np.max(
                        np.concatenate(
                            (np.flip(camera.size[np.newaxis, :]), pixels),
                            axis=0), axis=0)
                    needed_size = max_pixels - min_pixels
                    offset = -min_pixels
                    # pixels = pixels + offset

                    # insert new method here
                    # board_tmp = np.zeros(np.flip(needed_size),
                    #                      dtype='int')
                    # board_tmp[pixels[:, 1], pixels[:, 0]] = 1
                    ind_tmp = convex_fill(pixels, needed_size,
                                          only_index=True)
                    ind_in = (np.all(ind_tmp >= 0, axis=1) &
                              np.all(ind_tmp < np.flip(camera.size), axis=1))
                    ind_pj = ind_tmp[ind_in, :]
                    #board_cur = board_tmp[offset[0]:(offset[0] + camera.size[1]),
                    #                      offset[1]:(offset[1] + camera.size[0])]
                else:
                    continue

                n_occluded = np.sum(marking_board[ind_pj[:, 0], ind_pj[:, 1]])
                n_observed = ind_pj.shape[0] - n_occluded
                if n_observed > threshold:
                    self.planned[index] = "o"
                # I assume that if a voxel is labelled as occupied then it
                # must have already been seen.
                elif self.planned[index] != "o" and n_observed <= threshold:
                    self.planned[index] = "b"
                # NOTE: I am assuming here that every voxel in this
                # voxel_labelled is occupied
                marking_board[ind_pj[:, 0], ind_pj[:, 1]] = 1

            # plt.imshow(marking_board)
            # plt.show()


class camera:
    def __init__(self):
        self.camera_id = -1
        self.model = ""
        self.size = np.zeros((2,))
        self.focal_length = 0.0
        self.principal_pt = np.array([0, 0])

    # Given array of pts, return pixel coordinates in image
    def project_pt2image(self, pts, image):
        X = np.concatenate((pts, np.ones([pts.shape[0], 1])), axis=1)

        M_ex = np.concatenate((image.dcm, image.translation[:, np.newaxis]),
                              axis=1)
        M_in = np.array([[self.focal_length, 0, self.principal_pt[0]],
                         [0, self.focal_length, self.principal_pt[1]],
                         [0,                 0,                 1.0]])

        p_h = M_in.dot(M_ex.dot(X.T))

        # return the pixels such that y pixel values are in the first column
        # and x are in the second. This will make it easier to index with them
        return np.transpose(p_h[1::-1, :]/p_h[2, :])


class image:
    def __init__(self):
        self.image_id = -1
        self.quaternion = np.array([1, 0, 0, 0])
        self.translation = np.array([0, 0, 0])
        self.camera_id = -1
        self.name = ""

    def calc_rotation_matrix(self):
        q0 = self.quaternion[0]
        q1 = self.quaternion[1]
        q2 = self.quaternion[2]
        q3 = self.quaternion[3]

        self.dcm = np.zeros([3, 3])
        self.dcm[0, 0] = q0**2 + q1**2 - q2**2 - q3**2
        self.dcm[0, 1] = 2*(q1*q2 + q0*q3)
        self.dcm[0, 2] = 2*(q1*q3 - q0*q2)
        self.dcm[1, 0] = 2*(q1*q2 - q0*q3)
        self.dcm[1, 1] = q0**2 - q1**2 + q2**2 - q3**2
        self.dcm[1, 2] = 2*(q2*q3 + q0*q1)
        self.dcm[2, 0] = 2*(q1*q3 + q0*q2)
        self.dcm[2, 1] = 2*(q2*q3 - q0*q1)
        self.dcm[2, 2] = q0**2 - q1**2 - q2**2 + q3**2

        # a = self.quaternion[0]
        # b = self.quaternion[1]
        # c = self.quaternion[2]
        # d = self.quaternion[3]

        # self.dcm[0, 0] = 2*a**2 - 1 + 2*b**2
        # self.dcm[0, 1] = 2*b*c + 2*a*d
        # self.dcm[0, 2] = 2*b*d - 2*a*c
        # self.dcm[1, 0] = 2*b*c - 2*a*d
        # self.dcm[1, 1] = 2*a**2 - 1 + 2*c**2
        # self.dcm[1, 2] = 2*c*d + 2*a*b
        # self.dcm[2, 0] = 2*b*d + 2*a*c
        # self.dcm[2, 1] = 2*c*d - 2*a*b
        # self.dcm[2, 2] = 2*a**2 - 1 + 2*d**2
        self.dcm = self.dcm.T
