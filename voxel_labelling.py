import open3d.open3d as o3d
import numpy as np
# from skimage.morphology import convex_hull_image
from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm   # adds loading bars!
import copy

# import pickle

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


def import_BIM(fn, min_bound_coord, max_bound_coord, point_cloud_density):
    mesh_p = o3d.io.read_triangle_mesh(fn)
    print(mesh_p)

    # generate numpy array of vertices from as-planned model
    vert_p = np.asarray(mesh_p.vertices)

    # parse .obj file to find indices of named objects (building elements)
    print "Parsing *.obj file"
    elements = []

    with open(fn) as f_obj:
        lines_obj = f_obj.readlines()

    k = 0
    for line in lines_obj:
        if line.startswith('o '):
            elements.append(dict(name=line[2:-1],
                                 color=np.random.rand(3),
                                 v_index=[]))

        if line.startswith('v '):
            vert_cur = np.asarray(line[2:-1].split(' ')).astype('float')
            diff_cur = vert_cur - vert_p[k, :]
            if (np.abs(diff_cur) > 1E-5).any():
                print("Warning: vertex %G differed by %s" %
                      (k, np.array2string(diff_cur)))

            elements[-1]['v_index'].append(k)

            k = k + 1

    print "Generating point clouds and removing unwanted elements"
    # Loop through every named object from the obj in reverse order, mark ones
    # which fall outside the specified bounds, and save point clouds of the
    # ones that don't
    elements_to_delete = []
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
            # The area is used to ensure every element has a mostly equally
            # dense point cloud.
            n_pts = int(triangle_mesh_area(mesh_temp)*point_cloud_density)
            # if len(elements[k]['name']) > 40:
            #     elem_name = (elements[k]['name'][0:19] + " ... " +
            #                  elements[k]['name'][-15:0])
            # else:
            #     elem_name = elements[k]['name']

            print("Saved %s as point cloud with %G points."
                  % (elements[k]['name'], n_pts))
            pcd_temp = mesh_temp.sample_points_poisson_disk(n_pts)
            elements[k]['point_cloud'] = np.asarray(pcd_temp.points)
        else:
            elements_to_delete.append(k)

    # delete elements that were out of bounds
    for k in elements_to_delete:
        del elements[k]

    return elements, mesh_p


# Return the indices of a filled convex hull based on indices provided by
# pixels. Optionally, return the image generated by the indices instead (this
# will be slower than just returning the indices).
def convex_fill(pixels, return_image=False, size=None):
    hull = ConvexHull(pixels)
    deln = Delaunay(pixels[hull.vertices])

    # restrict to area bounded by max and min of convex hull
    y = np.arange(hull.min_bound[0], hull.max_bound[0])
    x = np.arange(hull.min_bound[1], hull.max_bound[1])
    idx = np.stack(np.meshgrid(y, x), axis=-1)

    # indices of pixels in convex hull
    out_idx = np.asarray(np.nonzero(deln.find_simplex(idx) + 1)).T
    out_idx = np.flip(out_idx, axis=1) + hull.min_bound.astype('int')
    if return_image:
        out_image = np.zeros(size, dtype='u1')
        out_image[out_idx[:, 0], out_idx[:, 1]] = 1
        return out_image
    else:
        return out_idx


# Base class for labelling voxels
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
        self.elements = -99*np.ones([n_voxels, n_elements_keep], dtype='int32')
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


# Voxels with labels of what construction elements they belong to. Intended to
# serve as a reference and return the voxel_labelled class.
# NOTE: the VoxelGrid object used to initialize should be generated with the
# same point clouds that will be used in create_element_labels. In this way no
# as-planned voxels can be labelled as empty.
class voxel_label_reference(voxel_label_base):
    pass

    # Fill out element labels using list of elements
    # For each voxel, loop through every element and check whether any points
    # fall inside that voxel. Save the indices of the elements from most to
    # least points inside the voxel.
    # NOTE: An element of -99 is used to indicate an unassigned label. All
    # element labels should be positive.
    # NOTE: For future reference, the order of elements should not change!
    def create_element_labels(self, elements):
        for k in tqdm(range(self.length)):
            voxel_min_coord, voxel_max_coord = self.voxel_bounds(index=k)
            n_inside_prior = np.zeros((self.elements.shape[1],))

            k_element = 0
            for element in elements:
                pts_cur = element['point_cloud']

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

    # Return a voxel_labelled class which only contains voxels labelled with
    # the desired_elements.
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


# Voxels with the additional labels built and planned for labelling the
# as-built model and as-planned model.
# NOTE: Should only be created by using export_voxel_labilled.
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

        # NOTE: Labels for planned and built can be 'e', 'o', or 'b'
        # (empty, occupied, blocked)

        # label for as-planned model
        self.planned = np.array(['b']*length)
        # label for as-built model
        self.built = np.array(['e']*length)

    # For every image provided, loop through every voxel from closest to
    # furthest from the image. Project the voxel onto the image plane and
    # check if it can be seen infront of other previously projected voxels. If
    # so then label it as occupied.
    # To speed things up, I could have a check that if every voxel has been
    # labelled as visible, then stop loop
    def create_planned_labels(self, camera, images):
        # number of pixels which must be changed on the marker_board in order
        # for a voxel to be considered observed by an image
        threshold = 20

        # Easy access to creating voxel vertices
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

        pbar = tqdm(images)
        for image in pbar:
            marking_board = np.zeros(np.flip(camera.size), dtype='u1')
            image_xyz = image.dcm.T.dot(-image.translation)

            voxel_distance = np.linalg.norm(pts - image_xyz, axis=1)

            indices = np.argsort(voxel_distance)

            for index in indices:
                vertices = ((grid_offset + self.grid_index[index, :])
                            * self.voxel_size + self.origin)

                X = np.concatenate((vertices, np.ones([vertices.shape[0], 1])),
                                   axis=1)
                M_ex = np.concatenate((image.dcm,
                                       image.translation[:, np.newaxis]),
                                      axis=1)
                # voxel vertex coordinates in camera frame
                vertices_c = M_ex.dot(X.T).T

                # vertices onto image plane and return the pixel indices [y, x]
                pixels = np.floor(camera.project_pt2image(vertices,
                                                          image)).astype('int')

                # which pixels fall within the image frame
                inside_image = (np.all(pixels >= 0, axis=1) &
                                np.all(pixels < np.flip(camera.size), axis=1))

                # If at least one vertex of the voxel falls within the image
                # bounds and is infront of the camera, project it onto the
                # marking_board.
                in_view = ((vertices_c[:, 2] > 0).any() and
                           np.sum(inside_image) >= 1)
                if in_view:
                    # indices of projection
                    ind_tmp = convex_fill(pixels)

                    if np.all(inside_image):
                        ind_pj = ind_tmp
                    else:  # remove indices outside bounds
                        ind_in = (np.all(ind_tmp >= 0, axis=1) &
                                  np.all(ind_tmp < np.flip(camera.size),
                                  axis=1))
                        ind_pj = ind_tmp[ind_in, :]

                    n_occluded = (np.sum(
                                  marking_board[ind_pj[:, 0], ind_pj[:, 1]]))
                    n_observed = ind_pj.shape[0] - n_occluded
                    if n_observed > threshold:
                        self.planned[index] = "o"
                    # I assume that if a voxel is labelled as occupied then it
                    # must have already been seen.
                    elif self.planned[index] != "o":
                        self.planned[index] = "b"

                    marking_board[ind_pj[:, 0], ind_pj[:, 1]] = 1

            pbar.set_description("Looking at %s" % image.name)
            # plt.imshow(marking_board)
            # plt.show()

    def create_built_labels(self, pcd_built):
        """
        If a voxel contains a point from the as-built point cloud, then label
        it as occupied. If it has no point, search the 6 neighboring voxels
        which are not already present in the voxel grid. If a point is found
        in the neighboring voxels, then label the original as occupied.
        """
        # find indices for occupied, visible voxels
        indices = np.where(self.planned == "o")[0]

        self.built[self.planned == "b"] = "b"

        pts_all = np.asarray(pcd_built.points)
        pts_ind = (np.all(pts_all >= self.min_bound, axis=1) &
                   np.all(pts_all <= self.max_bound, axis=1))
        pts = pts_all[pts_ind, :]

        for index in tqdm(indices):
            voxel_min, voxel_max = self.voxel_bounds(index=index)

            # Check if the current voxel has a point
            if np.any(np.all(pts >= voxel_min, axis=1) &
                      np.all(pts <= voxel_max, axis=1)):
                self.built[index] = "o"
            else:
                # generate neighboring grid indices
                neighbor_grid = np.concatenate((np.eye(3), -np.eye(3)), axis=0)
                neighbor_grid += self.grid_index[index, :]

                found_point = False
                k = 0
                while not found_point and k < neighbor_grid.shape[0]:
                    k_neighbor = neighbor_grid[k, :]

                    k_min, k_max = self.voxel_bounds(grid_index=k_neighbor)
                    index_taken = (self.grid_index ==
                                   k_neighbor).all(axis=1).any()

                    # If the current neighbor index isn't already in
                    # self.grid_index and contains an as-built point, then
                    # label as occupied
                    if (not index_taken and
                            np.any(np.all(pts >= k_min, axis=1) &
                                   np.all(pts <= k_max, axis=1))):
                        self.built[index] = "o"
                        found_point = True

                    k += 1

    def predict_progress(self):
        # threshold on the probability that an element is present for it to be
        # considered progressed
        t_predict = 0.25

        expected_elements = np.unique(self.elements)

        present_elements = []
        for element in expected_elements:
            # this isn't working?
            index = (self.elements == element) & (self.built != "b")

            n_occupied = np.sum(self.built[index] == "o", dtype='int')
            n_empty = np.sum(self.built[index] == "e", dtype='int')

            # Do not calculate probability if element is completely blocked
            if n_occupied == 0 and n_empty == 0:
                P_progress = np.nan
                print "Element %G is blocked from view." % (element)
            else:
                P_progress = n_occupied/float(n_occupied + n_empty)
                print "Element %G progress has probability %f" % (element,
                                                                  P_progress)
            if P_progress > t_predict:
                present_elements.append(element)

        return present_elements

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
        elif label_to_plot == "built":
            label_colors = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]])
            label_unique = np.array(["e", "o", "b"])
            labels = self.built
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

        # Code for plotting displaying voxelsing using matplotlib instead.
        # It's very slow though, so only useful for generating pictures...
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
        self.dcm = np.empty([3, 3])

    def calc_rotation_matrix(self):
        q0 = self.quaternion[0]
        q1 = self.quaternion[1]
        q2 = self.quaternion[2]
        q3 = self.quaternion[3]

        self.dcm[0, 0] = q0**2 + q1**2 - q2**2 - q3**2
        self.dcm[0, 1] = 2*(q1*q2 + q0*q3)
        self.dcm[0, 2] = 2*(q1*q3 - q0*q2)
        self.dcm[1, 0] = 2*(q1*q2 - q0*q3)
        self.dcm[1, 1] = q0**2 - q1**2 + q2**2 - q3**2
        self.dcm[1, 2] = 2*(q2*q3 + q0*q1)
        self.dcm[2, 0] = 2*(q1*q3 + q0*q2)
        self.dcm[2, 1] = 2*(q2*q3 - q0*q1)
        self.dcm[2, 2] = q0**2 - q1**2 - q2**2 + q3**2

        self.dcm = self.dcm.T
