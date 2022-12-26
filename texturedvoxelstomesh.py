import numpy as np
import trimesh
from trimesh.visual import TextureVisuals


class TexturedVoxelsToMesh:
    def __init__(self, voxel_grid, voxel_color_sampler):
        self.generated_mesh = None
        self._voxel_grid = voxel_grid
        self._voxel_color_sampler = voxel_color_sampler

        self.calc()

    @property
    def voxel_grid(self):
        return self._voxel_grid

    @voxel_grid.setter
    def voxel_grid(self, voxel_grid):
        self._voxel_grid = voxel_grid
        self.calc()

    @property
    def voxel_color_sampler(self):
        return self._voxel_color_sampler

    @voxel_color_sampler.setter
    def voxel_color_sampler(self, voxel_color_sampler):
        self._voxel_color_sampler = voxel_color_sampler
        self.calc()

    def calc(self):
        centers = self._voxel_grid.sparse_indices.astype(np.float64)
        num_voxels = self._voxel_grid.points.shape[0]
        single_box_vertices = np.asarray(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
             [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
             [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

        single_box_faces = [[0, 1, 2], [3, 2, 1], [4, 5, 6], [7, 6, 5], [8, 9, 10], [11, 10, 9], [12, 13, 14],
                            [15, 14, 13], [16, 17, 18], [19, 18, 17], [20, 21, 22], [23, 22, 21]]

        single_texture_distance = 1 / len(self._voxel_color_sampler.used_blocks)
        uv_constants = np.asarray([6, 5, 6, 5, 4, 5, 4, 5, 2, 1, 2, 1, 3, 4, 3, 4, 3, 2, 3, 2, 0, 0, 1, 1])
        uv_constants_full = np.tile(uv_constants, num_voxels)
        pinch = 0.0
        std = single_texture_distance - pinch
        uv_mutable = np.asarray([std, std, pinch, pinch, std,
                                 std, pinch, pinch, std, std, pinch, pinch,
                                 std, std, pinch, pinch, std,
                                 std, pinch, pinch, std, pinch, std, pinch])
        uv_mutable_full = np.tile(uv_mutable, num_voxels) + np.repeat(
            max(self._voxel_color_sampler.transposed_block_indexes) - self._voxel_color_sampler.transposed_block_indexes,
            uv_mutable.shape[0]) * single_texture_distance
        uv = np.column_stack((uv_constants_full, uv_mutable_full))
        vertices = np.tile(centers, (1, len(single_box_vertices))).reshape((-1, 3))
        vertices += np.tile(single_box_vertices, (len(centers), 1))
        faces = np.tile(single_box_faces, (len(centers), 1))
        faces += np.tile(np.arange(len(centers)) * len(single_box_vertices), (len(single_box_faces), 1)).T.reshape(
            (-1, 1))

        self.generated_mesh = trimesh.Trimesh(vertices=vertices,
                                              faces=faces,
                                              visual=TextureVisuals(uv=uv,
                                                                    image=self._voxel_color_sampler.texture_image))
        self.generated_mesh.apply_transform(self._voxel_grid.transform)
