from trimesh.proximity import ProximityQuery
from trimesh.visual import uv_to_interpolated_color
from scipy.spatial.distance import cdist
import numpy as np
import cv2
from PIL import Image
from enum import Enum


class VoxelColorSample:
    def __init__(self, mesh, voxel_grid, texture_pack, dither=0):
        self._mesh = mesh
        self._voxel_grid = voxel_grid
        self._texture_pack = texture_pack
        self._dither = dither

        self.used_blocks = None
        self.texture_image = None
        self.transposed_block_indexes = None
        self.closest_block_indexes = None
        self.color_type = None
        self._face_indexes = None
        self._distances = None
        self._closest = None
        self._voxel_lab_colors = None

        self.get_color_type()
        self.calc_voxel_points()
        self.calc_voxel_colors()
        self.apply_dither()
        self.match_to_blocks()
        self.assemble_texture()

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh
        self.get_color_type()
        self.calc_voxel_points()
        self.calc_voxel_colors()
        self.apply_dither()
        self.match_to_blocks()
        self.assemble_texture()

    @property
    def voxel_grid(self):
        return self._voxel_grid

    @voxel_grid.setter
    def voxel_grid(self, voxel_grid):
        self._voxel_grid = voxel_grid
        self.calc_voxel_points()
        self.calc_voxel_colors()
        self.apply_dither()
        self.match_to_blocks()
        self.assemble_texture()

    @property
    def texture_pack(self):
        return self._texture_pack

    @texture_pack.setter
    def texture_pack(self, texture_pack):
        self._texture_pack = texture_pack
        self.match_to_blocks()
        self.assemble_texture()

    @property
    def dither(self):
        return self._dither

    @dither.setter
    def dither(self, dither):
        self._dither = dither
        self.apply_dither()
        self.match_to_blocks()
        self.assemble_texture()

    def get_color_type(self):
        if self._mesh.visual.material.baseColorTexture is not None:
            self.color_type = ColorType.textureColor
        else:
            try:
                a = self._mesh.visual.to_color().vertex_colors
                self.color_type = ColorType.vertexColor
            except TypeError:
                self.color_type = ColorType.baseColor

    def calc_voxel_points(self):
        voxel_points = self._voxel_grid.points
        self._closest, self._distances, self._face_indexes = ProximityQuery(self._mesh).on_surface(voxel_points)

    def calc_voxel_colors(self):
        match self.color_type:
            case ColorType.textureColor:
                self.calc_voxel_colors_texture()
            case ColorType.vertexColor:
                self.calc_voxel_colors_vertexColors()
            case ColorType.baseColor:
                self.calc_voxel_colors_main_color()
            case _:
                self.calc_voxel_colors_main_color()


    def calc_voxel_colors_texture(self):
        p0s = self._mesh.vertices[self._mesh.faces[self._face_indexes][:, 0]]
        us = self._mesh.vertices[self._mesh.faces[self._face_indexes][:, 1]] - p0s
        vs = self._mesh.vertices[self._mesh.faces[self._face_indexes][:, 2]] - p0s
        ws = self._closest - p0s

        v_cross_ws = np.cross(vs, ws)
        u_cross_ws = np.cross(us, ws)
        u_cross_vs = np.cross(us, vs)
        divisor = np.linalg.norm(u_cross_vs, axis=1)
        b1s = np.linalg.norm(v_cross_ws, axis=1) / divisor
        b2s = np.linalg.norm(u_cross_ws, axis=1) / divisor
        b0s = 1 - b1s - b2s
        uv_vertex_pos = self._mesh.visual.uv[self._mesh.faces[self._face_indexes]].swapaxes(1, 2)
        uv_vertex_mult = np.column_stack((b0s, b1s, b2s)).repeat(2, axis=0).reshape(uv_vertex_pos.shape)
        uv_samples = np.multiply(uv_vertex_pos, uv_vertex_mult).sum(axis=2)

        voxel_colors = uv_to_interpolated_color(uv_samples, self._mesh.visual.material.baseColorTexture)
        self._voxel_lab_colors = np.asarray(
            cv2.cvtColor(np.asarray([voxel_colors[:, :3]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0],
            dtype='float')

    def calc_voxel_colors_vertexColors(self):
        #TODO: Implement
        self.calc_voxel_colors_main_color()

    def calc_voxel_colors_main_color(self):
        self._voxel_lab_colors = np.tile(self._mesh.visual.material.main_color[:3], (len(self._voxel_grid.points), 1))

    def calc_voxel_colors_noColor(self):
        self._voxel_lab_colors = np.array([128, 128, 128]*len(self._voxel_grid.points))

    def apply_dither(self):
        if not self._dither == 0:
            self._voxel_lab_colors = self._voxel_lab_colors.astype(np.float64)
            self._voxel_lab_colors += np.random.normal(0, self._dither, (self._voxel_grid.sparse_indices.shape[0], 3))

    def match_to_blocks(self):
        self.closest_block_indexes = np.argmin(cdist(self._voxel_lab_colors, self._texture_pack.block_colors_lab),
                                                axis=1)

    def assemble_texture(self):
        self.used_blocks = np.unique(self.closest_block_indexes)
        sorter = np.argsort(self.used_blocks)
        self.transposed_block_indexes = sorter[np.searchsorted(self.used_blocks, self.closest_block_indexes, sorter=sorter)]
        texture_image = np.concatenate(np.take(self._texture_pack.block_textures, self.used_blocks, axis=0))
        self.texture_image = Image.fromarray(texture_image)
        resize_size = np.array(np.array(self.texture_image.size, dtype=np.float64)*16384//self.texture_image.size[1], dtype=np.int)
        self.texture_image = self.texture_image.resize(resize_size, resample=4)


class ColorType(Enum):
    baseColor = 1
    vertexColor = 2
    textureColor = 3
