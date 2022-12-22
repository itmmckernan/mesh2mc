import json
import trimesh
from trimesh.proximity import ProximityQuery
from trimesh.voxel import ops
import numpy as np
import zipfile
import cv2
from scipy.spatial.distance import cdist


mesh = './BoxTextured.glb'
texture_pack = './VanillaDefault+1.19.3.zip'

block_colors_lab = []
block_colors_rgb = []
block_names = []
with zipfile.ZipFile(texture_pack, 'r') as zip_file:
    for block_state in filter(lambda e: 'assets/minecraft/models/block/' in e.filename and not e.is_dir(), zip_file.filelist):
        block_file = zip_file.read(block_state)
        block_info = json.loads(block_file)
        if 'parent' in block_info.keys() and block_info['parent'].startswith("minecraft:block/cube_all"):
            block_name = block_info['textures']['all'].split('/')[1]
            if block_name == 'ice':
                1 + 1
            img_as_bytes = np.asarray(bytearray(zip_file.open(f'assets/minecraft/textures/block/{block_name}.png').read()), dtype=np.uint8)
            img = cv2.imdecode(img_as_bytes, cv2.IMREAD_COLOR)
            img_labcolor = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            color_lab = img_labcolor.mean(axis=0).mean(axis=0)
            color_rgb = img.mean(axis=0).mean(axis=0)
            block_names += [block_name]
            block_colors_rgb += [color_rgb]
            block_colors_lab += [color_lab]

block_colors_lab = np.asarray(block_colors_lab)
block_colors_rgb = np.asarray(block_colors_rgb)

trimesh.util.attach_to_log()

scene = trimesh.load('./Duck.glb')
mesh = list(scene.geometry.items())[0][1]

pitch = .6
dither = 0

print(f'Number of voxels: {np.prod(np.ceil(mesh.extents/pitch))}')
voxel_grid = mesh.voxelized(pitch).hollow()

voxel_points = voxel_grid.points
closest, distances, tri_ids = ProximityQuery(mesh).on_surface(voxel_points)
spatial_voxel_colors = np.zeros([voxel_grid.shape[0], voxel_grid.shape[1], voxel_grid.shape[2], 4])
all_colors = np.asarray(mesh.visual.to_color().vertex_colors)
mesh.visual.material.to_color([[1, 1]])



voxel_colors = np.take(all_colors, vert_ids, axis=0)
voxel_indexes = voxel_grid.sparse_indices

for voxel_index, voxel_color in zip(voxel_indexes, voxel_colors):
    voxel_lab_color = np.asarray(cv2.cvtColor(np.asarray([[voxel_color]]), cv2.COLOR_RGB2LAB)[0][0], dtype='float')
    voxel_lab_color += np.random.normal(0, dither, 3)
    closest_block_index = np.argmin(cdist([voxel_lab_color[:3]], block_colors_lab))
    closest_block = block_names[closest_block_index]
    #print(f'Closest block found was {closest_block} with an RGB value of {block_colors_rgb[closest_block_index]} compared to the voxel\'s of {voxel_color}')
    spatial_voxel_colors[voxel_index[0], voxel_index[1], voxel_index[2], :] = np.append(block_colors_rgb[closest_block_index], 255)


voxel_out_mesh = ops.multibox(centers=voxel_grid.sparse_indices.astype(float), colors=spatial_voxel_colors[voxel_grid.encoding.dense], pitch=1/pitch)

voxel_out_mesh = voxel_out_mesh.apply_transform(voxel_grid.transform)

new_scene = trimesh.Scene()

new_scene.add_geometry(voxel_out_mesh)

new_scene.show()
