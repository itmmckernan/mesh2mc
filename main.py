import trimesh
from texturepack import TexturePack
from voxelcolorsample import VoxelColorSample
from texturedvoxelstomesh import TexturedVoxelsToMesh
import tqdm
import concurrent.futures
mesh = './models/detailed_old_nyc_building.glb'
texture_pack_path = './texture packs/VanillaDefault+1.19.3.zip'

#trimesh.util.attach_to_log()
texture_pack = TexturePack(texture_pack_path)


pitch = .5
dither = 8


scene = trimesh.Scene()
loaded_scene = trimesh.load(mesh)

for mesh in loaded_scene.geometry.values():
    voxel_grid = mesh.voxelized(pitch).hollow()
    num_voxels = voxel_grid.points.shape[0]
    voxel_color_sampler = VoxelColorSample(mesh, voxel_grid, texture_pack, dither)
    textured_voxels_to_mesh = TexturedVoxelsToMesh(voxel_grid, voxel_color_sampler)
    generated_trimesh = textured_voxels_to_mesh.generated_mesh
    scene.add_geometry(generated_trimesh)

scene.show()
