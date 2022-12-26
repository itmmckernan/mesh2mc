import zipfile
import json
import numpy as np
import cv2


class TexturePack:
    def __init__(self, texture_pack_path):
        block_colors_lab = []
        block_colors_rgb = []
        block_textures = []
        self.block_names = []
        self.texture_size = -1
        with zipfile.ZipFile(texture_pack_path, 'r') as zip_file:
            for block_state in filter(lambda e: 'assets/minecraft/models/block/' in e.filename and not e.is_dir(),
                                      zip_file.filelist):
                block_file = zip_file.read(block_state)
                block_info = json.loads(block_file)
                if 'parent' in block_info.keys() and block_info['parent'].startswith("minecraft:block/cube_all"):
                    block_name = block_info['textures']['all'].split('/')[1]
                    if f'assets/minecraft/textures/block/{block_name}.png.mcmeta' not in [e.filename for e in
                                                                                          zip_file.filelist]:
                        img_as_bytes = np.asarray(
                            bytearray(zip_file.open(f'assets/minecraft/textures/block/{block_name}.png').read()),
                            dtype=np.uint8)
                        img = cv2.imdecode(img_as_bytes, cv2.IMREAD_COLOR)
                        self.texture_size = img.shape[0]
                        block_textures += [np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))]
                        img_labcolor = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        color_lab = img_labcolor.mean(axis=0).mean(axis=0)
                        color_rgb = img.mean(axis=0).mean(axis=0)
                        self.block_names += [block_name]
                        block_colors_rgb += [color_rgb[::-1]]
                        block_colors_lab += [color_lab]

        self.block_colors_lab = np.asarray(block_colors_lab)
        self.block_colors_rgb = np.asarray(block_colors_rgb)
        self.block_textures = np.stack(block_textures)
