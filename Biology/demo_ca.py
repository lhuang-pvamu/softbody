import numpy as np
import taichi as ti
from lib.CA_Model import CA_Model
import torch

ti.init(arch=ti.cpu)
img_size = 100
CHANNEL_N = 16
CELL_FIRE_RATE = 0.5
model_path = "models/gen_2.model"
device = torch.device("cpu")

def make_seed(shape, n_channels):
    seed = np.zeros([shape[0], shape[1], n_channels], np.float32)
    seed[shape[0]//2, shape[1]//2, 3:] = 1.0
    return seed
def to_alpha(x):
    return np.clip(x[..., 3:4], 0, 0.9999)
def to_rgb(x):
    # assume rgb premultiplied by alpha
    x = np.rot90(x, k=1, axes=(1,0))
    rgb, a = x[..., :3], to_alpha(x)
    return np.clip(1.0-a+rgb, 0, 0.9999)

_map = make_seed((img_size, img_size), CHANNEL_N)

model = CA_Model(CHANNEL_N, CELL_FIRE_RATE, device).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
output = model(torch.from_numpy(_map.reshape([1,img_size,img_size,CHANNEL_N]).astype(np.float32)), 1)

gui = ti.GUI('Game of Life', (img_size, img_size))
gui.fps_limit = 15
steps = 100
step = 1

paused = False
while gui.running:
    if step < steps:
        output = model(output, 1)
        step+=1

    gui.set_image(to_rgb(output.detach().numpy()[0]))
    #gui.show()
    gui.show(f'output/{step:06d}.png')

