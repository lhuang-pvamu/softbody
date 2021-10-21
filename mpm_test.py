import os
import numpy as np
import cv2
import taichi as ti
from Physics.mpm_simulator import MPMSimulator
from Physics.taichi_env import TaichiEnv

# TODO: run on GPU, fast_math will cause error on float64's sqrt; removing it cuases compile error..
ti.init(arch=ti.cpu, debug=False, fast_math=True)

gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF, show_gui=True)
#res = (1920, 1080)
#window = ti.ui.Window("Real MPM 3D", res, vsync=True)

# frame_id = 0
# canvas = window.get_canvas()
# scene = ti.ui.Scene()
# camera = ti.ui.make_camera()
# camera.position(0.5, 1.0, 1.95)
# camera.lookat(0.5, 0.3, 0.5)
# camera.fov(55)

def visualize(env, s, folder):
    aid = -1
    particles = env.simulator.get_x(s)
    n_particles = particles.shape[0]
    #print(n_particles, particles)
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    for i in range(n_particles):
        color = 0x111111
        # if aid[i] != -1:
        #     act = actuation[s - 1, aid[i]]
        #     color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')



cfg_path = "./envs/robot.yml"
version = 1
cfg = TaichiEnv.load_varaints(cfg_path, version)
print(cfg)
env = TaichiEnv(cfg, nn=False, loss=False)
env.initialize()
env.set_copy(True)

# particles = env.simulator.get_x(0)
# n_particles = particles.size
# colors = ti.Vector.field(2, float, n_particles)
# colors_random = ti.Vector.field(2, float, n_particles)
# use_random_colors = True
# particles_radius = 0.02
# for i in range(n_particles):
#     colors_random[i] = ti.Vector([ti.random(), ti.random()])

for i in range(100):
    env.step()
    visualize(env, i, 'diffmpm/iter{:05d}/'.format(i))
    #render(env.simulator.get_x(i))
#env.render()