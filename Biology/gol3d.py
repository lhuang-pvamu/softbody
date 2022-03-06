# run in Blender

import numblend as nb
import taichi_glsl as tl
import taichi as ti
import numpy as np
import bpy
import time

nb.init()
ti.init(arch=ti.cpu)


n=16
cell_size = 8
img_size = n * cell_size
grid = ti.field(int, shape=(n,n,n))
count = ti.field(int, shape=(n,n,n))
reproduce_rule = [5, 6] # the number of neighors to reproduce the cell
live_rule = [9, 10, 11, 12, 13, 14, 15, 16]  # the number of neighors to keep it alive

pos = ti.Vector.field(3, float, (n, n, n))
offset = 2

@ti.kernel
def init():
    for i, j, k in pos:
        pos[i, j, k] = ti.Vector([0,0,0]) #ti.Vector([i - n / 2, j - n / 2, k - n/2  ])

    for i in range(-offset,offset):
        for j in range(-offset,offset):
            for k in range(-offset, offset):
                grid[n/2+i, n/2+j, n/2+k] = 1

@ti.func
def get_count(i, j, k):
    return (grid[i - 1, j, k] + grid[i + 1, j, k] + grid[i, j - 1, k] +
            grid[i, j + 1, k] + grid[i - 1, j - 1, k] + grid[i + 1, j - 1, k] +
            grid[i - 1, j + 1, k] + grid[i + 1, j + 1, k] +
            grid[i - 1, j, k-1] + grid[i + 1, j, k-1] + grid[i, j - 1, k-1] +
            grid[i, j + 1, k-1] + grid[i - 1, j - 1, k-1] + grid[i + 1, j - 1, k-1] +
            grid[i - 1, j + 1, k-1] + grid[i + 1, j + 1, k-1] +
            grid[i - 1, j, k+1] + grid[i + 1, j, k+1] + grid[i, j - 1, k+1] +
            grid[i, j + 1, k+1] + grid[i - 1, j - 1, k+1] + grid[i + 1, j - 1, k+1] +
            grid[i - 1, j + 1, k+1] + grid[i + 1, j + 1, k+1] +
            grid[i, j, k-1] + grid[i, j, k+1])

@ti.func
def calc_rule(status, neighbors):
    if status == 0:
        for r in ti.static(reproduce_rule):
            if neighbors == r:
                status = 1
    elif status == 1:
        status = 0
        for r in ti.static(live_rule):
            if neighbors == r:
                status = 1
    return status

@ti.kernel
def update(t: float):
    for i, j, k in pos:
        pos[i, j, k].z = ti.sin(pos[i, j, k].xyz.norm() * 0.5 - t * 2)


@ti.kernel
def run():
    for i, j, k in grid:
        count[i, j, k] = get_count(i, j, k)

    for i, j, k in grid:
        grid[i, j, k] = calc_rule(grid[i, j, k], count[i, j, k])
        if grid[i, j, k] == 0:
            pos[i,j,k] = ti.Vector([0,0,0])
        else:
            pos[i,j,k] = ti.Vector([i - n / 2, j - n / 2, k - n/2  ])


## delete old mesh & object (if any)
#nb.delete_mesh('point_cloud')
#nb.delete_object('point_cloud')

## create a new point cloud
#mesh = nb.new_mesh('point_cloud', np.zeros((n**3, 3)))
#object = nb.new_object('point_cloud', mesh)

#verts, edges, faces, uv = nb.meshgrid(n*n)
#nb.delete_mesh('point_cloud')
#nb.delete_object('point_cloud')
#mesh = nb.new_mesh('point_cloud', verts, edges, faces, uv)
#nb.new_object('point_cloud', mesh)

objects = []
for i in range(n**3):
    nb.delete_object(f'cube_{i}')
    bpy.ops.mesh.primitive_cube_add(size=1)
    bpy.context.object.name = f'cube_{i}'
    objects.append(bpy.context.object)

# define animation iterator body
@nb.add_animation
def main():
    init()
    for frame in range(1000):
        run()
        #time.sleep(0.1)
        #update(frame * 0.03)
        #yield nb.mesh_update(mesh, pos=pos.to_numpy().reshape(n**3, 3))
        yield nb.objects_update(objects, location=pos.to_numpy().reshape(n**3, 3))