__author__ = 'Lei Huang'

import numpy as np
import taichi as ti
ti.init(arch=ti.cuda)

n=64
cell_size = 8
img_size = n * cell_size
grid = ti.field(int, shape=(n,n))
count = ti.field(int, shape=(n,n))
reproduce_rule = [1, 2] # the number of neighors to reproduce the cell
live_rule = [2, 3]  # the number of neighors to keep it alive

@ti.func
def get_count(i, j):
    return (grid[i - 1, j] + grid[i + 1, j] + grid[i, j - 1] +
            grid[i, j + 1] + grid[i - 1, j - 1] + grid[i + 1, j - 1] +
            grid[i - 1, j + 1] + grid[i + 1, j + 1])

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
def run():
    for i, j in grid:
        count[i, j] = get_count(i, j)

    for i, j in grid:
        grid[i, j] = calc_rule(grid[i, j], count[i, j])

@ti.kernel
def init():
    for i, j in grid:
        if i > n/2-5 and i<n/2+5 and j>n/2-5 and j<n/2+5: # ti.random() > 0.8:
            grid[i, j] = 1
        else:
            grid[i, j] = 0

gui = ti.GUI('Game of Life', (img_size, img_size))
gui.fps_limit = 15

print('[Hint] Press `r` to reset')
print('[Hint] Press SPACE to pause')
print('[Hint] Click LMB, RMB and drag to add alive / dead cells')

init()
paused = False
while gui.running:
    for e in gui.get_events(gui.PRESS, gui.MOTION):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.SPACE:
            paused = not paused
        elif e.key == 'r':
            grid.fill(0)

    if gui.is_pressed(gui.LMB, gui.RMB):
        mx, my = gui.get_cursor_pos()
        grid[int(mx * n), int(my * n)] = gui.is_pressed(gui.LMB)
        paused = True

    if not paused:
        run()

    gui.set_image(ti.imresize(grid, img_size).astype(np.uint8) * 255)
    gui.show()