import math, taichi as ti, numpy as np

ti.init(arch=ti.cpu)

rows = 16
cols = 40
max_neighbors = 6
N = rows*cols
dt = 0.02
s = 10
ks = 8.0
paused = True
steps = 10
DX, DY = (800, 400)
inv_h = 1.0/(4*s)
nx = math.ceil(DX*inv_h)
ny = math.ceil(DY*inv_h)
n1 = 4
Nw = 4*n1

pos = ti.Vector.field(2, dtype = ti.f32, shape = N)
vel = ti.Vector.field(2, dtype = ti.f32, shape = N)
force = ti.Vector.field(2, dtype = ti.f32, shape = N)
cell = ti.Vector.field(max_neighbors, dtype = ti.i32, shape=(nx,ny))
grid = ti.field(dtype = ti.i32, shape=(nx,ny))
wheel = ti.Vector.field(2, dtype = ti.f32, shape = Nw)
pos_w = ti.Vector.field(2, dtype = ti.f32, shape = Nw)
angle = ti.field(ti.f32, ())
omega = ti.field(ti.f32, ())
torque = ti.field(ti.f32, ())
inertia = ti.field(ti.f32, ())

@ti.kernel
def initialize():
    dx, dy = (21,21)
    angle[None] = 0
    omega[None] = -0.01
    for i in range(rows):
        for j in range(cols):
            p = j + i*cols
            pos[p] = [(j - cols + 1)*dx , (i*dy + (DY - (rows-1)*dy)/2.0)] 
            vel[p] = [2.0 + (1-2*ti.random())*0.1, (1-2*ti.random())*0.1]
    for i in range(n1):
        wheel[i] = [2*(i+1)*s, 0]
        wheel[n1 + i] = [0, 2*(i+1)*s ] 
        wheel[2*n1 + i] = [-2*(i+1)*s, 0] 
        wheel[3*n1 + i] = [0, -2*(i+1)*s]
    for i in wheel:
        inertia[None] += wheel[i][0]**2 + wheel[i][1]**2
        pos_w[i] = wheel[i] + ti.Matrix([DX/2, DY/2])
    

@ti.kernel
def update():
    for p in pos:
        vel[p] += dt*force[p]
        pos[p] += dt*vel[p]
        pos[p] = pos[p] % [DX,DY]
    omega[None] += torque[None]/inertia[None]*dt
    angle[None] += omega[None]*dt
    R = ti.Matrix([[ti.cos(angle[None]), -ti.sin(angle[None])],[ti.sin(angle[None]),ti.cos(angle[None])]])
    CM = ti.Matrix([DX/2, DY/2])
    for p in wheel:
        pos_w[p] = R @ wheel[p] + CM

@ti.kernel 
def p2g():
    for i,j in grid:
        grid[i,j] = 0
    
    for p in pos:
        base = ti.cast(pos[p]*inv_h, ti.i32)
        I = grid[base]
        for i in ti.static(range(6)):
            if i == I: 
                cell[base][i] = p   
        grid[base] += 1

@ti.kernel 
def compute_force():
    for p in force:
        force[p] = [0,0]
    for i, j in cell:
        for k1 in range(grid[i,j]):
            for k2 in range(grid[i,j]):
                if k1 < k2:
                    r12 = pos[k1] - pos[k2]
                    
@ti.kernel 
def compute_force_on2():
    for p in force:
        force[p] = [0,0]
    torque[None] = 0
    for p in range(N):
        for q in range(N):
            if p < q:
                rpq = pos[p] - pos[q]
                fs = ks*(s*ti.rsqrt(rpq[0]**2 + rpq[1]**2) - 1.0)
                if fs > 0:
                    force[p] +=  fs*rpq
                    force[q] += -fs*rpq
    for p in range(N):
        for q in range(Nw):
            rpq = pos[p] - pos_w[q]
            fs = ks*(s*ti.rsqrt(rpq[0]**2 + rpq[1]**2) - 1.0)
            if fs > 0:
                force[p] += fs*rpq
                torque[None] += fs*(rpq[0]*(pos_w[q][1] - DY/2) - rpq[1]*(pos_w[q][0] - DX/2))
            
gui = ti.GUI('Wind Tunnel',(800,400), background_color=0x112F41)
initialize()

while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == 'r':
            initialize()
        elif e.key == ti.GUI.SPACE:
            paused = not paused

    gui.circle((0.5,0.5), color=0xff5733, radius=4)
    if not paused:
        for k in range(steps):
            compute_force_on2()
            update()

    gui.circles(pos.to_numpy()/[DX, DY], color=0x068587, radius=2)
    gui.circles(pos_w.to_numpy()/[DX, DY], color=0xff5733, radius=s)
    gui.show()



