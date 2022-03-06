import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

from CA_Model import CA_Model

def load_emoji(index, path="data/dog.png"):
    im = imageio.imread(path)
    print(im.shape)
    if im.shape[2] == 3:
        im = np.concatenate([im, np.ones((40,40,1))*255], axis=2)
    print(im.shape)
    emoji = np.array(im[:, index*40:(index+1)*40].astype(np.float32))

    emoji /= 255.0
    return emoji

def to_alpha(x):
    return np.clip(x[..., 3:4], 0, 0.9999)

def to_rgb(x):
    # assume rgb premultiplied by alpha
    print(x.shape)
    rgb, a = x[..., :3], to_alpha(x)
    return np.clip(1.0-a+rgb, 0, 0.9999)

def make_seeds(shape, n_channels, n=1):
    x = np.zeros([n, shape[0], shape[1], n_channels], np.float32)
    x[:, shape[0]//2, shape[1]//2, 3:] = 1.0
    return x

def make_seed(shape, n_channels):
    seed = np.zeros([shape[0], shape[1], n_channels], np.float32)
    seed[shape[0]//2, shape[1]//2, 3:] = 1.0
    return seed

device = torch.device("cpu")
model_path = "models/gen_2.model"

CHANNEL_N = 16        # Number of CA state channels
TARGET_PADDING = 16   # Number of pixels used to pad the target image border
TARGET_SIZE = 40

lr = 2e-3
lr_gamma = 0.9999
betas = (0.5, 0.5)
n_epoch = 80000

BATCH_SIZE = 4
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.5

TARGET_EMOJI = 0
target_img = load_emoji(TARGET_EMOJI)
plt.figure(figsize=(4,4))
plt.imshow(to_rgb(target_img))
plt.show()


p = TARGET_PADDING
pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
h, w = pad_target.shape[:2]
pad_target = np.expand_dims(pad_target, axis=0)
pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(device)
pad_target = pad_target.repeat(BATCH_SIZE,1,1,1)

seed = make_seed((h, w), CHANNEL_N)
# pool = SamplePool(x=np.repeat(seed[None, ...], POOL_SIZE, 0))
# batch = pool.sample(BATCH_SIZE).x

ca = CA_Model(CHANNEL_N, CELL_FIRE_RATE, device).to(device)
#ca.load_state_dict(torch.load(model_path))

optimizer = opt.Adam(ca.parameters(), lr=lr, betas=betas)
scheduler = opt.lr_scheduler.ExponentialLR(optimizer, lr_gamma)

loss_log = []
loss_f = nn.MSELoss()

def train(x, target, steps, optimizer, scheduler):
    x = ca(x, steps=steps)
    loss = loss_f(x[:, :, :, :4], target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return x, loss

# def loss_f(x, target):
#     return torch.mean(torch.pow(x[..., :4]-target, 2), [-2,-3,-1])

for i in range(n_epoch+1):
    x0 = np.repeat(seed[None, ...], BATCH_SIZE, 0)
    x0 = torch.from_numpy(x0.astype(np.float32)).to(device)

    x, loss = train(x0, pad_target, np.random.randint(64,96), optimizer, scheduler)

    step_i = len(loss_log)
    loss_log.append(loss.item())
    print(step_i, "loss =", loss.item())

    if step_i%100 == 0:
        #clear_output()
        #visualize_batch(x0.detach().cpu().numpy(), x.detach().cpu().numpy())
        #plot_loss(loss_log)
        torch.save(ca.state_dict(), model_path)