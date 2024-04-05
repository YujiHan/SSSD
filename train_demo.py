import os
import numpy as np
import torch
import torch.nn as nn

from utils.util import print_size, training_loss, calc_diffusion_hyperparams


from imputers.SSSDS4Imputer import SSSDS4Imputer


def get_mask_rand(sample, k):
    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))  # lenght of series indexes
    perm = torch.randperm(len(length_index))
    idx = perm[0:k]
    mask[idx, :] = 0

    return mask


def get_mask_given(sample, miss_list):
    mask = torch.ones(sample.shape)
    mask[miss_list, :] = 0

    return mask


global trainset_config
global diffusion_hyperparams
global model_config

config = {
    "diffusion_config": {"T": 200, "beta_0": 0.0001, "beta_T": 0.02},
    "wavenet_config": {
        "in_channels": 128,
        "out_channels": 128,
        "num_res_layers": 36,
        "res_channels": 256,
        "skip_channels": 256,
        "diffusion_step_embed_dim_in": 128,
        "diffusion_step_embed_dim_mid": 512,
        "diffusion_step_embed_dim_out": 512,
        "s4_lmax": 100,
        "s4_d_state": 64,
        "s4_dropout": 0.0,
        "s4_bidirectional": 1,
        "s4_layernorm": 1,
    },
    "train_config": {
        "output_directory": "/home/hanyuji/data/SSSD_results/zebrafish",
        "ckpt_iter": -1,
        "iters_per_ckpt": 1000,
        "iters_per_logging": 100,
        "n_iters": 150000,
        "learning_rate": 2e-4,
        "only_generate_missing": 1,
        "use_model": 2,
        "masking": "rm",
        "missing_k": 3,
    },
    "trainset_config": {
        "train_data_path": "/home/hanyuji/data/mujoco_dataset/train_mujoco.npy",
        "test_data_path": "/home/hanyuji/data/SSSD_results/VAE_result/vae_10.npy",
        "segment_length": 100,
        "sampling_rate": 100,
    },
    "gen_config": {
        "output_directory": "/home/hanyuji/data/SSSD_results/zebrafish",
        "ckpt_path": "/home/hanyuji/data/SSSD_results/zebrafish",
    },
}

train_config = config["train_config"]  # training parameters
trainset_config = config["trainset_config"]  # to load trainset
model_config = config['wavenet_config']
diffusion_hyperparams = calc_diffusion_hyperparams(
    **config["diffusion_config"]
)  # dictionary of all diffusion hyperparameters


# train(**train_config)

output_directory = train_config['output_directory']
ckpt_iter = train_config['ckpt_iter']
iters_per_ckpt = train_config['iters_per_ckpt']
iters_per_logging = train_config['iters_per_logging']
n_iters = train_config['n_iters']
learning_rate = train_config['learning_rate']
only_generate_missing = train_config['only_generate_missing']
masking = train_config['masking']
missing_k = train_config['missing_k']


"""
Train Diffusion Models

Parameters:
output_directory (str):         save model checkpoints to this path
ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                automatically selects the maximum iteration if 'max' is selected
data_path (str):                path to dataset, numpy array.
n_iters (int):                  number of iterations to train
iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                default is 10k, for models with residual_channel=64 this number can be larger
iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
learning_rate (float):          learning rate

use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
missing_k (int):                k missing time steps for each feature across the sample length.
"""


# map diffusion hyperparameters to gpu
for key in diffusion_hyperparams:
    if key != "T":
        diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

# predefine model
net = SSSDS4Imputer(**model_config).cuda()
print_size(net)

# define optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


### data loading and reshaping ###

# training_data = np.load(trainset_config['train_data_path'])
training_data = np.load(
    '/mnt/sdb/hanyuji-data/SSSD_results/wot_result/gene_traj_VAE_latent.npy'
)

### norm ###
training_data = training_data / 100
### norm ###

print(training_data.shape)
training_data = np.split(training_data, 47, 0)
training_data = np.array(training_data)
print(training_data.shape)

training_data = torch.from_numpy(training_data).float().cuda()
print('Data loaded')

# 8000, 100, 14)
# (160, 50, 100, 14)
# Data loaded

# (9400, 12, 2000)
# (47, 200, 12, 2000)


# training
n_iter = ckpt_iter + 1
while n_iter < n_iters + 1:
    for batch in training_data:

        transposed_mask = get_mask_rand(batch[0], missing_k)

        mask = transposed_mask.permute(1, 0)
        mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
        loss_mask = ~mask.bool()
        batch = batch.permute(0, 2, 1)

        assert batch.size() == mask.size() == loss_mask.size()

        # back-propagation
        optimizer.zero_grad()
        X = batch, batch, mask, loss_mask
        loss = training_loss(
            net,
            nn.MSELoss(),
            X,
            diffusion_hyperparams,
            only_generate_missing=only_generate_missing,
        )

        loss.backward()
        optimizer.step()

        if n_iter % iters_per_logging == 0:
            print("iteration: {} \tloss: {}".format(n_iter, loss.item()))

        # save checkpoint
        if n_iter > 0 and n_iter % iters_per_ckpt == 0:
            checkpoint_name = '{}.pkl'.format(n_iter)
            torch.save(
                {
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                os.path.join(output_directory, checkpoint_name),
            )
            print('model at iteration %s is saved' % n_iter)

        n_iter += 1
