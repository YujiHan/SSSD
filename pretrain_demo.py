import numpy as np
import torch
from torch import optim
from tqdm import tqdm


from imputers.VAE_model import VanillaVAE

from dataset.dataset_utils import loadSCData

ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(
    "zebrafish", "three_interpolation"
)
data = ann_data.X
traj_data = [data[np.where(cell_tps == t)[0], :] for t in range(1, n_tps + 1)]


data_np = np.load('/mnt/sdb/hanyuji-data/SSSD_results/wot_result/gene_traj_009_by5.npy')
print(data_np.shape)  # (12, 9582, 2000)
data_np = data_np.transpose(1, 0, 2)
print(data_np.shape)  # (9582, 12, 2000)


result = []
for i in range(int(data_np.shape[0] / 200)):
    result.append(data_np[i * 200 : (i + 1) * 200, :, :])
data_np = np.asarray(result)
print(data_np.shape)  # (47, 200, 12, 2000)


# 配置
input_features = 2000
latent_dim = 128
epochs = 10
learning_rate = 1e-3
device = torch.device("cuda:0")

# 初始化模型和优化器
model = VanillaVAE(input_features, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
model.train()
for epoch in tqdm(range(epochs)):
    overall_loss = 0
    for batch in data_np:
        for item in batch:
            item = torch.tensor(item).float().to(device)

            optimizer.zero_grad()

            # 前向传播
            recons, input, mu, log_var = model(item)

            # 计算损失
            loss_dict = model.loss_function(recons, input, mu, log_var)
            loss = loss_dict['loss']

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            overall_loss += loss.item()

    print(f'Epoch {epoch}, Average Loss: {overall_loss}')

print("Training complete")


# 定义保存路径
save_path = '/mnt/sdb/hanyuji-data/SSSD_results/VAE_result/VAE_by5.pth'

# 保存模型状态字典
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
