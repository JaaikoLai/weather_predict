import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import io
import os
import glob


class Model(nn.Module):
    def __init__(self, out_len=3):
        super(Model, self).__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=50, num_layers=3, )
        # self.decoder = nn.ModuleList()
        # self.decoder.append(nn.LSTMCell(input_size=50, hidden_size=50))
        # self.decoder.append(nn.LSTMCell(input_size=50, hidden_size=50))
        self.decoder = nn.LSTMCell(input_size=50, hidden_size=50)
        self.linear = nn.Linear(50, 1)
        self.out_len = out_len

    def forward(self, inputs):
        """
        :param inputs: (batch_size, seq_len, vec_len)
        :return:
        """
        batch_size = inputs.shape[0]
        outputs = torch.zeros((batch_size, self.out_len), device=inputs.device)
        hide_out, (h, c) = self.encoder(inputs.permute(1, 0, 2))#神经网络的输入尺寸为(T, N, D)
        h = h[-1, ...]
        c = c[-1, ...]
        for i in range(self.out_len):
            cur_input = self.attention(hide_out, h)
            h, c = self.decoder(cur_input, hx=(h, c))
            outputs[:, i] = self.linear(h).view(-1)
        return outputs

    @staticmethod
    def attention(encoder_hide, cur_hide):
        dist = torch.sum(encoder_hide * cur_hide[None], dim=-1)
        wt = F.softmax(dist, dim=0)
        cur_input = torch.sum(wt[..., None] * encoder_hide, dim=0)
        return cur_input


class MyDadaSet(Dataset):
    def __init__(self, seq_len=12, out_len=3):
        self.dataset_len = 1
        self.data_root = './data/round1/'
        self.seq_len = seq_len
        self.out_len = out_len
        self.data_container = np.zeros((1200, 1200, self.seq_len), dtype=np.float32)
        self.label_container = np.zeros((1200, 1200, self.out_len), dtype=np.float32)
        self.data = None
        self.label = None

    def __getitem__(self, index):
        dt = self.data[index, ...].view(-1, 1)
        lb = self.label[index, ...]
        return dt, lb

    def __len__(self):
        return self.dataset_len

    def update_sampled_img(self):
        fold_id = np.random.randint(1, 5)
        img_id = np.random.randint(1, 212-self.seq_len-self.out_len+1)
        for s in range(self.seq_len):
            img = io.imread(self.data_root+f'Z{fold_id}/Z{fold_id}-{img_id+s:03}.tif')
            self.data_container[..., s] = img / 10000

        for s in range(self.out_len):
            img = io.imread(self.data_root+f'Z{fold_id}/Z{fold_id}-{img_id+self.seq_len+s:03}.tif')
            # print(img.shape)
            # print(img.dtype)
            self.label_container[..., s] = img / 10000

        mask = np.sum(self.data_container > 0, axis=-1) > self.seq_len/2
        self.data = torch.from_numpy(self.data_container[mask, :])
        self.label = torch.from_numpy(self.label_container[mask, :])
        self.dataset_len = self.data.shape[0]
        # print(mask.shape)
        # print('data shape: ', self.data.shape, 'label shape: ', self.label.shape)
        # print(self.label.dtype)


def train(gpu_id=0):
    device = torch.device(f'cuda:{gpu_id}')
    print(device)
    epochs = 100
    batch_size = 4096

    dataset = MyDadaSet()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,)
    model_prefix = 'weights/hihi'
    os.makedirs(model_prefix, exist_ok=True)
    net = Model()
    net.to(device)

    criteria = nn.MSELoss()

    opt = optim.RMSprop(net.parameters(), lr=1e-3, weight_decay=1e-5, )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)

    for epoch in range(epochs):
        net.train()
        lr_scheduler.step()
        dataset.update_sampled_img()
        epoch_loss = 0.0
        for dt, lb in train_loader:
            dt, lb = dt.to(device), lb.to(device)
            out = net(dt)
            loss = criteria(out, lb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * dt.shape[0]

        # print(torch.max(out.detach()), torch.min(out.detach()))
        print(f'epoch: {epoch+1}, loss: {np.sqrt(epoch_loss/len(dataset)):.6f}')
        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), f'{model_prefix}/epoch_{epoch+1:03}.pth')


def infer(gpu_id=0):
    device = torch.device(f'cuda:{gpu_id}')
    data_root = 'data/round1/'
    sv_path = 'weights/submit_hihi'
    os.makedirs(sv_path, exist_ok=True)
    net = Model()
    state_dict = torch.load('weights/hihi/epoch_100.pth')
    # print(state_dict.keys())
    net.load_state_dict(state_dict)
    net.eval()
    net.to(device)

    seq_len = 12
    # fold_id = 1
    for fold_id in range(1, 5):
        img_nms = sorted(glob.glob(data_root + f'Z{fold_id}/*tif'))[-seq_len:]
        data_container = torch.zeros((1200, 1200, seq_len), dtype=torch.float32)
        for s, img_nm in enumerate(img_nms):
            img = io.imread(img_nm)
            data_container[..., s] = torch.from_numpy(img / 10000)
        data_container = data_container.reshape(-1, seq_len, 1)
        data_container = data_container.to(device)
        out_img = []
        with torch.set_grad_enabled(False):
            for i in range(1200):
                out = net(data_container[i*1200:(i+1)*1200, ...]).reshape(1, 1200, 3).cpu().detach().numpy()*10000
                out = np.ceil(out).astype(np.int16)
                out_img.append(out)
        out = np.concatenate(tuple(out_img), axis=0)
        print(out.shape)
        for i in range(3):
            io.imsave(f'{sv_path}/Z{fold_id}-21{i+3}.tif', out[..., i])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train()
    # infer()