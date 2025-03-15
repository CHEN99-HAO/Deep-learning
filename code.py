import numpy as np
from sklearn.preprocessing import StandardScaler
from lightning_lite.utilities.seed import seed_everything
import torch
import torch.nn as nn
from scipy.spatial.distance import jensenshannon
import torch.nn.functional as F
import argparse
import random
from torch.utils import data as da
from torchmetrics import MeanMetric, Accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--source_data', type=str,default=r, help='')
    parser.add_argument('--source_label', type=str,default=r, help='')
    parser.add_argument('--target_data', type=str, default=r, help='')
    parser.add_argument('--target_label', type=str, default=r, help='')
    parser.add_argument('--batch_size', type=int, default= 32,help='batchsize of the training process')
    parser.add_argument('--nepoch', type=int, default=10, help='max number of epoch')
    parser.add_argument('--num_classes', type=int, default=4, help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='initialization list')
    args = parser.parse_args()
    return args


class Dataset(da.Dataset):
    def __init__(self, X, y):
        self.Data = X
        self.Label = y

    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label

    def __len__(self):
        return len(self.Data)


def load_data():
    source_data = np.load(args.source_data)
    source_label = np.load(args.source_label).argmax(axis=-1)
    target_data = np.load(args.target_data)
    target_label = np.load(args.target_label).argmax(axis=-1)
    source_data = StandardScaler().fit_transform(source_data.T).T
    target_data = StandardScaler().fit_transform(target_data.T).T
    Train_source = Dataset(source_data, source_label)
    Train_target = Dataset(target_data, target_label)
    return Train_source, Train_target


class MMD(nn.Module):
    def __init__(self, m, n):
        super(MMD, self).__init__()
        self.m = m
        self.n = n

    def _if_list(self, list):
        if len(list) == 0:
            list = torch.tensor(list)
        else:
            list = torch.vstack(list)
        return list

    def _classification_division(self, data, label):
        N = data.size()[0]
        a, b, c, d= [], [], [], []
        for i in range(N):
            if label[i] == 0:
                a.append(data[i])
            elif label[i] == 1:
                b.append(data[i])
            elif label[i] == 2:
                c.append(data[i])
            elif label[i] == 3:
                d.append(data[i])

        return self._if_list(a), self._if_list(b), self._if_list(c), self._if_list(d)

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        l2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def MDA(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num,
                                       fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        kernel_loss = (torch.sum(XX) / (self.m * self.m)) + (torch.sum(YY) / (self.n * self.n)) - (
                    torch.sum(XY) / (self.m * self.n)) - (torch.sum(YX) / (self.n * self.m))
        return kernel_loss

    def CDA(self, output1, source_label, output2, pseudo_label):
        s_0, s_1, s_2, s_3 = self._classification_division(output1, source_label)
        t_0, t_1, t_2, t_3 = self._classification_division(output2, pseudo_label)
        CDA_loss = 0
        s1, s2, s3, s4 = 0, 0, 0, 0
        if t_0.size()[0] != 0 and s_0.size()[0] != 0:
            s1 = 1
            CDA_loss += self.MDA(s_0, t_0)
        if t_1.size()[0] != 0 and s_1.size()[0] != 0:
            s2 = 1
            CDA_loss += self.MDA(s_1, t_1)
        if t_2.size()[0] != 0 and s_2.size()[0] != 0:
            s3 = 1
            CDA_loss += self.MDA(s_2, t_2)
        if t_3.size()[0] != 0 and s_3.size()[0] != 0:
            s4 = 1
            CDA_loss += self.MDA(s_3, t_3)
        s = s1 + s2 + s3 + s4
        return CDA_loss / s


def dy_pseudo(labels, pred1, pred2):
    loss = nn.CrossEntropyLoss()(pred1, labels)
    kl_distance = nn.KLDivLoss(reduction='none', log_target=True)
    sm = torch.nn.Softmax(dim=1)
    pred11 = sm(pred1)
    pred12 = sm(pred2)
    M = 0.5 * (pred11 + pred12)
    kl1 = kl_distance(pred11.log(), M.log())
    kl2 = kl_distance(pred12.log(), M.log())
    js_divergence = 0.5 * (kl1 + kl2)
    js = torch.sum(js_divergence, dim=1)
    exp_variance = torch.exp(-js)
    loss1 = torch.mean(loss * exp_variance + js)
    return loss1


class MKWJDAN(nn.Module):
    def __init__(self):
        super(MKWJDAN, self).__init__()
        self.p1_1 = nn.Sequential(nn.Conv1d(1, 32, kernel_size=64, stride=16, padding=24),
                                  nn.BatchNorm1d(32),
                                  nn.ReLU())
        self.p1_2 = nn.MaxPool1d(2, 2)
        self.p2_1 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU())
        self.p2_2 = nn.MaxPool1d(2, 2)
        self.p3_1 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(128),
                                  nn.ReLU())
        self.p3_2 = nn.MaxPool1d(2, 2)
        self.p4_1 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU())
        self.p4_2 = nn.MaxPool1d(2, 2)
        self.p5_1 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU())
        self.p5_2 = nn.MaxPool1d(2, 2)
        self.p6 = nn.AdaptiveAvgPool1d(1)
        self.p7_1 = nn.Sequential(nn.Linear(512, 256), nn.Dropout(0.1),
                                  nn.ReLU())
        self.p7_2 = nn.Sequential(nn.Linear(256, args.num_classes))

    def forward(self, x, y):
        x = self.p1_2(self.p1_1(x))
        x = self.p2_2(self.p2_1(x))
        x = self.p3_2(self.p3_1(x))
        x = self.p4_2(self.p4_1(x))
        x = self.p5_2(self.p5_1(x))
        x = self.p7_1(self.p6(x).squeeze())
        x = self.p7_2(x)
        y = self.p1_2(self.p1_1(y))
        y = self.p2_2(self.p2_1(y))
        y = self.p3_2(self.p3_1(y))
        y = self.p4_2(self.p4_1(y))
        y = self.p5_2(self.p5_1(y))
        y = self.p7_1(self.p6(y).squeeze())
        y = self.p7_2(y)
        return x, y

    def shallow(self, x):
        x = self.p1_2(self.p1_1(x))
        x = self.p2_2(self.p2_1(x))
        x = self.p3_2(self.p3_1(x))
        x = self.p4_1(x)
        x = self.p6(x)
        x = x.squeeze()
        x = x.view(x.size(0), -1)
        x = self.p7_2(x)
        return x

    def predict(self, y):
        y = self.p1_2(self.p1_1(y))
        y = self.p2_2(self.p2_1(y))
        y = self.p3_2(self.p3_1(y))
        y = self.p4_2(self.p4_1(y))
        y = self.p5_2(self.p5_1(y))
        y = self.p7_1(self.p6(y).squeeze())
        y = self.p7_2(y)
        return y


def train(model, source_loader, target_loader, optimizer):
    model.train()
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    num_iter = len(source_loader)
    for i in range(0, num_iter):
        l = 3 / (1 + (num_iter - i) / num_iter * np.exp(i / num_iter))-1
        source_data, source_label = next(iter_source)
        target_data, _ = next(iter_target)
        source_data, source_label = source_data.to(device), source_label.to(device)
        target_data = target_data.to(device)
        optimizer.zero_grad()
        output = model.shallow(target_data.float().requires_grad_(requires_grad=False))
        output1, output2 = model(source_data.float(), target_data.float())
        clc_loss_step = nn.CrossEntropyLoss(label_smoothing=0.1)(output1, source_label.long())
        pre_pseudo_label = torch.argmax(output2, dim=-1)
        loss = dy_pseudo(pre_pseudo_label, output2, output)
        CDA_loss = MMD(source_data.size()[0], target_data.size()[0]).CDA(output1, source_label, output2, pre_pseudo_label)
        MDA_loss = MMD(source_data.size()[0], target_data.size()[0]).MDA(source=output1, target=output2)
        u = CDA_loss / (CDA_loss + MDA_loss)
        loss_step = clc_loss_step + l * ((1 - u) * MDA_loss + u * CDA_loss) + 0.1 * loss
        loss_step.backward()
        optimizer.step()
        metric_accuracy_1.update(output1.max(1)[1], source_label)
        metric_mean_1.update(loss_step)
        metric_mean_2.update(clc_loss_step)
        metric_mean_3.update(loss)
        metric_mean_4.update(CDA_loss)
        metric_mean_5.update(MDA_loss)
    train_acc = metric_accuracy_1.compute()
    train_all_loss = metric_mean_1.compute()
    source_cla_loss = metric_mean_2.compute()
    rectification_loss = metric_mean_3.compute()
    cda_loss = metric_mean_4.compute()
    mda_loss = metric_mean_5.compute()
    metric_accuracy_1.reset()
    metric_mean_1.reset()
    metric_mean_2.reset()
    metric_mean_3.reset()
    metric_mean_4.reset()
    metric_mean_5.reset()
    return train_acc, train_all_loss, source_cla_loss, rectification_loss, cda_loss, mda_loss


def test(model, target_loader_test):
    model.eval()
    iter_target = iter(target_loader_test)
    num_iter = len(target_loader_test)
    for i in range(0, num_iter):
        target_data, target_label = next(iter_target)
        target_data, target_label = target_data.to(device), target_label.to(device)
        output2 = model.predict(target_data.float())
        metric_accuracy_2.update(output2.max(1)[1], target_label)
    test_acc = metric_accuracy_2.compute()
    metric_accuracy_2.reset()
    return test_acc


if __name__ == '__main__':
    for i in range(80):
        seed_everything(i)
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        args = parse_args()
        metric_accuracy_1 = Accuracy(task="multiclass", num_classes=args.num_classes).to(device)
        metric_accuracy_2 = Accuracy(task="multiclass", num_classes=args.num_classes).to(device)
        metric_mean_1 = MeanMetric().to(device)
        metric_mean_2 = MeanMetric().to(device)
        metric_mean_3 = MeanMetric().to(device)
        metric_mean_4 = MeanMetric().to(device)
        metric_mean_5 = MeanMetric().to(device)
        Train_source, Train_target = load_data()
        combined1 = np.column_stack((Train_source.Data, Train_source.Label))
        class01 = combined1[:400]
        class02 = combined1[500:900]
        class03 = combined1[1000:1400]
        class04 = combined1[1500:1900]
        np.random.shuffle(class01)
        np.random.shuffle(class02)
        np.random.shuffle(class03)
        np.random.shuffle(class04)
        class111 = class01[:200]
        class121 = class01[200:]
        class211 = class02[:200]
        class221 = class02[200:]
        class311 = class03[:200]
        class321 = class03[200:]
        class411 = class04[:200]
        class421 = class04[200:]
        first_half_data1 = np.vstack((class111[:, :-1], class211[:, :-1], class311[:, :-1], class411[:, :-1]))
        first_half_label1 = np.vstack((class111[:, -1], class211[:, -1], class311[:, -1], class411[:, -1]
                                       ))
        first_half_label11 = first_half_label1.reshape(-1)
        first_half_data1 = np.expand_dims(first_half_data1, axis=1)
        combined = np.column_stack((Train_target.Data, Train_target.Label))
        class01 = combined[:400]
        class02 = combined[500:900]
        class03 = combined[1000:1400]
        class04 = combined[1500:1900]
        np.random.shuffle(class01)
        np.random.shuffle(class02)
        np.random.shuffle(class03)
        np.random.shuffle(class04)
        class11 = class01[:200]
        class12 = class01[200:]
        class21 = class02[:200]
        class22 = class02[200:]
        class31 = class03[:200]
        class32 = class03[200:]
        class41 = class04[:200]
        class42 = class04[200:]
        first_half_data = np.vstack((
            class11[:, :-1], class21[:, :-1], class31[:, :-1], class41[:, :-1]))
        first_half_label = np.vstack((class11[:, -1], class21[:, -1], class31[:, -1], class41[:, -1]))
        first_half_label = first_half_label.reshape(-1)
        second_half_data = np.vstack((class12[:, :-1], class22[:, :-1], class32[:, :-1], class42[:, :-1],
                                      ))
        second_half_label = np.vstack((class12[:, -1], class22[:, -1], class32[:, -1], class42[:, -1]))
        second_half_label = second_half_label.reshape(-1)
        first_half_data = np.expand_dims(first_half_data, axis=1)
        second_half_data = np.expand_dims(second_half_data, axis=1)
        Train_source = Dataset(first_half_data1, first_half_label11)
        Train_target_first_half = Dataset(first_half_data, first_half_label)
        Train_target_second_half = Dataset(second_half_data, second_half_label)
        source_loader = da.DataLoader(dataset=Train_source, batch_size=args.batch_size, shuffle=True)
        target_loader = da.DataLoader(dataset=Train_target_first_half, batch_size=args.batch_size, shuffle=True)
        target_loader_test = da.DataLoader(dataset=Train_target_second_half, batch_size=args.batch_size, shuffle=True)
        model = MKWJDAN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(0, args.nepoch):
            train_acc, train_all_loss, source_cla_loss, rectification_loss, cda_loss, mda_loss = train(
                model, source_loader, target_loader, optimizer)
            print(
                'Epoch{}, train_loss is {:.5f}, train_accuracy is {:.5f},'
                'source_cla_loss is {:.5f},cda_loss is {:.5f},mda_loss is {:.5f},'.format(epoch + 1, train_all_loss, train_acc,
                source_cla_loss, cda_loss, mda_loss))
        test_acc = test(model, target_loader_test)
        print('test_acc:{:.5f}'.format(test_acc))