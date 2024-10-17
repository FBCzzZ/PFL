import numpy as np
import copy
import torch
from models.server.serverbase import Serverbase


class Serverour(Serverbase):
    def __init__(self, args):
        super().__init__(args)
        global_model_weights = self.get_cla_weight()
        self.momentum = {key: torch.zeros_like(val) for key, val in global_model_weights.items()}
        # 动量系数 beta
        self.beta = 0.9
        # 学习率
        self.lr = 0.01

    def agg_spec(self, w_FFT_list):
        w_FFT_np = np.vstack(w_FFT_list)
        w_FFT_glob = np.mean(w_FFT_np, axis=0)
        return w_FFT_glob

    def agg_cla_Momentum(self, client_weights):
        global_weights = self.get_cla_weight()
        # 聚合客户端的更新
        avg_weights = copy.deepcopy(global_weights)
        for key in avg_weights.keys():
            avg_weights[key] = torch.mean(torch.stack([client_weights[i][key] for i in range(len(client_weights))]), dim=0)

        # 计算动量并更新全局模型
        for key in global_weights.keys():
            # 动量更新
            self.momentum[key] = self.beta * self.momentum[key] + (avg_weights[key] - global_weights[key])
            # 用动量更新全局模型
            global_weights[key] += self.lr * self.momentum[key]
        return global_weights


    def get_ExFeature_weight(self):
        conv_weights = {}
        for name, param in self.net.named_parameters():
            if 'conv' in name or 'fc1' in name or 'fc2' in name:
                conv_weights[name] = param.data
        return conv_weights

    def get_cla_weight(self):
        cla_weights = {}
        for name, param in self.net.named_parameters():
            if 'fc3' in name:
                cla_weights[name] = param.data
        return cla_weights

    def update_weight_ExFeature(self, conv_weights):
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                if 'conv' in name or 'fc1' in name or 'fc2' in name:
                    param.data.copy_(conv_weights[name])

    def update_weight_classifier(self, fc_weights):
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                if 'fc3' in name:
                    param.data.copy_(fc_weights[name])

