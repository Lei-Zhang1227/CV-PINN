"""
添加了PDM的PINN
去除了多余的模块
添加了warmming内容:
当arg中的warmming参数为正时表示先训练IC条件warmming轮
当args中传入的warmming参数为负时表示使用初始条件重置配点直至下一个PDM行为触发
当args中不传入warmming时，照常训练；
5/27：尝试修改平滑矩阵
以前是kernel = np.array([[0, 0.1, 0],
                           [0.1, 1, 0.1],
                           [0, 0.1, 0]]) / 1.4
6/4：天添加了一个函数，用于绘制PDM过程图
"""
import pickle
from collections import OrderedDict
import imageio.v2 as imageio
import numpy as np
import psutil
import torch
from pyDOE import lhs
from ComplexLinearNN import *
import pandas as pd
from visualize import *
import os
import gc
import time
from collections import Counter
from numpy import sin, cos
from torch.nn.utils import clip_grad_norm_
import hashlib
import copy
from numpy import exp


def find_index(lst, n):
    try:
        # 如果 n 在列表中，返回其索引
        return lst.index(n)
    except ValueError:
        # 如果 n 不在列表中，返回 False
        return False


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def to_magnitude(x):
    if x == 0:
        return 'E00'
    else:
        exponent = math.floor(math.log10(abs(x)))
        return f'E{exponent:02d}'


# the deep neural network
class ComlexValueDNN(torch.nn.Module):
    def __init__(self, layers, activation, low_point, max_point, device):
        super(ComlexValueDNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1
        if activation == 'identity':
            self.activation = torch.nn.Identity
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh
        elif activation == 'relu':
            self.activation = torch.nn.ReLU
        elif activation == 'gelu':
            self.activation = torch.nn.GELU

        layer_list = list()
        layer_list.append(('layer_0', ComplexLinearFirst(layers[0], layers[1])))
        layer_list.append(('activation_0', self.activation()))
        for i in range(1, self.depth - 1):
            layer_list.append(('layer_%d' % i, ComplexLinearMidden(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), ComplexLinearMidden(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        self.ub = torch.from_numpy(max_point).float().to(device)
        self.lb = torch.from_numpy(low_point).float().to(device)

    def forward(self, x):
        x = (x - self.lb) / (self.ub - self.lb)
        out = self.layers(x)
        return out


class RealVaulueDNN(torch.nn.Module):
    def __init__(self, layers, activation, lb, ub, device, use_batch_norm=False, use_instance_norm=False):
        super(RealVaulueDNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        if activation == 'identity':
            self.activation = torch.nn.Identity
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh
        elif activation == 'relu':
            self.activation = torch.nn.ReLU
        elif activation == 'gelu':
            self.activation = torch.nn.GELU
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            if self.use_batch_norm:
                layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i + 1])))
            if self.use_instance_norm:
                layer_list.append(('instancenorm_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i + 1])))

            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        self.ub = torch.from_numpy(ub).float().to(device)
        self.lb = torch.from_numpy(lb).float().to(device)

    def forward(self, x):
        x = (x - self.lb) / (self.ub - self.lb)
        out = self.layers(x)
        return out


def state_dict_hash(state_dict):
    hasher = hashlib.md5()
    for key, tensor in state_dict.items():
        hasher.update(tensor.cpu().numpy().tobytes())
    return hasher.hexdigest()


# points_initialization, points_init, points_boundary, points_collection, points_regularization,
#     RealUV_t_x_metric, test_inputs, test_label, min_point, max_point,x_axis, t_axis

class PhysicsInformedNN():
    """PINN network"""

    def __init__(self, args, x_axis, t_axis, RealUV_t_x_metric, points_init, points_collecation, points_boundary,
                 min_points, max_point, test_inputs, test_label):
        self.device = device
        self.load_LogIter = 0
        self.current_error = 100.0
        self.current_start_time = time.time()
        self.time0 = time.time()
        self.warmming = args.warmming
        self.equation = args.equation[8:]
        self.init_num = int(args.equation[7])
        print('self.equation,self.init:', self.equation, self.init_num)
        self.N_f = args.N_f
        self.optimization = args.optimization
        self.scheduler_name = args.scheduler
        self.step_sizeORpatience = args.step_sizeORpatience
        self.gammaORfactor = args.gammaORfactor

        self.weights_of_loss = [int(item) for item in args.weight_loss.split(',')]
        layers = [int(item) for item in args.layers.split(',')]

        self.axis_x = x_axis
        self.axis_t = t_axis
        self.num_x = args.num_x
        self.num_t = args.num_t
        self.horizontal_divisions = args.t_divisions  # 时间轴上的划分大小
        self.vertical_divisions = args.x_divisions  # x轴上的划分大小
        self.real_uv_meatric = RealUV_t_x_metric
        self.epoch_adam = args.epoch_adam  # min(epoch_adam, args.restart_interval)
        self.restart_interval = args.restart_interval
        self.check_interval = args.check_interval
        self.verbose_interval = args.verbose_interval
        self.PDM_interval = [int(item) for item in args.PDM_interval.split(',')]
        self.PDM_lr = [float(item) for item in args.PDM_lr.split(',')]
        self.PDM_notion = [int(item) for item in args.PDM_notion.split(',')]
        # PED
        self.x_f = torch.tensor(points_collecation[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(points_collecation[:, 1:2], requires_grad=True).float().to(device)

        # init
        self.x_init = torch.tensor(points_init[:, 0:1], requires_grad=True).float().to(device)
        self.t_init = torch.tensor(points_init[:, 1:2], requires_grad=True).float().to(device)
        self.u_init = torch.tensor(points_init[:, 2:3], requires_grad=True).float().to(device)
        self.v_init = torch.tensor(points_init[:, 3:4], requires_grad=True).float().to(device)

        # boundary
        self.x_bc = torch.tensor(points_boundary[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc = torch.tensor(points_boundary[:, 1:2], requires_grad=True).float().to(device)
        self.u_bc = torch.tensor(points_boundary[:, 2:3], requires_grad=True).float().to(device)
        self.v_bc = torch.tensor(points_boundary[:, -1:], requires_grad=True).float().to(device)

        # Test
        self.test_input = test_inputs
        self.test_label = test_label

        # model
        if args.usecomplex:
            self.plot_label1 = 'Complex'
            self.dnn = ComlexValueDNN(layers, args.activation, min_points, max_point, device).to(device)
            print(self.dnn)
            print('--搭建complex NN--')
            print(f"Number of parameters: {self.count_parameters(self.dnn)}")
            f = open(f'Experiment_record.txt', 'a')
            f.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            f.write(f'  【完成模型搭建】，可训练参数 {self.count_parameters(self.dnn)} 个--\n')
            f.close()
        else:
            self.plot_label1 = 'Real'
            self.dnn = RealVaulueDNN(layers, args.activation, min_points, max_point, device).to(device)
            print(self.dnn)
            print('--搭建Real NN--')
            print(f"Number of parameters: {self.count_parameters(self.dnn)}")
            f = open(f'Experiment_record.txt', 'a')
            f.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            f.write(f'  【完成模型搭建】，可训练参数 {self.count_parameters(self.dnn)} 个--\n')
            f.close()
        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08,
                                               weight_decay=0, amsgrad=False)
        self.last_state_of_dnn = copy.deepcopy(self.dnn.state_dict())
        torch.save(self.last_state_of_dnn, 'origin_state_of_dnn.pth')
        if self.scheduler_name == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_adam,
                                                             step_size=self.step_sizeORpatience,
                                                             gamma=self.gammaORfactor)
        elif self.scheduler_name == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_adam, mode='min',
                                                                        factor=self.gammaORfactor,
                                                                        patience=self.step_sizeORpatience,
                                                                        threshold=1e-4,
                                                                        threshold_mode='rel', cooldown=5,
                                                                        min_lr=1e-8,
                                                                        eps=1e-8, verbose=True)
        else:
            print('Wrong scheduler_name!')

        if args.load_model_path:
            if args.find_old_points and args.old_state:
                self.last_state_of_dnn = self.load_model_state(args.old_state)
                self.dnn.load_state_dict(self.load_model_state(args.find_old_points))
                # self.PDM('LOAD_POINTS')
                self.plot_for_pdm('test')
                print(f'--loading old state from {args.old_state}')
                # self.LogIter = self.load_model_Logiter(args.load_model_path)
                # print(f'current model LogIter: {self.LogIter}')

                f = open(f'Experiment_record.txt', 'a')
                f.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                f.write(f'  【加载old_state】,路径： {args.old_state} \n')
                f.write(f'  【加载previous current_state】,路径： {args.find_old_points} \n')
                f.write(f'  【更新配点】\n')
                f.close()

            print(f"--Loading model from: {args.load_model_path}")
            self.load_state(filename=args.load_model_path)
            print(f'load {self.plot_label1} net')
            result = find_index(self.PDM_interval, self.LogIter)
            if result is not False:
                if self.PDM_lr[result] != 0.0:
                    print('修改lr')
                    for param_group in self.optimizer_adam.param_groups:
                        param_group['lr'] = self.PDM_lr[result]
            f = open(f'Experiment_record.txt', 'a')
            f.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            f.write(
                f'  【加载{self.plot_label1} NN】,模型可训练参数 {self.count_parameters(self.dnn)} 个，路径：{args.load_model_path}--\n')
            if args.old_state:
                self.last_state_of_dnn = self.load_model_state(args.old_state)
                print(f'--loading old state from {args.old_state}')
                f.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                f.write(f'  【加载old_state】,路径： {args.old_state} --\n')
            if args.restart:
                self.PDM('restart')
                loaded_state = torch.load(args.restart)
                self.dnn.load_state_dict(loaded_state)
                for param_group in self.optimizer_adam.param_groups:
                    param_group['lr'] = 0.01
                f.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                f.write(f'  【重置模型】,路径： {args.restart}，学习率初始化为 0.01 --\n')
            f.close()
        else:
            self.loss_list = []
            self.lr_list = []
            self.model_save_record = []
            self.best_error = 100.0
            self.LogIter = 0

    def plot_for_pdm(self, label):
        print('plot for PDM')
        self.dnn.eval()
        path = './PDM-plot'
        if not os.path.exists(path):
            os.makedirs(path)
        # region COMPARE
        # 生成评估点
        X, x_axis, t_axis = self.generate_grid(self.num_x, self.num_t)
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        # 计算check点的值
        current_state_of_dnn = copy.deepcopy(self.dnn.state_dict())
        u, v = self.net_uv(x, t)
        u_pred = u.detach().cpu().numpy().flatten()[:, None]
        v_pred = v.detach().cpu().numpy().flatten()[:, None]
        uv_pred = [(u * u + v * v) ** (1 / 2) for u, v in zip(u_pred, v_pred)]
        uv_pred_metric = np.array(uv_pred).reshape(len(t_axis), len(x_axis))
        self.dnn.load_state_dict(self.last_state_of_dnn)
        u, v = self.net_uv(x, t)
        u_pred = u.detach().cpu().numpy().flatten()[:, None]
        v_pred = v.detach().cpu().numpy().flatten()[:, None]
        uv_pred = [(u * u + v * v) ** (1 / 2) for u, v in zip(u_pred, v_pred)]
        uv_pred_metric_old = np.array(uv_pred).reshape(len(t_axis), len(x_axis))
        self.dnn.load_state_dict(current_state_of_dnn)
        dynamic_changes = np.abs(uv_pred_metric_old - uv_pred_metric)

        self.last_state_of_dnn = copy.deepcopy(current_state_of_dnn)
        # endregion

        # region Analysis
        dynamic_changes_matrix = dynamic_changes.reshape(self.num_t + 1, self.num_x + 1).T  # 这里是随机生成的数据
        delta_x = 40.0 / self.num_x
        delta_t = 5.0 / self.num_t
        height = (dynamic_changes_matrix.shape[0] - 1) // self.vertical_divisions
        width = (dynamic_changes_matrix.shape[1] - 1) // self.horizontal_divisions
        print(dynamic_changes_matrix.shape[0], height)
        print(dynamic_changes_matrix.shape[1], width)

        score = np.zeros((self.vertical_divisions, self.horizontal_divisions))
        position_metric = []
        # 计算每个小方格的平均值
        for i in range(self.vertical_divisions):
            current_position = []
            for j in range(self.horizontal_divisions):
                # 获取小方格的数据
                sub_matrix = dynamic_changes_matrix[i * height:(i + 1) * height, j * width:(j + 1) * width]
                # 计算平均值
                score[i, j] = np.mean(sub_matrix)
                current_position.append(
                    [i * height * delta_x - 20, (i + 1) * height * delta_x - 20, j * width * delta_t,
                     (j + 1) * width * delta_t])
            print(current_position)
            position_metric.append(current_position)

        # 输出结果
        print('--score--')
        print(score)
        score = np.array(score)
        plot_solution_with_new_points3(path, x_axis, t_axis, score, 'score', save_svg=False)
        print('plot socre success')
        plot_solution_with_new_points3(path, x_axis, t_axis, dynamic_changes_matrix, 'change', save_svg=False)

        # endregion
        # generate points
        generated_points = self.generate_new_points(score, np.array(position_metric), self.N_f)
        plot_solution_with_new_points(self.real_uv_meatric, x_axis, t_axis, path,
                                      label, generated_points[:, 0:1],
                                      generated_points[:, -1:], uv_pred_metric.reshape(self.num_t + 1, self.num_x + 1),
                                      dynamic_changes_matrix.reshape(self.num_t + 1, self.num_x + 1), save_svg=True)

        fraction = 0.6

        # 计算抽取的行数
        num_samples = int(generated_points.shape[0] * fraction)
        # 生成均匀间隔的索引
        indices = np.linspace(0, generated_points.shape[0] - 1, num_samples, dtype=int)
        # 根据索引抽取行
        sampled_data = generated_points[indices]
        print(sampled_data.shape)
        plot_solution_with_new_points2(path,
                                       label, sampled_data[:, 0:1],
                                       sampled_data[:, -1:], save_svg=False)
        self.x_f = torch.tensor(generated_points[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(generated_points[:, 1:2], requires_grad=True).float().to(device)
        print('--配点更新完毕--')

    def PDM_FOE_INIT(self, label):
        self.dnn.eval()
        path = './PDM'
        if not os.path.exists(path):
            os.makedirs(path)
        # region COMPARE
        # 生成评估点
        X, x_axis, t_axis = self.generate_grid(self.num_x, self.num_t)
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        # 计算check点的值
        u, v = self.net_uv(x, t)
        u_pred = u.detach().cpu().numpy().flatten()[:, None]
        v_pred = v.detach().cpu().numpy().flatten()[:, None]
        uv_pred = [(u * u + v * v) ** (1 / 2) for u, v in zip(u_pred, v_pred)]
        uv_pred_metric = np.array(uv_pred).reshape(len(t_axis), len(x_axis))

        x_range = [-20, 20]
        x_values = np.linspace(x_range[0], x_range[1], self.num_x + 1).reshape(-1, 1)
        if self.init_num == 0:
            u_init = np.array(exp(-(x_values ** 2 / 10))).reshape(-1, 1)
            v_init = (u_init * 0).reshape(-1, 1)
        elif self.init_num == 1:
            u_init = np.array(sin(x_values) * 2.0 / (exp(x_values) + exp(-x_values))).reshape(-1, 1)
            v_init = (u_init * 0).reshape(-1, 1)
        elif self.init_num == 2:
            u_init = np.array(cos(x_values) * 2.0 / (exp(x_values) + exp(-x_values))).reshape(-1, 1)
            v_init = (u_init * 0).reshape(-1, 1)
        else:
            raise ValueError("缺少初始条件。")
        sumuv = sum(u_init ** 2 * 0.05 + v_init ** 2 * 0.05) ** 0.5
        uv_init = np.array(u_init / sumuv).reshape(-1, 1)  # 归一化
        print('shape of pred init:', uv_init.shape)
        uv_pred_metric_old = np.tile(uv_init.T, (801, 1))
        print('shape of pred init after tile:', uv_pred_metric_old.shape)

        dynamic_changes = np.abs(uv_pred_metric_old - uv_pred_metric)
        # endregion

        # region Analysis
        dynamic_changes_matrix = dynamic_changes.reshape(self.num_t + 1, self.num_x + 1).T  # 这里是随机生成的数据
        delta_x = 40.0 / self.num_x
        delta_t = 5.0 / self.num_t
        height = (dynamic_changes_matrix.shape[0] - 1) // self.vertical_divisions
        width = (dynamic_changes_matrix.shape[1] - 1) // self.horizontal_divisions
        print(dynamic_changes_matrix.shape[0], height)
        print(dynamic_changes_matrix.shape[1], width)

        score = np.zeros((self.vertical_divisions, self.horizontal_divisions))
        position_metric = []
        # 计算每个小方格的平均值
        for i in range(self.vertical_divisions):
            current_position = []
            for j in range(self.horizontal_divisions):
                # 获取小方格的数据
                sub_matrix = dynamic_changes_matrix[i * height:(i + 1) * height, j * width:(j + 1) * width]
                # 计算平均值
                score[i, j] = np.mean(sub_matrix)
                current_position.append(
                    [i * height * delta_x - 20, (i + 1) * height * delta_x - 20, j * width * delta_t,
                     (j + 1) * width * delta_t])
            print(current_position)
            position_metric.append(current_position)

        # 输出结果
        print('--score--')
        print(score)
        score = np.array(score)

        # endregion
        # generate points
        generated_points = self.generate_new_points(score, np.array(position_metric), self.N_f)
        fraction = 0.4

        # 计算抽取的行数
        num_samples = int(generated_points.shape[0] * fraction)
        # 生成均匀间隔的索引
        indices = np.linspace(0, generated_points.shape[0] - 1, num_samples, dtype=int)
        # 根据索引抽取行
        sampled_data = generated_points[indices]
        print(sampled_data.shape)
        plot_solution_with_new_points2(path,
                                       label, sampled_data[:, 0:1],
                                       sampled_data[:, -1:], save_svg=True)
        self.x_f = torch.tensor(generated_points[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(generated_points[:, 1:2], requires_grad=True).float().to(device)
        print('--配点更新完毕--')
        self.last_state_of_dnn = copy.deepcopy(self.dnn.state_dict())
        print('--重置old state 为当前状态--')

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def generate_new_points(self, score, value, num):
        '''
        :param score: 取点权重矩阵
        :param value: 网格范围
        :param num: 配点总量
        :return:
        '''
        total_value = np.sum(score)
        # 根据socre矩阵计算每个位置生成点的数量（差异越大分配的配点越多）
        points_per_position = (score / total_value) * num
        points_per_position = np.round(points_per_position).astype(int)  # 取整数
        print('original points_per_position:', points_per_position)

        # 计算原始总和
        original_sum = np.sum(points_per_position)
        # 对矩阵进行平滑处理 为了不至于让配点完全集中在某些区域进而丧失其它区域的优化，配点矩阵进行平滑处理；
        kernel = np.array([[0, 0.3, 0],
                           [0.3, 1, 0.3],
                           [0, 0.3, 0]]) / 2.2
        smoothed_matrix = np.copy(points_per_position).astype(int)
        rows, cols = points_per_position.shape

        for i in range(rows):
            for j in range(cols):
                # 取周围的点来计算均值
                sub_matrix = points_per_position[max(0, i - 1):min(rows, i + 2), max(0, j - 1):min(cols, j + 2)]
                smoothed_matrix[i, j] = np.sum(sub_matrix * kernel[:sub_matrix.shape[0], :sub_matrix.shape[1]])

        # 计算平滑后的总和并调整比例
        smoothed_sum = np.sum(smoothed_matrix)
        scale_factor = original_sum / smoothed_sum
        adjusted_matrix = np.round(smoothed_matrix * scale_factor).astype(int)
        # 打印调整后的矩阵

        adjusted_matrix[adjusted_matrix == 0] = 1
        print('adjusted points_per_position:', adjusted_matrix)
        # 确保调整后的总和与原始总和一致

        # 生成点
        generated_points = np.array([]).reshape(-1, 2)
        # 遍历B的每个元素，生成对应数量的点
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                # 当前位置的顶点范围
                x_min, x_max, y_min, y_max = value[i, j][0], value[i, j][1], value[i, j][2], value[i, j][3]
                # 当前位置需要生成的点的数量
                num_points = adjusted_matrix[i, j]
                # 在给定范围内生成点
                points = np.array([x_min, y_min]) + (np.array([x_max, y_max]) - np.array([x_min, y_min])) * lhs(2,
                                                                                                                int(num_points))
                # 将生成的点添加到列表中
                generated_points = np.vstack((generated_points, points))
        # self.x_f = torch.tensor(generated_points[:, 0:1], requires_grad=True).float().to(device)
        # self.t_f = torch.tensor(generated_points[:, 1:2], requires_grad=True).float().to(device)
        return generated_points

    def PDM(self, label):
        self.dnn.eval()
        path = './PDM'
        if not os.path.exists(path):
            os.makedirs(path)
        # region COMPARE
        # 生成评估点
        X, x_axis, t_axis = self.generate_grid(self.num_x, self.num_t)
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        # 计算check点的值
        current_state_of_dnn = copy.deepcopy(self.dnn.state_dict())
        u, v = self.net_uv(x, t)
        u_pred = u.detach().cpu().numpy().flatten()[:, None]
        v_pred = v.detach().cpu().numpy().flatten()[:, None]
        uv_pred = [(u * u + v * v) ** (1 / 2) for u, v in zip(u_pred, v_pred)]
        uv_pred_metric = np.array(uv_pred).reshape(len(t_axis), len(x_axis))
        self.dnn.load_state_dict(self.last_state_of_dnn)
        u, v = self.net_uv(x, t)
        u_pred = u.detach().cpu().numpy().flatten()[:, None]
        v_pred = v.detach().cpu().numpy().flatten()[:, None]
        uv_pred = [(u * u + v * v) ** (1 / 2) for u, v in zip(u_pred, v_pred)]
        uv_pred_metric_old = np.array(uv_pred).reshape(len(t_axis), len(x_axis))
        self.dnn.load_state_dict(current_state_of_dnn)
        dynamic_changes = np.abs(uv_pred_metric_old - uv_pred_metric)
        self.last_state_of_dnn = copy.deepcopy(current_state_of_dnn)
        # endregion

        # region Analysis
        dynamic_changes_matrix = dynamic_changes.reshape(self.num_t + 1, self.num_x + 1).T  # 这里是随机生成的数据
        delta_x = 40.0 / self.num_x
        delta_t = 5.0 / self.num_t
        height = (dynamic_changes_matrix.shape[0] - 1) // self.vertical_divisions
        width = (dynamic_changes_matrix.shape[1] - 1) // self.horizontal_divisions
        print(dynamic_changes_matrix.shape[0], height)
        print(dynamic_changes_matrix.shape[1], width)

        score = np.zeros((self.vertical_divisions, self.horizontal_divisions))
        position_metric = []
        # 计算每个小方格的平均值
        for i in range(self.vertical_divisions):
            current_position = []
            for j in range(self.horizontal_divisions):
                # 获取小方格的数据
                sub_matrix = dynamic_changes_matrix[i * height:(i + 1) * height, j * width:(j + 1) * width]
                # 计算平均值
                score[i, j] = np.mean(sub_matrix)
                current_position.append(
                    [i * height * delta_x - 20, (i + 1) * height * delta_x - 20, j * width * delta_t,
                     (j + 1) * width * delta_t])
            print(current_position)
            position_metric.append(current_position)

        # 输出结果
        print('--score--')
        print(score)
        score = np.array(score)

        # endregion
        # generate points
        generated_points = self.generate_new_points(score, np.array(position_metric), self.N_f)
        #         plot_solution_with_new_points(self.real_uv_meatric, x_axis, t_axis, path,
        #                                       label, generated_points[:, 0:1],
        #                                       generated_points[:, -1:], uv_pred_metric.reshape(self.num_t + 1, self.num_x + 1),
        #                                       uv_pred_metric_old.reshape(self.num_t + 1, self.num_x + 1), save_svg=True)
        fraction = 0.4

        # 计算抽取的行数
        num_samples = int(generated_points.shape[0] * fraction)
        # 生成均匀间隔的索引
        indices = np.linspace(0, generated_points.shape[0] - 1, num_samples, dtype=int)
        # 根据索引抽取行
        sampled_data = generated_points[indices]
        print(sampled_data.shape)
        plot_solution_with_new_points2(path,
                                       label, sampled_data[:, 0:1],
                                       sampled_data[:, -1:], save_svg=True)
        self.x_f = torch.tensor(generated_points[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(generated_points[:, 1:2], requires_grad=True).float().to(device)
        print('--配点更新完毕--')
        # endregion

    def generate_grid(self, num_x, num_t):
        x_range = [-20, 20]
        t_range = [0, 5]
        x_values = np.linspace(x_range[0], x_range[1], num_x + 1)
        t_values = np.linspace(t_range[0], t_range[1], num_t + 1)
        X, T = np.meshgrid(x_values, t_values)
        x_point = X.flatten()[:, None]
        t_point = T.flatten()[:, None]
        points_grid = np.hstack((x_point, t_point))
        return points_grid, x_values, t_values

    def net_uv(self, x, t):
        """The standard DNN that takes (x_r,t_r) --> u,v."""
        out = self.dnn(torch.cat([x, t], dim=1))
        u = out[:, 0:1]
        v = out[:, -1:]
        return u, v

    def PDE_residual(self, x, t):
        """
        Manually implement physics constraint:
        Autograd for calculating the residual for different systems.
        手动的计算DNN的近似的方程的残差。
        """
        u, v = self.net_uv(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
        #
        if self.equation == '1':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v

        elif self.equation == '2':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u

        elif self.equation == '3':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v - 0.2 * x * x * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u - 0.2 * x * x * u
        elif self.equation == '5':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v - 0.5 * torch.cos(
                x) * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u - 0.5 * torch.cos(
                x) * u
        elif self.equation == '6':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v - 0.5 * torch.sin(
                x) * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u - 0.5 * torch.sin(
                x) * u
        elif self.equation == '4':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v - 0.5 * torch.sin(
                x) ** 2 * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u - 0.5 * torch.sin(
                x) ** 2 * u
        elif self.equation == '7':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v - 0.5 * torch.cos(
                x) ** 2 * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u - 0.5 * torch.cos(
                x) ** 2 * u
        elif self.equation == '8':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v - (
                    v * (1.0 / torch.cosh(x)) ** 2 + u * torch.tanh(x) / torch.cosh(x))
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u - (
                    u * (1.0 / torch.cosh(x)) ** 2 - v * torch.tanh(x) / torch.cosh(x))
        elif self.equation == '10':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v - (
                    0.5 * torch.cos(x) + 0.2 * x * x) * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u - (
                    0.5 * torch.cos(x) + 0.2 * x * x) * u
        elif self.equation == '11':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v - 0.5 * torch.sin(
                x) * v - 0.2 * x * x * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u - 0.5 * torch.sin(
                x) * u - 0.2 * x * x * u
        elif self.equation == '12':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v - 0.5 * torch.cos(
                x) ** 2 * v - 0.2 * x * x * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u - 0.5 * torch.cos(
                x) ** 2 * u - 0.2 * x * x * u
        else:
            f_im = None
            f_real = None
            print('【ERROR】--no equation can calculate residual--')
        return f_im, f_real

    def loss_pinn(self):
        """ Loss function. """
        if torch.is_grad_enabled():
            self.optimizer_adam.zero_grad()

        # -------PDE-------
        #         time1 = time.time()
        pde_Comlex, pde_Real = self.PDE_residual(self.x_f, self.t_f)
        loss_pde_real = torch.mean(pde_Real ** 2)
        loss_pde_complex = torch.mean(pde_Comlex ** 2)
        loss_PDE = loss_pde_real + loss_pde_complex
        #         time2 = time.time()
        # -------Init-------
        u_pred_init, v_pred_init = self.net_uv(self.x_init, self.t_init)
        loss_u_init = torch.mean((self.u_init - u_pred_init) ** 2)
        loss_v_init = torch.mean((self.v_init - v_pred_init) ** 2)
        loss_init = loss_u_init + loss_v_init

        # -------BC-------
        u_pred_bc, v_pred_bc = self.net_uv(self.x_bc, self.t_bc)
        loss_bc_u = torch.mean((self.u_bc - u_pred_bc) ** 2)
        loss_bc_v = torch.mean((self.v_bc - v_pred_bc) ** 2)
        loss_BC = loss_bc_u + loss_bc_v
        if self.warmming and self.LogIter <= self.warmming:
            print(
                f'only reduce init and bc:{self.LogIter}, {self.warmming}, self.LogIter <= self.warmming is {self.LogIter <= self.warmming}')
            loss = loss_init + loss_BC
        else:
            loss = self.weights_of_loss[0] * loss_init + self.weights_of_loss[1] * loss_PDE + self.weights_of_loss[
                2] * loss_BC

        now_lr = self.optimizer_adam.state_dict()['param_groups'][0]['lr']  # 当前学习率查看

        self.LogIter += 1
        if loss.requires_grad:
            loss.backward()
        clip_grad_norm_(self.dnn.parameters(), max_norm=1)

        if self.LogIter % self.verbose_interval == 0:
            self.current_end_time = time.time()
            current_elapsed_time_hours = self.current_end_time - self.current_start_time
            print(
                'epoch %d, loss: %.5e, loss_init: %.5e, loss_bc: %.5e,loss_PDE: %.5e,current_elapsed_time_hours: %d,now_lr: %2e' % (
                    self.LogIter, loss.item(), loss_init.item(), loss_BC.item(), loss_PDE.item(),
                    current_elapsed_time_hours, now_lr))
            loss_item = [loss.item(), loss_init.item(), loss_BC.item(), loss_PDE.item()]
            self.loss_list.append(loss_item)
            self.lr_list.append(now_lr)
            if self.LogIter % self.check_interval == 0:
                self.test()
                if self.LogIter % self.restart_interval == 0 and self.LogIter != 0:
                    memory_allocated = torch.cuda.memory_allocated()
                    print('--【after training】')
                    print(f"--当前GPU内存分配: {memory_allocated / 1073741824} G")
                    memory = psutil.virtual_memory()
                    print(f"--Total memory: {memory.total / (1024 ** 3):.2f} GB")
                    print(f"--Used memory: {memory.used / (1024 ** 3):.2f} GB")
                    print(f"--Memory usage: {memory.percent}%")
                    self.save_checkpoint_state('checkpoint_' + str(self.LogIter))
                    self.evaluation(self.test_input, self.test_label, self.LogIter)
                    time1 = time.time()
                    training_time = "{:.2f}".format((time1 - self.time0) / 3600)
                    f = open(f'Experiment_record.txt', 'a')
                    f.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                    f.write(f'  【{self.LogIter}轮训练完成】,当前error：{self.current_error},阶段用时：{training_time} h\n')
                    f.close()
                    self.time0 = time.time()

                result = find_index(self.PDM_interval, self.LogIter)
                if result is not False:
                    print(f"--【PDM】: 更换配点,调整学习率至 {self.PDM_lr[result]}--")
                    self.PDM(self.LogIter)
                    f = open(f'Experiment_record.txt', 'a')
                    f.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                    f.write(f'  【PDM】,更换配点,调整学习率至 {self.PDM_lr[result]}\n')
                    if self.PDM_lr[result] != 0.0:
                        for param_group in self.optimizer_adam.param_groups:
                            param_group['lr'] = self.PDM_lr[result]
                    if self.PDM_notion[result] == 1:
                        loaded_state = torch.load('origin_state_of_dnn.pth')
                        self.dnn.load_state_dict(loaded_state)
                        print('--【初始化模型】--')
                        f.write(f'  【PDM】,初始化模型参数\n')
                    f.close()
            self.current_start_time = time.time()
            if self.LogIter % abs(self.warmming) == 0 and self.warmming and self.warmming < 0:
                self.PDM_FOE_INIT('INIT_WARMMING')
                print('重置配点成功')
                self.warmming = 1
        return loss

    def save_checkpoint(self, filename='checkpoint'):
        state = {
            'epoch': self.LogIter,
            'model': self.dnn.state_dict(),
            'optimizer': self.optimizer_adam.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss_list': self.loss_list,
            'save_record': self.model_save_record,
            'lr_list': self.lr_list,
        }
        print(self.LogIter)
        torch.save(state, filename + '.pth.tar')
        torch.save(self.dnn, filename + '.pkl')
        f = open(f'Experiment_record.txt', 'a')
        f.write(f'--【存档{filename}】：已完成{self.LogIter}次迭代----当前最佳error:{self.model_save_record[-1]}--\n')
        f.close()

    def save_checkpoint_state(self, filename='checkpoint'):
        state = {
            'epoch': self.LogIter,
            'model': self.dnn.state_dict(),
            'optimizer': self.optimizer_adam.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss_list': self.loss_list,
            'save_record': self.model_save_record,
            'lr_list': self.lr_list,
        }
        torch.save(state, filename + '.pth.tar')

    def load_model_state(self, filename='checkpoint'):
        checkpoint = torch.load(filename + '.pth.tar')
        return checkpoint['model']

    def load_model_Logiter(self, filename='checkpoint'):
        checkpoint = torch.load(filename + '.pth.tar')
        return checkpoint['epoch']

    def load_checkpoint(self, filename='checkpoint'):
        '''
        加载模型、定义优化器、加载优化器状态、加载LogIter、加载loss_list
        :param filename:
        :return:
        '''
        checkpoint = torch.load(filename + '.pth.tar')
        self.dnn = torch.load(filename + '.pkl')
        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08,
                                               weight_decay=0, amsgrad=False)
        if self.scheduler_name == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_adam, step_size=self.step_sizeORpatience,
                                                             gamma=self.gammaORfactor)
        elif self.scheduler_name == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_adam, mode='min',
                                                                        factor=self.gammaORfactor,
                                                                        patience=self.step_sizeORpatience,
                                                                        threshold=1e-4,
                                                                        threshold_mode='rel', cooldown=5, min_lr=1e-8,
                                                                        eps=1e-8, verbose=True)
        else:
            print('Wrong scheduler_name!')

        self.optimizer_adam.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.LogIter = checkpoint['epoch']
        self.loss_list = checkpoint['loss_list']
        self.model_save_record = checkpoint['save_record']
        self.lr_list = checkpoint['lr_list']
        self.best_error = float(self.model_save_record[-1][1])
        print('after load: len(self.loss_list):', len(self.loss_list))
        f = open(f'Experiment_record.txt', 'a')
        f.write(f'--【读档{filename}】:开始第{self.LogIter + 1}次迭代----当前最佳error:{self.model_save_record[-1]}--\n')
        f.close()

    def load_state(self, filename='checkpoint'):
        '''
        加载模型、定义优化器、加载优化器状态、加载LogIter、加载loss_list
        :param filename:
        :return:
        '''
        checkpoint = torch.load(filename + '.pth.tar')
        self.optimizer_adam.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.LogIter = checkpoint['epoch']
        self.loss_list = checkpoint['loss_list']
        self.model_save_record = checkpoint['save_record']
        self.lr_list = checkpoint['lr_list']
        self.best_error = float(self.model_save_record[-1][1])
        self.dnn.load_state_dict(checkpoint['model'])
        print('after load: len(self.loss_list):', len(self.loss_list))
        f = open(f'Experiment_record.txt', 'a')
        f.write(f'--【读档{filename}】:开始第{self.LogIter + 1}次迭代----当前最佳error:{self.model_save_record[-1]}--\n')
        f.close()

    def train(self):
        self.dnn.train()
        print(r'Training Process is going, current lr is '.format(
            self.optimizer_adam.state_dict()['param_groups'][0]['lr']))
        for i in range(self.epoch_adam):
            loss = self.loss_pinn()
            self.optimizer_adam.step()
            self.scheduler.step(loss)
        # region record
        f = open(f'Experiment_record.txt', 'a')
        f.write(f'-- 可训练参数梯度量级 --\n')
        gradients = []
        for param in self.dnn.parameters():
            if param.grad is not None:
                gradients.append([param.grad.min(), param.grad.max()])
        gradients = [[to_magnitude(item) for item in sublist] for sublist in gradients]
        flat_list = [item for sublist in gradients for item in sublist]
        element_counts = Counter(flat_list)
        f.write(f'-- {element_counts} --\n')
        f.close()
        # endregion

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.dnn.eval()
        u, v = self.net_uv(x, t)
        u = u.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        return u, v

    def plot_residual(self, mode, num_x=400, num_t=1000):
        self.dnn.eval()
        path = './Residual/'
        if not os.path.exists(path):
            os.makedirs(path)
        if mode == 'FDM':
            X, x_axis, t_axis = self.generate_grid(num_x, num_t)
            u, v = self.predict(X)
            U_t_x_metric = torch.Tensor(u.reshape(num_t + 1, num_x + 1)).view(num_t + 1, 1, num_x + 1)
            V_t_x_metric = torch.Tensor(v.reshape(num_t + 1, num_x + 1)).view(num_t + 1, 1, num_x + 1)
            f_im, f_real = self.calculate_residual_FDM(t_axis, x_axis, U_t_x_metric, V_t_x_metric)  # 得到的残差矩阵形状为t*x
            plot_double_residual(f_im, f_real, x_axis, t_axis, path, save_title=f'Residual({mode})',
                                 label2=self.LogIter)
            plot_single_residual(torch.add(f_im, f_real), x_axis, t_axis, path='', title='The PDE Residual(FDM) of ',
                                 save_title=f'ALL-Residual({mode})',
                                 label2=self.LogIter)
            print(f'【{self.LogIter}】---------Residual(FDM) Plot Created---------')
        elif mode == 'AD':
            X, x_axis, t_axis = self.generate_grid(num_x, num_t)
            x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
            t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
            f_im, f_real = self.calculate_residual_AD(x, t)
            print(f_im)
            f_im = np.abs(f_im.detach().cpu().numpy().reshape(num_t + 1, num_x + 1))
            f_real = np.abs(f_real.detach().cpu().numpy().reshape(num_t + 1, num_x + 1))
            plot_double_residual(f_im, f_real, x_axis, t_axis, path, save_title=f'Residual({mode})',
                                 label2=self.LogIter)
            plot_single_residual(torch.add(f_im, f_real), x_axis, t_axis, path=path, title='The PDE Residual(FDM) of ',
                                 save_title=f'ALL-Residual({mode})',
                                 label2=self.LogIter)
            print(f'【{self.LogIter}】---------Residual(AD) Plot Created---------')
        elif mode == 'all':
            X, x_axis, t_axis = self.generate_grid(num_x, num_t)
            x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
            t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
            f_im, f_real = self.calculate_residual_AD(x, t)
            f_im = np.abs(f_im.detach().cpu().numpy().reshape(num_t + 1, num_x + 1))
            f_real = np.abs(f_real.detach().cpu().numpy().reshape(num_t + 1, num_x + 1))
            plot_double_residual(f_im, f_real, x_axis, t_axis, path, save_title=f'Residual(AD)',
                                 label2=self.LogIter)
            plot_single_residual(np.add(f_im, f_real), x_axis, t_axis, path=path, title='The PDE Residual(FDM) of ',
                                 save_title='ALL-Residual(FDM)',
                                 label2=self.LogIter)
            u, v = self.predict(X)
            U_t_x_metric = torch.Tensor(u.reshape(num_t + 1, num_x + 1)).view(num_t + 1, 1, num_x + 1)
            V_t_x_metric = torch.Tensor(v.reshape(num_t + 1, num_x + 1)).view(num_t + 1, 1, num_x + 1)
            f_im, f_real = self.calculate_residual_FDM(t_axis, x_axis, U_t_x_metric, V_t_x_metric)  # 得到的残差矩阵形状为t*x
            plot_double_residual(f_im, f_real, x_axis, t_axis, path, save_title=f'Residual(FDM)',
                                 label2=self.LogIter)
            plot_single_residual(torch.add(f_im, f_real), x_axis, t_axis, path=path, title='The PDE Residual(FDM) of ',
                                 save_title='ALL-Residual(AD)',
                                 label2=self.LogIter)
            print(f'【{self.LogIter}】---------Residual(FDM＆AD) Plot Created---------')
        else:
            print('【ERROR】--wrong mode when plotting residual--')

    def plot_error(self, error):
        '''
        绘制误差分布图
        :param pred:
        :return:
        '''
        path = './Error/'
        if not os.path.exists(path):
            os.makedirs(path)
        x_range = [-20, 20]
        t_range = [0, 5]
        x_values = np.linspace(x_range[0], x_range[1], error.shape[1])
        t_values = np.linspace(t_range[0], t_range[1], error.shape[0])
        plot_single_residual(error, x_values, t_values, path, title='Absolute Error', save_title='Error',
                             label2=self.LogIter, save_svg=True, Aspect_Ratio=1.5,
                             show_title=False)
        print(f'【{self.LogIter}】---------Absolute Error Plot Created---------')
        return x_values, t_values

    def calculate_residual_AD(self, x, t):
        """
        Manually implement physics constraint:
        Autograd for calculating the residual for different systems.
        手动的计算DNN的近似的方程的残差。
        """
        u, v = self.net_uv(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=False)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=False)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=False)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=False, create_graph=False)[0]
        #
        if self.equation == '1':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v

        elif self.equation == '2':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u

        elif self.equation == '3':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v - 0.2 * x * x * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u - 0.2 * x * x * u
        else:
            f_im = None
            f_real = None
            print('【ERROR】--no equation can calculate residual--')
        del u, v, u_x, u_t, u_xx, v_x, v_t, v_xx
        gc.collect()
        return f_im, f_real

    def plot_Gradient(self, num_x, num_t):
        self.dnn.eval()
        path = './Grsdient/'
        if not os.path.exists(path):
            os.makedirs(path)
        X, x_axis, t_axis = self.generate_grid(num_x, num_t)
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        u, v = self.net_uv(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=False)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=False)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=False)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=False, create_graph=False)[0]
        u_t = np.abs(u_t.detach().cpu().numpy().reshape(len(t_axis), len(x_axis)))
        u_x = np.abs(u_x.detach().cpu().numpy().reshape(len(t_axis), len(x_axis)))
        v_t = np.abs(v_t.detach().cpu().numpy().reshape(len(t_axis), len(x_axis)))
        v_x = np.abs(v_x.detach().cpu().numpy().reshape(len(t_axis), len(x_axis)))
        plot_double_residual(u_t, u_x, x_axis, t_axis, path, save_title=f'Gradient(u)',
                             label2=self.LogIter)
        plot_double_residual(v_t, v_x, x_axis, t_axis, path, save_title=f'Gradient(v)',
                             label2=self.LogIter)
        print(f'【{self.LogIter}】---------Gradient Plot Created---------')
        del u_x, u_t, v_x, v_t
        gc.collect()
        return u, v, x_axis, t_axis

    def test(self):
        self.dnn.eval()
        X, x_axis, t_axis = self.generate_grid(self.num_x, self.num_t)
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        u, v = self.net_uv(x, t)
        u_pred = u.detach().cpu().numpy().flatten()[:, None]
        v_pred = v.detach().cpu().numpy().flatten()[:, None]
        uv_pred = [(u * u + v * v) ** (1 / 2) for u, v in zip(u_pred, v_pred)]
        uv_pred_metric = np.array(uv_pred).reshape(len(t_axis), len(x_axis))
        x_ratio = (len(x_axis) - 1) / (len(self.axis_x) - 1)
        t_ratio = (len(t_axis) - 1) / (len(self.axis_t) - 1)
        real_uv_meatric = self.real_uv_meatric
        if x_ratio > 1.0:
            uv_pred_metric = uv_pred_metric[:, 0:len(x_axis):int(x_ratio)]
        elif x_ratio < 1.0:
            real_uv_meatric = real_uv_meatric[:, 0:len(self.axis_x):int((len(self.axis_x) - 1) / (len(x_axis) - 1))]
        if t_ratio > 1.0:
            uv_pred_metric = uv_pred_metric[0:len(t_axis):int(t_ratio), :]
        elif t_ratio < 1.0:
            real_uv_meatric = real_uv_meatric[0:len(self.axis_t):int((len(self.axis_t) - 1) / (len(t_axis) - 1)), :]
        assert uv_pred_metric.shape == real_uv_meatric.shape
        error = np.abs(uv_pred_metric - real_uv_meatric)
        self.current_error = np.mean(error)
        x_range = [-20, 20]
        t_range = [0, 5]
        x_values = np.linspace(x_range[0], x_range[1], error.shape[1])
        t_values = np.linspace(t_range[0], t_range[1], error.shape[0])
        path = './Solution/'
        if not os.path.exists(path):
            os.makedirs(path)
        plot_solution_with_error(real_uv_meatric, uv_pred_metric, error, x_values, t_values, path, self.LogIter,
                                 save_svg=False)
        error_u_abs = np.mean(error)
        if self.best_error > error_u_abs:
            self.best_error = error_u_abs
            self.model_save_record.append([self.LogIter, error_u_abs])
            self.save_and_record()
        self.dnn.train()

    def save_and_record(self, filename='BestModel'):
        state = {
            'epoch': self.LogIter,
            'optimizer': self.optimizer_adam.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss_list': self.loss_list,
            'save_record': self.model_save_record,
            'lr_list': self.lr_list,
            'model': self.dnn.state_dict(),
        }
        torch.save(state, filename + '.pth.tar')
        torch.save(self.dnn, filename + '.pkl')

    def plot_history(self):
        loss_list = np.array(self.loss_list)
        iterations_all = len(loss_list)
        x_axis = np.arange(1, iterations_all + 1, 1) * 50 / 1000
        Aspect_Ratio = float(2 * 4 / 3)
        # fig = plt.figure(figsize=(calculate_fig_size(Aspect_Ratio)[0] / 2, calculate_fig_size(Aspect_Ratio)[1]))
        fig, ax = plt.subplots(figsize=(calculate_fig_size(Aspect_Ratio)[0] / 2, calculate_fig_size(Aspect_Ratio)[1]))
        plt.rcParams['font.family'] = 'DejaVu Serif'
        fontdict = {'fontsize': 6, 'fontweight': 'normal', 'fontname': 'DejaVu Serif'}
        all = loss_list[:, 1:2] + loss_list[:, 2:3] + loss_list[:, 3:4] + loss_list[:, 4:5]
        ax.plot(x_axis, loss_list[:, 1:2], linestyle='-.', color=(254 / 255, 183 / 255, 5 / 255, 0.8),
                linewidth=0.5,
                label='I.C.Loss')
        ax.plot(x_axis, loss_list[:, 2:3], linestyle='-.', color=(19 / 255, 103 / 255, 158 / 255, 0.8), alpha=0.8,
                linewidth=0.5, label='B.C.Loss')
        ax.plot(x_axis, loss_list[:, 3:4], linestyle='-.', color=(42 / 255, 157 / 255, 142 / 255, 0.8), alpha=0.8,
                linewidth=0.5, label='E.Loss')
        if loss_list[:, 4:5][25] != 0.0:
            ax.plot(x_axis, loss_list[:, 4:5], linestyle='-.', color='b', alpha=0.5, linewidth=0.5, label='D.Loss')
        # plt.plot(x_axis, loss_list[:, 5:6], linestyle='-.', color='y', alpha=0.7, linewidth=1, label='R.Loss')
        ax.plot(x_axis, all, linestyle='-', color=(239 / 255, 65 / 255, 67 / 255, 0.9), linewidth=0.5,
                label='Total Loss')
        ax.set_xlabel('Epochs($\\times 10^{3}$)', fontdict=fontdict, labelpad=4, )  # 设置x轴刻度标签
        ax.set_ylabel('MAE', fontdict=fontdict, labelpad=4)  # 设置y轴刻度标签
        ax.set_ylim(0, 0.00005)
        ax.set_xlim(0, x_axis[-1])
        ax.legend(fontsize=6, frameon=False, markerscale=3)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

        # 设置x轴和y轴的刻度密度
        ax.locator_params(axis='x', nbins=15)  # 设置x轴的刻度线数量
        ax.locator_params(axis='y', nbins=15)

        offset_text = ax.xaxis.get_offset_text()
        offset_text.set_fontsize(6)
        offset_text.set_position((0, 0))

        ax.yaxis.set_major_formatter(CustomFormatter())  # 假设CustomFormatter是一个自定义格式化器
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.yaxis.get_offset_text().set_position((-0.03, 0))  # 调整偏移文本的位置
        ax.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置

        plt.savefig('loss_history.png', dpi=500, bbox_inches='tight', transparent=True)
        plt.close()

    def evaluation(self, inputs, outputs, name=None):
        # R2
        referred_u = outputs[:, 0:1]
        referred_v = outputs[:, 1:2]
        referred_uv = outputs[:, 2:3]
        outputs_u, outputs_v = self.predict(inputs)
        outputs_uv = np.sqrt(np.multiply(outputs_u, outputs_u) + np.multiply(outputs_v, outputs_v))

        u_mean = np.mean(referred_u)
        SSR_u = np.sum(np.square(outputs_u - u_mean))
        SSE_u = np.sum(np.square(outputs_u - referred_u))
        R2_u = SSR_u / (SSR_u + SSE_u)

        v_mean = np.mean(referred_v)
        SSR_v = np.sum(np.square(outputs_v - v_mean))
        SSE_v = np.sum(np.square(outputs_v - referred_v))
        R2_v = SSR_v / (SSR_v + SSE_v)

        uv_mean = np.mean(referred_uv)
        SSR_uv = np.sum(np.square(outputs_uv - uv_mean))
        SSE_uv = np.sum(np.square(outputs_uv - referred_uv))
        R2_uv = SSR_uv / (SSR_uv + SSE_uv)

        R2 = np.array([[R2_u, R2_v, R2_uv]])

        # MAE
        mae_u = np.mean(np.abs(referred_u - outputs_u))
        mae_v = np.mean(np.abs(referred_v - outputs_v))
        mae_uv = np.mean(np.abs(referred_uv - outputs_uv))
        mae = np.array([[mae_u, mae_v, mae_uv]])

        # MSE
        mse_u = np.mean(np.square(referred_u - outputs_u))
        mse_v = np.mean(np.square(referred_v - outputs_v))
        mse_uv = np.mean(np.square(referred_uv - outputs_uv))
        mse = np.array([[mse_u, mse_v, mse_uv]])

        evaluation_list = np.concatenate([R2, mae], axis=0)
        evaluation_list = np.concatenate([evaluation_list, mse], axis=0)
        if name is None:
            name = self.LogIter

        pd.DataFrame(evaluation_list, columns=['u', 'v', 'uv'], index=['R2', 'MAE', 'MSE']).to_csv(
            f'{name}.csv',
            encoding='utf-8')

    def generate_git(self):
        count = 0
        for filename in os.listdir('./Solution'):
            if filename.endswith(".png"):
                print('png !')
                count += 1
        images = []
        print(count)
        for i in range(0, count):
            filename = './Solution/' + str(i * self.check_interval) + '.png'
            images.append(imageio.imread(filename))
        imageio.mimsave('Training-process.gif', images, duration=3)

    def return_training_epoch(self):
        return self.LogIter
