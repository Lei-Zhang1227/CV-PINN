"""
修改了绘图逻辑，使其适应于长时间情景
添加了double RV-PINN
"""
import pickle
from collections import OrderedDict
import imageio.v2 as imageio
import numpy as np
import torch

from ComplexLinearNN import *
import pandas as pd
from visualize import *
import os
import gc
import time
from collections import Counter
from numpy import sin, cos
from torch.nn.utils import clip_grad_norm_

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


class DoubleRealVaulueDNN(torch.nn.Module):
    def __init__(self, layers, activation, lb, ub, device, use_batch_norm=False, use_instance_norm=False):
        super(DoubleRealVaulueDNN, self).__init__()

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

        layer_list1 = list()
        for i in range(self.depth - 1):
            layer_list1.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            if self.use_batch_norm:
                layer_list1.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i + 1])))
            if self.use_instance_norm:
                layer_list1.append(('instancenorm_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i + 1])))

            layer_list1.append(('activation_%d' % i, self.activation()))

        layer_list1.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict1 = OrderedDict(layer_list1)

        layer_list2 = list()
        for i in range(self.depth - 1):
            layer_list2.append(
                ('layer2_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            if self.use_batch_norm:
                layer_list2.append(('batchnorm2_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i + 1])))
            if self.use_instance_norm:
                layer_list2.append(('instancenorm2_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i + 1])))

            layer_list2.append(('activation2_%d' % i, self.activation()))

        layer_list2.append(
            ('layer2_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict2 = OrderedDict(layer_list2)

        # deploy layers
        self.layers1 = torch.nn.Sequential(layerDict1)
        self.layers2 = torch.nn.Sequential(layerDict2)
        self.ub = torch.from_numpy(ub).float().to(device)
        self.lb = torch.from_numpy(lb).float().to(device)

    def forward(self, x):
        x = (x - self.lb) / (self.ub - self.lb)
        out_u = self.layers1(x)
        out_v = self.layers2(x)
        out = torch.cat([out_u, out_v], dim=1)
        return out


# points_initialization, points_init, points_boundary, points_collection, points_regularization,
#     RealUV_t_x_metric, test_inputs, test_label, min_point, max_point,x_axis, t_axis

class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)

    def forward(self, input):
        '''
        这里的self.resol一般是指比如dx ** 2这样子；
        :param input:
        :return:
        '''
        derivative = self.filter(input)
        return derivative / self.resol


class PhysicsInformedNN():
    """PINN network"""

    def __init__(self, args, x_axis, t_axis, RealUV_t_x_metric, points_collecation,
                 points_init, points_boundary, points_initialization, points_regularization, points_observation,
                 min_points, max_point, writer, training_load=False):

        self.load_LogIter = 0
        self.time3 = time.time()
        self.current_start_time = time.time()
        self.time_max = args.time_max
        self.num_t = args.num_t
        self.num_x = args.num_x

        self.equation = args.equation[8:]
        print('self.equation:', self.equation)
        self.optimization = args.optimization
        self.scheduler_name = args.scheduler
        self.step_sizeORpatience = args.step_sizeORpatience
        self.gammaORfactor = args.gammaORfactor
        self.observation = args.observation
        self.regularization = args.regularization
        self.initialization = args.initialization
        self.add_noise = args.add_noise
        self.discret_step = args.discret_step
        self.weights_of_loss = [int(item) for item in args.weight_loss.split(',')]
        layers = [int(item) for item in args.layers.split(',')]

        self.axis_x = x_axis
        self.axis_t = t_axis
        self.real_uv_meatric = RealUV_t_x_metric
        self.epoch_adam = args.epoch_adam  # min(epoch_adam, args.restart_interval)
        self.epoch_data = args.epoch_data
        self.restart_interval = args.restart_interval
        self.training_load = training_load
        self.epoch_lbfgs = args.epoch_lbfgs

        self.writer = writer
        self.check_interval = args.check_interval
        self.verbose_interval = args.verbose_interval

        # initialization
        if self.initialization:
            print('--initialization--')
            self.x_initialization = torch.tensor(points_initialization[:, 0:1], requires_grad=True).float().to(device)
            self.t_initialization = torch.tensor(points_initialization[:, 1:2], requires_grad=True).float().to(device)
            self.u_initialization = torch.tensor(points_initialization[:, 2:3], requires_grad=True).float().to(device)
            self.v_initialization = torch.tensor(points_initialization[:, 3:4], requires_grad=True).float().to(device)
        # observation
        if self.observation:
            print('--observation--')
            self.x_observe = torch.tensor(points_observation[:, 0:1], requires_grad=True).float().to(device)
            self.t_observe = torch.tensor(points_observation[:, 1:2], requires_grad=True).float().to(device)
            self.u_observe = torch.tensor(points_observation[:, 2:3], requires_grad=True).float().to(device)
            self.v_observe = torch.tensor(points_observation[:, 3:4], requires_grad=True).float().to(device)
            if self.add_noise:
                self.noise_std = args.noise_std
                noise_u = torch.tensor(torch.randn(self.u_observe.size()) * self.noise_std).float().to(device)
                noise_v = torch.tensor(torch.randn(self.v_observe.size()) * self.noise_std).float().to(device)
                self.u_observe = self.u_observe + noise_u
                self.v_observe = self.v_observe + noise_v
        if self.regularization:
            print('--regularization--')
            self.x_reg = torch.tensor(points_regularization['x'], requires_grad=True).float().to(device)
            self.t_reg = torch.tensor(points_regularization['t'], requires_grad=True).float().to(device)
            self.r_x_axis = points_regularization['x_axis']
            self.r_t_axis = points_regularization['t_axis']
        # init
        self.x_init = torch.tensor(points_init[:, 0:1], requires_grad=True).float().to(device)
        self.t_init = torch.tensor(points_init[:, 1:2], requires_grad=True).float().to(device)
        self.u_init = torch.tensor(points_init[:, 2:3], requires_grad=True).float().to(device)
        self.v_init = torch.tensor(points_init[:, 3:4], requires_grad=True).float().to(device)
        # collocation
        self.x_f = torch.tensor(points_collecation[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(points_collecation[:, 1:2], requires_grad=True).float().to(device)
        # boundary
        self.x_bc = torch.tensor(points_boundary[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc = torch.tensor(points_boundary[:, 1:2], requires_grad=True).float().to(device)
        self.u_bc = torch.tensor(points_boundary[:, 2:3], requires_grad=True).float().to(device)
        self.v_bc = torch.tensor(points_boundary[:, -1:], requires_grad=True).float().to(device)
        # model
        if args.load_model_path is not None and not training_load:
            print(f"--Loading model from: {args.load_model_path}")
            self.load_checkpoint(filename=args.load_model_path)
            self.save_checkpoint('old_checkpoint')
            model_str = self.dnn.__str__()
            if model_str[:5] == 'Comle':
                self.plot_label1 = 'Complex'
            else:
                self.plot_label1 = 'Real'
            print(f'加载{self.plot_label1}网络')
            f = open(f'Experiment_record.txt', 'a')
            f.write(f'----{self.plot_label1}NN,模型可训练参数 {self.count_parameters(self.dnn)} 个--\n')
            f.close()
        elif training_load:
            self.load_checkpoint()
            self.save_checkpoint_state(f'checkpoint_{self.LogIter}')
            if self.epoch_data and self.LogIter >= self.epoch_data:
                # 进去PDE模式；
                self.weights_of_loss = [100, 100, 1, 100, 1]
            model_str = self.dnn.__str__()
            if model_str[:5] == 'Comle':
                self.plot_label1 = 'Complex'
            else:
                self.plot_label1 = 'Real'
        else:
            if args.usecomplex:
                self.plot_label1 = 'Complex'
                self.dnn = ComlexValueDNN(layers, args.activation, min_points, max_point, device).to(device)
                print('--搭建complex NN--')
                print(f"Number of parameters: {self.count_parameters(self.dnn)}")
                f = open(f'Experiment_record.txt', 'a')
                f.write(f'--模型可训练参数 {self.count_parameters(self.dnn)} 个--\n')
                f.close()
            elif args.double:
                self.plot_label1 = 'Doubel Real'
                self.dnn = DoubleRealVaulueDNN(layers, args.activation, min_points, max_point, device).to(device)
                print(self.dnn)
                print('--搭建Double Real NN--')
                print(f"Number of parameters: {self.count_parameters(self.dnn)}")
                f = open(f'Experiment_record.txt', 'a')
                f.write(f'--模型可训练参数 {self.count_parameters(self.dnn)} 个--\n')
                f.close()
            else:
                self.plot_label1 = 'Real'
                self.dnn = RealVaulueDNN(layers, args.activation, min_points, max_point, device).to(device)
                print(self.dnn)
                print('--搭建Real NN--')
                print(f"Number of parameters: {self.count_parameters(self.dnn)}")
                f = open(f'Experiment_record.txt', 'a')
                f.write(f'--模型可训练参数 {self.count_parameters(self.dnn)} 个--\n')
                f.close()
            self.optimizer_lbfgs = torch.optim.LBFGS(self.dnn.parameters(),
                                                     max_iter=self.epoch_lbfgs,
                                                     history_size=200, tolerance_grad=1e-40, tolerance_change=1e-40,
                                                     line_search_fn="strong_wolfe")
            self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08,
                                                   weight_decay=0, amsgrad=False)
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
            self.loss_list = []
            self.lr_list = []
            self.model_save_record = []
            self.best_error = 100.0
            self.LogIter = 0

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def generate_grid(self, num_x, num_t):
        x_range = [-20, 20]
        t_range = [0, self.time_max]
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
        elif self.equation == '6':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v - 0.5 * torch.sin(
                x) * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u - 0.5 * torch.sin(
                x) * u
        elif self.equation == '5':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v - 0.5 * torch.cos(
                x) * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u - 0.5 * torch.cos(
                x) * u
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

    def calculate_regularization(self):
        regulation_list = []
        random_numbers = [random.randint(0, 49) for _ in range(5)]
        for i in random_numbers:
            time_0 = time.time()
            u_reg, v_reg = self.net_uv(self.x_reg[i * 80000:(i + 1) * 80000], self.t_reg[i * 80000:(i + 1) * 80000])
            time_0_1 = time.time()
            uv_reg = torch.add(torch.multiply(u_reg, u_reg), torch.multiply(v_reg, v_reg))
            time_0_2 = time.time()
            # uv_reg = uv_reg.reshape(1, len(self.r_x_axis))
            uv_reg = uv_reg[1:] * self.discret_step
            time_0_3 = time.time()
            regulation_current = torch.sum(uv_reg)
            time_0_4 = time.time()
            regulation_list.append(regulation_current - 1.0)
            time_0_5 = time.time()
            #             del uv_reg, u_reg, v_reg, regulation_current
            #             gc.collect()
            #             torch.cuda.empty_cache()
            time_1 = time.time()
            time_1_all = time_0_1 - time_0 + time_0_2 - time_0_1 + time_0_3 - time_0_2 + time_0_4 - time_0_3 + time_0_5 - time_0_4 + time_1 - time_0_5
            elapsed_time_hours = time_1 - time_0
        #             print(
        #                 f'【{self.LogIter}】[{i}]:{elapsed_time_hours}/{time_1_all}--{time_0_1 - time_0}--{time_0_2 - time_0_1}--{time_0_3 - time_0_2}--{time_0_4 - time_0_3}--{time_0_5 - time_0_4}--{time_1 - time_0_5}')
        regulation_list = torch.abs(torch.Tensor(regulation_list))
        regulation = torch.mean(regulation_list.to(device))
        time_2 = time.time()
        #         print(f'【{self.LogIter}】[calculate_regularization]:{time_2 - time_1}')
        #         del regulation_list
        #         gc.collect()
        #         torch.cuda.empty_cache()
        return regulation

    def loss_pinn(self):
        """ Loss function. """
        if torch.is_grad_enabled():
            self.optimizer_lbfgs.zero_grad()
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

        # -------observation-------
        if self.observation:
            u_pred_observ, v_pred_observ = self.net_uv(self.x_observe, self.t_observe)
            loss_observ_u = torch.mean((self.u_observe - u_pred_observ) ** 2)
            loss_observ_v = torch.mean((self.v_observe - v_pred_observ) ** 2)
            loss_observe = loss_observ_u + loss_observ_v
        else:
            loss_observe = torch.tensor(0)

        # -------regularization-------
        if self.regularization:
            regulation = self.calculate_regularization()
        else:
            regulation = torch.tensor(0)

        loss = self.weights_of_loss[0] * loss_init + self.weights_of_loss[1] * loss_PDE + self.weights_of_loss[
            2] * loss_BC + self.weights_of_loss[3] * loss_observe + self.weights_of_loss[4] * regulation

        now_lr = self.optimizer_adam.state_dict()['param_groups'][0]['lr']  # 当前学习率查看
        if self.LogIter % self.verbose_interval == 0:
            self.current_end_time = time.time()
            current_elapsed_time_hours = self.current_end_time - self.current_start_time
            print(
                'epoch %d, loss: %.5e, loss_init: %.5e, loss_bc: %.5e,loss_PDE: %.5e,loss_observe: %.5e,regularization: %.5e,current_elapsed_time_hours: %d,now_lr: %2e' % (
                    self.LogIter, loss.item(), loss_init.item(), loss_BC.item(), loss_PDE.item(),
                    loss_observe.item(), regulation.item(), current_elapsed_time_hours, now_lr))
            loss_item = [loss.item(), loss_init.item(), loss_BC.item(), loss_PDE.item(), loss_observe.item(),
                         regulation.item()]
            #             print(f'PDE Loss 计算耗时{time2 - time1}s; loss.backward()耗时{self.time3 - time1}s')
            # loss_absolute_item = [(X * 100 / sum(loss_item[1:])) for X in loss_item]
            self.loss_list.append(loss_item)
            self.lr_list.append(now_lr)
            if self.LogIter % self.check_interval == 0:
                self.test()
            self.current_start_time = time.time()
        # self.writer.add_scalar(tag="loss-all", scalar_value=loss.item(), global_step=self.LogIter)
        self.LogIter += 1
        if self.epoch_data and self.LogIter == self.epoch_data:
            self.weights_of_loss = [100, 100, 1, 100, 1]
        #         self.time3 = time.time()
        if loss.requires_grad:
            loss.backward()
        clip_grad_norm_(self.dnn.parameters(), max_norm=1.0)
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

    def load_checkpoint(self, filename='checkpoint'):
        '''
        加载模型、定义优化器、加载优化器状态、加载LogIter、加载loss_list
        :param filename:
        :return:
        '''
        checkpoint = torch.load(filename + '.pth.tar')
        self.dnn = torch.load(filename + '.pkl')
        self.optimizer_lbfgs = torch.optim.LBFGS(self.dnn.parameters(),
                                                 max_iter=self.epoch_lbfgs,
                                                 history_size=200, tolerance_grad=1e-40, tolerance_change=1e-40,
                                                 line_search_fn="strong_wolfe")
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

    def train(self):
        self.dnn.train()
        if self.optimization == 'AL' or self.optimization == 'A':
            print("---- adam start ----")
            if self.scheduler_name == 'ReduceLROnPlateau':
                print(self.epoch_adam, self.load_LogIter, self.LogIter)
                for i in range(min(self.epoch_adam + self.load_LogIter - self.LogIter, self.restart_interval)):
                    loss = self.loss_pinn()
                    self.optimizer_adam.step()
                    self.scheduler.step(loss)
            else:
                for i in range(min(self.epoch_adam + self.load_LogIter - self.LogIter, self.restart_interval)):
                    loss = self.loss_pinn()
                    self.optimizer_adam.step()
                    self.scheduler.step()
        print(self.epoch_adam, self.load_LogIter)
        print(self.LogIter)
        if (self.optimization == 'AL' and (
                self.epoch_adam + self.load_LogIter) == self.LogIter) or self.optimization == 'L':
            print("---- lbfgs start ----")
            f = open(f'Experiment_record.txt', 'a')
            f.write(
                f'-- Adam 优化器训练完成，lbfgs start，当前epoch{self.LogIter}，loss_list{len(self.loss_list)}个记录 --\n')
            f.close()
            self.optimizer_lbfgs.step(self.loss_pinn)
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
        t_range = [0, self.time_max]
        x_values = np.linspace(x_range[0], x_range[1], error.shape[1])
        t_values = np.linspace(t_range[0], t_range[1], error.shape[0])
        # plot_single_residual(error, x_values, t_values, path, title='Absolute Error', save_title='Error',
        #                      label2=self.LogIter, save_svg=True, Aspect_Ratio=1.5,
        #                      show_title=False)
        # print(f'【{self.LogIter}】---------Absolute Error Plot Created---------')
        return x_values, t_values

    def calculate_residual_FDM(self, t_axis, x_axis, u, v):
        """

        :param t_axis:
        :param x_axis:
        :param u:
        :param v:
        :return:
        """
        delat_t = t_axis[2] - t_axis[1]
        delat_x = x_axis[2] - x_axis[1]
        dt = Conv1dDerivative(DerFilter=[[[-1, 1, 0]]],
                              resol=(delat_t * 1),
                              kernel_size=3,
                              name='partial_t')
        laplace = Conv1dDerivative(DerFilter=[[[-1 / 12, 16 / 12, -30 / 12, 16 / 12, -1 / 12]]],
                                   resol=(delat_x ** 2),
                                   kernel_size=5,
                                   name='laplace_operator')
        u_xx = laplace(u)
        u_xx = u_xx[1:-1, :, :]
        v_xx = laplace(v)
        v_xx = v_xx[1:-1, :, :]
        u_conv_for_t = u.permute(2, 1, 0)
        v_conv_for_t = v.permute(2, 1, 0)
        u_t = dt(u_conv_for_t)
        u_t = u_t.permute(2, 1, 0)[:, :, 2:-2]
        v_t = dt(v_conv_for_t)
        v_t = v_t.permute(2, 1, 0)[:, :, 2:-2]
        u = u[1:-1, :, 2:-2]
        v = v[1:-1, :, 2:-2]
        if self.equation == '1':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v

        elif self.equation == '2':
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u

        elif self.equation == '3':
            X, T = np.meshgrid(x_axis, t_axis)
            X = torch.Tensor(X).view(len(t_axis), 1, len(x_axis))[1:-1, :, 2:-2]
            f_im = u_t + 0.5 * v_xx - 2 * v * u * u - 2 * v * v * v - 2 * u * u * u * u * v - 4 * u * u * v * v * v - 2 * v * v * v * v * v - 0.2 * X * X * v
            f_real = 0.5 * u_xx - 1 * v_t - 2 * u * u * u - 2 * u * v * v - 2 * u * u * u * u * u - 4 * u * u * u * v * v - 2 * v * v * v * v * u - 0.2 * X * X * u
        else:
            f_im = None
            f_real = None
            print('【ERROR】--no equation can calculate residual--')

        f_im = torch.cat((f_im[0:1, :, :], f_im, f_im[-1:, :, :]), dim=0)
        f_im = torch.cat((f_im[:, :, 0:2], f_im, f_im[:, :, -2:]), dim=2).squeeze(1)
        f_im = np.abs(f_im)

        f_real = torch.cat((f_real[0:1, :, :], f_real, f_real[-1:, :, :]), dim=0)
        f_real = torch.cat((f_real[:, :, 0:2], f_real, f_real[:, :, -2:]), dim=2).squeeze(1)
        f_real = np.abs(f_real)
        return f_im, f_real

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
        # u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=False)[0]
        # u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=False)[0]
        # v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=False)[0]
        # v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=False, create_graph=False)[0]
        # u_t = np.abs(u_t.detach().cpu().numpy().reshape(len(t_axis), len(x_axis)))
        # u_x = np.abs(u_x.detach().cpu().numpy().reshape(len(t_axis), len(x_axis)))
        # v_t = np.abs(v_t.detach().cpu().numpy().reshape(len(t_axis), len(x_axis)))
        # v_x = np.abs(v_x.detach().cpu().numpy().reshape(len(t_axis), len(x_axis)))
        # plot_double_residual(u_t, u_x, x_axis, t_axis, path, save_title=f'Gradient(u)',
        #                      label2=self.LogIter)
        # plot_double_residual(v_t, v_x, x_axis, t_axis, path, save_title=f'Gradient(v)',
        #                      label2=self.LogIter)
        # print(f'【{self.LogIter}】---------Gradient Plot Created---------')
        # del u_x, u_t, v_x, v_t
        # gc.collect()
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
        if self.LogIter % 5000 == 0:
            path = './Solution/'
            if not os.path.exists(path):
                os.makedirs(path)
            plot_solution_with_error(real_uv_meatric, uv_pred_metric, error, x_values, t_values, path, self.LogIter,
                                     save_svg=False)
            # self.plot_residual(mode='all')

        error_u_abs = np.mean(error)
        if self.best_error > error_u_abs:
            self.best_error = error_u_abs
            self.model_save_record.append([self.LogIter, error_u_abs])
            self.save_and_record()
        self.dnn.train()

    def predict_plot(self, x, t, x_axis, t_axis, RX):
        self.dnn.eval()
        path = './test/'
        if not os.path.exists(path):
            os.makedirs(path)
        u, v = self.net_uv(torch.Tensor(x).float().to(device), torch.Tensor(t).float().to(device))
        u_pred = u.detach().cpu().numpy().flatten()[:, None]
        v_pred = v.detach().cpu().numpy().flatten()[:, None]
        uv_pred = [(u * u + v * v) ** (1 / 2) for u, v in zip(u_pred, v_pred)]
        uv_pred_metric = np.array(uv_pred).reshape(len(t_axis), len(x_axis))
        plot_solutions_2(uv_pred_metric, RX, x_axis, t_axis, path, label1='test',
                         label2='')

    def save_and_record(self, filename='BestModel'):
        state = {
            'epoch': self.LogIter,
            'optimizer': self.optimizer_adam.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss_list': self.loss_list,
            'save_record': self.model_save_record,
            'lr_list': self.lr_list,
        }
        torch.save(state, filename + '.pth.tar')
        torch.save(self.dnn, filename + '.pkl')

    def save_log_data(self):
        file_name = "loss_list.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(self.loss_list, open_file)
        open_file.close()

        file_name = "model_save_record.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(self.model_save_record, open_file)
        open_file.close()

    def train_for_initialization(self):
        self.dnn.train()
        pass

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

        plt.savefig('loss_history.svg', bbox_inches='tight', transparent=True)
        plt.close()

    def evaluation(self, inputs, outputs):
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

        pd.DataFrame(evaluation_list, columns=['u', 'v', 'uv'], index=['R2', 'MAE', 'MSE']).to_csv(
            f'metric{self.LogIter}.csv',
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
