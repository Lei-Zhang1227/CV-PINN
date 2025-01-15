"""
针对PINN-PDM的train模块
4/26
1.优化了record部分
2.去除了warmming部分，将其嵌入到PDM_interval中
3.完全使用--PDM_interval,--PDM_lr,--PDM_notion 三个参数来控制配点的更新；其中：
PDM_interval：列表，指定需要进行PDM的时刻；
PDM_lr：列表，与PDM_interval形状一致，指定是否需要在PDM后调整学习率、调整学习率的数值；（若为0.0，则认为学习率不变）
PDM_notion：列表，与PDM_interval形状一致，指定是否需要重置模型至初始状态；
"""
# -*-coding:utf-8 -*-
import argparse
import time
import torch
import platform
import subprocess
from data_process import generate_data
from torch.utils.tensorboard import SummaryWriter
from PINN_PDM import *
import numpy as np
################
# Arguments
################
import psutil
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'Ture'
torch.cuda.empty_cache()


# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     np.random.seed(seed)  # Numpy module.
#     random.seed(seed)  # Python random module.
#     os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    parser = argparse.ArgumentParser(description='LEI ZHANG')

    parser.add_argument('--Project_num', type=str, default='test',
                        help='Save path,usually named like 1101-1')
    parser.add_argument('--equation', type=str, default='1Dse-03',
                        help="The system is designed for studying different scenarios. "
                             'In each option, 1D represents the dimensionality of the equation,'
                             'while 01 corresponds to GPE1 with the initial conditions.')
    parser.add_argument('--seed', type=int, default=1227, help='Random initialization.')
    parser.add_argument('--time_max', type=float, default=5.0, help='--')
    parser.add_argument('--num_x', type=int, default=400, help='--')
    parser.add_argument('--num_t', type=int, default=1000, help='--')
    parser.add_argument('--t_divisions', type=int, default=5, help='--')
    parser.add_argument('--x_divisions', type=int, default=16, help='--')

    parser.add_argument('--verbose_interval', type=int, default=50, help='Epoch to show loss')
    parser.add_argument('--check_interval', type=int, default=2500,
                        help='Epoch to check error,it should be a multiple of verbose_interval.')
    parser.add_argument('--restart_interval', type=int, default=10000,
                        help='save and restart')
    parser.add_argument('--PDM_interval', type=str, default=[10000, 20000, 30000], help='change points')
    parser.add_argument('--PDM_lr', type=str, default=[5e-03, 5e-04, 1e-04], help='change lr')
    parser.add_argument('--PDM_notion', type=str, default=[0, 0, 0], help='0不重置，重置模型为初始状态')

    parser.add_argument('--weight_loss', type=str, default='100,100,1,1,1',
                        help='Weight of loss: Init, PDE, BC, Label, Regularization')
    parser.add_argument('--layers', type=str, default='2,180, 140, 160, 140 ,1',
                        help='Dimensions/layers of the NN, minus the first layer.')
    parser.add_argument('--usecomplex', default=False, action='store_true', help='whether to use complex NN or not')
    parser.add_argument('--activation', default='gelu', help='Activation to use in the network.')

    parser.add_argument('--optimization', default='A')
    parser.add_argument('--N_f', type=int, default=80000, help='Number of collocation points to sample.')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau')
    parser.add_argument('--step_sizeORpatience', type=int, default=200)
    parser.add_argument('--gammaORfactor', type=float, default=0.98)

    parser.add_argument('--epoch_adam', default=50000, type=int, help='epoch numbers for adam optimizer')
    parser.add_argument('--epoch_warmming', default=None, type=int, help='epoch numbers for adam optimizer')
    parser.add_argument('--load_model_path', default=None, type=str,
                        help='checkpoint path for pretraining')
    parser.add_argument('--old_state', default=None, type=str,
                        help='checkpoint path for pretraining')
    parser.add_argument('--find_old_points', default=None, type=str,
                        help='checkpoint path for pretraining')
    parser.add_argument('--restart', default=None, type=str,
                        help='checkpoint path for pretraining')
    parser.add_argument('--warmming', default=None, type=int,
                        help='checkpoint path for pretraining')

    parser.add_argument('--bc', default=False, action='store_true', help='whether to use bc or not')
    parser.add_argument('--observation', default=False, action='store_true',
                        help='whether to use observational_data or not')
    parser.add_argument('--initialization', default=False, action='store_true',
                        help='whether to use initialization or not')
    parser.add_argument('--regularization', default=False, action='store_true',
                        help='whether to use regularization or not')

    args = parser.parse_args()
    train(args)


def log_writer(args):
    print('-- write information --')
    f = open(f'Experiment_record.txt', 'a')
    f.write(f'\n**********************【{args.Project_num}】**********************\n')
    f.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + '    zhang lei \n')
    f.write('--------------【System Configuration】--------------\n')
    f.write(f'--CPU: {platform.processor()}\n')
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        f.write(f'--Number of GPUs: {num_gpus}\n')
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # 转换为 GB
            f.write(f'--GPU {i}: {gpu_name} with {total_memory:.2f} GB\n')
    else:
        f.write('--CUDA is not available\n')
    f.write('--------------【  Basic Information 】--------------\n')
    f.write('--equation：{}\n'.format(args.equation))
    f.write('--seed：{}\n'.format(args.seed))
    f.write('--N_f：{}\n'.format(args.N_f))
    f.write('--time_max：{}\n'.format(args.time_max))
    f.write('--num_x：{}\n'.format(args.num_x))
    f.write('--num_t：{}\n'.format(args.num_t))
    f.write('--------------【  Training Setting  】--------------\n')
    f.write('--optimization：{}\n'.format(args.optimization))
    f.write('--epoch_adam：{}\n'.format(args.epoch_adam))
    f.write('--scheduler：{}\n'.format(args.scheduler))
    f.write('----step_size(for StepLR)/patience(for ReduceLROnPlateau)：{}\n'.format(args.step_sizeORpatience))
    f.write('----gamma(for StepLR)/factor(for ReduceLROnPlateau)：{}\n'.format(args.gammaORfactor))
    f.write('--check_interval：{}\n'.format(args.check_interval))
    f.write('--verbose_interval：{}\n'.format(args.verbose_interval))
    f.write('--restart_interval：{}\n'.format(args.restart_interval))
    f.write('--PDM_interval：{}\n'.format(args.PDM_interval))
    f.write('--PDM_lr：{}\n'.format(args.PDM_lr))
    f.write('--PDM_notion：{}\n'.format(args.PDM_notion))
    f.write('--t_divisions：{}\n'.format(args.t_divisions))
    f.write('--x_divisions：{}\n'.format(args.x_divisions))
    f.write('--warmming：{}\n'.format(args.warmming))
    f.write('--------------【Load Model/New Model】--------------\n')
    f.write('--load model or not：{}\n'.format(args.load_model_path))
    if args.load_model_path is not None:
        f.write('--load trained model from：{}\n'.format(args.load_model_path))
    if args.old_state is not None:
        f.write('--load old state from：{}\n'.format(args.old_state))
    if args.find_old_points is not None:
        f.write('--load PDM state from：{}\n'.format(args.find_old_points))
    f.write('--------------【Model Hyperparameters】--------------\n')
    f.write('--use complex or not：{}\n'.format(args.usecomplex))
    f.write('--layers：{}\n'.format(args.layers))
    f.write('--activation：{}\n'.format(args.activation))
    f.write('--weight_loss：{}\n'.format(args.weight_loss))
    f.write('{}  实验设置完成\n'.format(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))))
    f.close()


def train(args):
    # CUDA support
    zero_point = time.time()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('cuda or not:', device)
    first_dic = f"./{args.Project_num}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)

    set_seed(args.seed)
    log_writer(args)
    points_initialization, points_init, points_boundary, points_collection, points_regularization, \
    points_observation, RealUV_t_x_metric, points_plot_observe, test_inputs, test_label, min_point, \
    max_point, x_axis, t_axis = generate_data(args)
    memory_allocated = torch.cuda.memory_allocated()
    print('--【before while】:')
    print(f"--当前GPU内存分配: {memory_allocated / 1073741824} G")
    memory = psutil.virtual_memory()
    print(f"--Total memory: {memory.total / (1024 ** 3):.2f} GB")
    print(f"--Used memory: {memory.used / (1024 ** 3):.2f} GB")
    print(f"--Memory usage: {memory.percent}%")
    model = PhysicsInformedNN(args, x_axis, t_axis, RealUV_t_x_metric, points_init,
                              points_collection, points_boundary, min_point, max_point, test_inputs, test_label)
    f = open(f'Experiment_record.txt', 'a')
    f.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    f.write(f'  【训练开始】\n')
    f.close()
    model.train()
    tail_point = time.time()
    all_elapsed_time = "{:.2f}".format((tail_point - zero_point) / 3600)
    f = open(f'Experiment_record.txt', 'a')
    print('------------------Success------------------')
    f.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    f.write(f'  【全部训练完成】，总用时：{all_elapsed_time} h，当前error：{model.current_error}\n')
    f.close()
    model.evaluation(test_inputs, test_label, model.LogIter)
    model.plot_history()
    model.generate_git()


if __name__ == '__main__':
    main()
