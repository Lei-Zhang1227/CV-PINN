"""Run PINNs for convection/reaction/reaction-diffusion with periodic boundary conditions."""
# -*-coding:utf-8 -*-
import argparse
import time

from data_process import generate_data
from torch.utils.tensorboard import SummaryWriter
from doubleRVPINN import *

################
# Arguments
################
import psutil
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'Ture'
torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='LEI ZHANG')

    parser.add_argument('--Project_num', type=str, default='test',
                        help='Save path,usually named like 1101-1')
    parser.add_argument('--equation', type=str, default='1Dse-03',
                        help="The system is designed for studying different scenarios. "
                             'In each option, 1D represents the dimensionality of the equation,'
                             'while 01 corresponds to GPE1 with the initial conditions.')
    
    parser.add_argument('--time_max', type=float, default=5.0, help='--')
    parser.add_argument('--num_x', type=int, default=400, help='--')
    parser.add_argument('--num_t', type=int, default=4000, help='--')
    parser.add_argument('--usecomplex', default=False, action='store_true', help='whether to use complex NN or not')
    parser.add_argument('--double', default=False, action='store_true', help='whether to use complex NN or not')

    parser.add_argument('--regularization', default=False, action='store_true',
                        help='whether to use regularization or not')
    parser.add_argument('--discret_step', type=float, default=0.0005, help='--')

    parser.add_argument('--initialization', default=False, action='store_true',
                        help='whether to use initialization or not')

    parser.add_argument('--observation', default=False, action='store_true',
                        help='whether to use observational_data or not')
    parser.add_argument('--x_proportion', type=float, default=0.6, help='--')
    parser.add_argument('--t_proportion', type=float, default=0.2, help='--')
    parser.add_argument('--t_range', type=float, default=0.3, help='--')
    parser.add_argument('--add_noise', default=False, action='store_true',
                        help='whether add noise to observational_data')
    parser.add_argument('--noise_std', type=float, default=0.02, help='--')

    parser.add_argument('--bc', default=False, action='store_true', help='whether to use bc or not')
    parser.add_argument('--segments', type=int, default=5, help='whether to use bc or not')
    parser.add_argument('--seed', type=int, default=1227, help='Random initialization.')

    parser.add_argument('--verbose_interval', type=int, default=50, help='Epoch to show loss')
    parser.add_argument('--check_interval', type=int, default=200,
                        help='Epoch to check error,it should be a multiple of verbose_interval.')
    parser.add_argument('--restart_interval', type=int, default=50000,
                        help='save and restart')

    parser.add_argument('--weight_loss', type=str, default='100,100,1,1,1',
                        help='Weight of loss: Init, PDE, BC, Label, Regularization')
    parser.add_argument('--layers', type=str, default='2,180, 140, 160, 140 ,1',
                        help='Dimensions/layers of the NN, minus the first layer.')
    parser.add_argument('--activation', default='gelu', help='Activation to use in the network.')
    parser.add_argument('--optimization', default='AL', )
    parser.add_argument('--N_f', type=int, default=80000, help='Number of collocation points to sample.')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau')
    parser.add_argument('--step_sizeORpatience', type=int, default=200)
    parser.add_argument('--gammaORfactor', type=float, default=0.98)

    parser.add_argument('--epoch_adam', default=30000, type=int, help='epoch numbers for adam optimizer')
    parser.add_argument('--epoch_data', default=None, type=int, help='epoch numbers for data_driven')
    parser.add_argument('--epoch_lbfgs', default=60000, type=int, help='epoch numbers for lbfgs optimizer')
    parser.add_argument('--init_epoch_adam', default=10000, type=int,
                        help='epoch numbers for adam optimizer in init process')
    parser.add_argument('--init_epoch_lbfgs', default=10000, type=int,
                        help='epoch numbers for lbfgs optimizer in init process')

    parser.add_argument('--load_model_path', default=None, type=str,
                        help='checkpoint path for pretraining')
    args = parser.parse_args()
    train(args)


def log_writer(args):
    print('write information')
    f = open(f'Experiment_record.txt', 'a')
    f.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    f.write('\n------------------实验记录--------------------\n')
    f.write('--time:{}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    f.write('--Project_num：{}\n'.format(args.Project_num))
    f.write('--equation：{}\n'.format(args.equation))
    f.write('--seed：{}\n'.format(args.seed))

    f.write('--use complex or not：{}\n'.format(args.usecomplex))
    f.write('--use double or not：{}\n'.format(args.double))
    f.write('--activation：{}\n'.format(args.activation))
    f.write('--layers：{}\n'.format(args.layers))
    f.write('--N_f：{}\n'.format(args.N_f))
    f.write('--weights of loss：{}\n'.format(args.weight_loss))

    f.write('--optimization：{}\n'.format(args.optimization))
    f.write('--epoch_adam：{}\n'.format(args.epoch_adam))
    if args.epoch_data:
        f.write('--先使用label数据训练，训练轮次：{}\n'.format(args.epoch_data))
    f.write('--epoch_lbfgs：{}\n'.format(args.epoch_lbfgs))
    f.write('--scheduler：{}\n'.format(args.scheduler))
    f.write('----step_size(for StepLR)/patience(for ReduceLROnPlateau)：{}\n'.format(args.step_sizeORpatience))
    f.write('----gamma(for StepLR)/factor(for ReduceLROnPlateau)：{}\n'.format(args.gammaORfactor))

    f.write('--check_interval：{}\n'.format(args.check_interval))
    f.write('--verbose_interval：{}\n'.format(args.verbose_interval))
    f.write('--restart_interval：{}\n'.format(args.restart_interval))

    f.write('--load model or not：{}\n'.format(args.load_model_path))
    if args.load_model_path is not None:
        f.write('--path：{}\n'.format(args.load_model_path))

    f.write('--initialization or not：{}\n'.format(args.initialization))
    if args.initialization:
        f.write('--init_epoch_adam：{}\n'.format(args.init_epoch_adam))
        f.write('--init_epoch_lbfgs：{}\n'.format(args.init_epoch_lbfgs))

    f.write('--regularization or not：{}\n'.format(args.regularization))
    if args.regularization:
        f.write('------discret_step：{}\n'.format(args.discret_step))

    f.write('--use measurement data or not：{}\n'.format(args.observation))
    if args.observation:
        f.write('------x_proportion：{}\n'.format(args.x_proportion))
        f.write('------t_proportion：{}\n'.format(args.t_proportion))
        f.write('------t_range：{}\n'.format(args.t_range))
        f.write('------add_noise：{}\n'.format(args.add_noise))
        f.write('------noise_std：{}\n'.format(args.noise_std))

    f.write('--use bc or not：{}\n'.format(args.bc))
    if args.bc:
        f.write('--segments：{}\n'.format(args.segments))
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
    writer = SummaryWriter('logs', flush_secs=60)

    points_initialization, points_init, points_boundary, points_collection, points_regularization, \
    points_observation, RealUV_t_x_metric, points_plot_observe, test_inputs, test_label, min_point, \
    max_point, x_axis, t_axis = generate_data(args)
    finished_epoch = 0
    finished_epoch_when_load = 0
    if_go_on = False
    memory_allocated = torch.cuda.memory_allocated()
    print('--【before while】:')
    print(f"--当前GPU内存分配: {memory_allocated / 1073741824} G")
    memory = psutil.virtual_memory()
    print(f"--Total memory: {memory.total / (1024 ** 3):.2f} GB")
    print(f"--Used memory: {memory.used / (1024 ** 3):.2f} GB")
    print(f"--Memory usage: {memory.percent}%")
    while args.epoch_adam > finished_epoch:
        model = PhysicsInformedNN(args, x_axis, t_axis, RealUV_t_x_metric,
                                  points_collection, points_init, points_boundary, points_initialization,
                                  points_regularization, points_observation, min_point, max_point, writer=writer,
                                  training_load=if_go_on)
        if not if_go_on:
            finished_epoch_when_load = model.return_training_epoch()
        model.load_LogIter = finished_epoch_when_load
        start_time = time.time()
        model.train()
        writer.close()
        end_time = time.time()
        elapsed_time_hours = (end_time - start_time) / 3600
        all_elapsed_time = (end_time - zero_point) / 3600
        finished_epoch = model.return_training_epoch() - finished_epoch_when_load
        f = open(f'Experiment_record.txt', 'a')
        f.write('{}'.format(str(time.strftime('%Y-%m-%d %H:%M:%S\n', time.localtime(time.time())))))
        f.write('--模型训练完成，已训练{}轮，当前训练周期用时:{}h，总用时：{}h--\n'.format(model.return_training_epoch(),
                                                                                       elapsed_time_hours,
                                                                                       all_elapsed_time))
        f.close()
        model.evaluation(test_inputs, test_label)
        model.save_checkpoint()
        if_go_on = True
        print('finished_epoch:', finished_epoch)
        memory_allocated = torch.cuda.memory_allocated()
        print('--【after training】')
        print(f"--当前GPU内存分配: {memory_allocated / 1073741824} G")
        memory = psutil.virtual_memory()
        print(f"--Total memory: {memory.total / (1024 ** 3):.2f} GB")
        print(f"--Used memory: {memory.used / (1024 ** 3):.2f} GB")
        print(f"--Memory usage: {memory.percent}%")
    writer.close()
    tail_point = time.time()
    all_elapsed_time = (tail_point - zero_point) / 3600
    f = open(f'Experiment_record.txt', 'a')
    print('------------------Success------------------')
    f.write('{}'.format(str(time.strftime('%Y-%m-%d %H:%M:%S\n', time.localtime(time.time())))))
    f.write('--总用时：{}h--\n'.format(all_elapsed_time))
    f.close()
    model.generate_git()


if __name__ == '__main__':
    main()
