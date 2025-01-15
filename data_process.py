import pandas as pd
from pyDOE import lhs
from visualize import *
import random
from numpy import exp
from numpy import sin,cos


def generate_t_regularization(r=1.132355, n=50):
    """
    生成一个等比数列，实现后面紧密前面稀疏,设定的时间长度是5
    :param r: 比率
    :param n: 数列长度
    :return:
    """
    t_regularization = []
    for i in range(0, n):
        t_regularization.append(5 - 0.01 * r ** i)
    return t_regularization


# t_regularization = generate_t_regularization()
# print(t_regularization)


def generate_regularization_points(discret_step):
    """
    按照一定的积分步长离散x轴，用于计算x轴上的积分
    :param discret_step: Discretization Step Size,越小越好
    :return:
    """
    points_regularization = {}
    x_regularization = np.arange(-20, 20, discret_step).reshape(-1, 1)
    print('x_regularization.shape:', len(x_regularization))
    N_regulation_x = x_regularization.shape[0]
    t_regularization = generate_t_regularization()
    print('t_regularization.shape:', len(t_regularization))
    X_t_x_regularization, T_t_x_regularization = np.meshgrid(x_regularization, t_regularization)
    # print(X_t_x_regularization)
    # print(T_t_x_regularization)
    X_regularization = X_t_x_regularization.reshape(-1, 1)
    # print(x_regularization)
    T_regularization = T_t_x_regularization.reshape(-1, 1)
    # print(T_regularization)
    # points_regularization = np.hstack((X_regularization, t_regularization))

    points_regularization['x'] = X_regularization
    points_regularization['t'] = T_regularization
    points_regularization['x_axis'] = x_regularization
    points_regularization['t_axis'] = t_regularization
    points_regularization['help'] = '计算数值积分时，将uv值reshape为(len(t_axis),len(x_axis))'
    points_regularization['discret_step'] = discret_step
    return points_regularization


# points_regularization = generate_regularization_points(0.5)
# print(points_regularization['x_r'])
# print(points_regularization['t_r'][:, 1:2])


def generate_observation_data(x_axis, t_axis, RealU_t_x_metric, RealV_t_x_metric, x_proportion, t_proportion, t_range):
    """
    添加噪声等GPT恢复以后再议。。。

    用于生成指定范围内的观测数据
    :param x_axis: 求解域的x轴
    :param t_axis: 求解域的y轴
    :param RealU_t_x_metric: 实部的观测值矩阵[t_r,x_r]
    :param RealV_t_x_metric: 虚部的观测值矩阵[t_r,x_r]
    :param x_proportion: 在x轴上取点的比例[0,1]
    :param t_proportion: 在t轴上取点的比例[0,1]
    :param t_range:在t轴上去点的额范围[0,1]
    :return:
    X_item  选取点的x轴坐标，用于后期出图
    T_item  选取点的t轴左边，用于后期出图
    points_observation  选取点，col分别为：[x_r,t_r,u,v]

    example:
    选择 x_proportion=0.7，t_proportion=0.3，t_range=0.4
    这意味着所选观测点在x轴上占据了70%，在t轴上占据了30%，特别的，t轴上的30%只会在前40%的时间域内出现；
    """
    selected_x_amount = int(len(x_axis) * x_proportion)
    selected_x_item = [random.randint(0, len(x_axis) - 1) for _ in range(selected_x_amount)]
    selected_x = x_axis[selected_x_item]
    selected_t_amount = int(len(t_axis) * t_proportion)
    selected_t_item = [random.randint(0, int(len(t_axis) * t_range) - 1) for _ in range(selected_t_amount)]
    selected_t = t_axis[selected_t_item]
    selected_matrix = RealU_t_x_metric[selected_t_item][:, selected_x_item]

    selected_u = (RealU_t_x_metric[selected_t_item][:, selected_x_item]).reshape(-1, 1)
    selected_v = (RealV_t_x_metric[selected_t_item][:, selected_x_item]).reshape(-1, 1)
    # print(selected_v.reshape(len(selected_t_item), len(selected_x_item)))

    X_item, T_item, = np.meshgrid(selected_x_item, selected_t_item, )
    X_item = X_item.reshape(-1, 1)
    T_item = T_item.reshape(-1, 1)
    # print(T_item.reshape(len(selected_t_item), len(selected_x_item)))
    points_plot_observe = np.hstack((X_item, T_item))

    X_observation, T_observation, = np.meshgrid(selected_x, selected_t, )
    x_observation = X_observation.reshape(-1, 1)
    t_observation = T_observation.reshape(-1, 1)

    points_observation = np.hstack((x_observation, t_observation, selected_u, selected_v))
    # print('x_observation:', X_observation)
    # print('t_observation:', T_observation)
    # print('x_observation:', x_observation)
    # print('t_observation:', t_observation)
    # print('selected_x_amount:', selected_x_amount)
    # print('selected_x_item:', selected_x_item)
    # print('selected_x:', selected_x)
    # print('selected_t_amount:', selected_t_amount)
    # print('selected_x_item:', selected_x_item)
    # print('selected_t:', selected_t)
    # print('selected_matrix:', selected_matrix)
    # print('points_observation:', points_observation)
    return points_plot_observe, points_observation


def generate_data(args, ):
    data = pd.read_csv('/code/data/{}.csv'.format(args.equation))
    N_f = args.N_f
    # ############################ 数据准备 #############################
    x_axis = np.array(sorted(list(set(list(data['x']))), reverse=False)).reshape(-1, 1)
    t_axis = np.array(sorted(list(set(list(data['t']))), reverse=False)).reshape(-1, 1)
    # --生成x轴，t轴--
    min_point = np.array([-20.0, np.min(t_axis)])
    max_point = np.array([20.0, np.max(t_axis)])
    # --生成用于归一化的最大最小值--

    #  reference
    RealUV_t_x_metric = np.array(data['uv']).reshape(len(t_axis), len(x_axis))
    RealU_t_x_metric = np.array(data['u']).reshape(len(t_axis), len(x_axis))
    RealV_t_x_metric = np.array(data['v']).reshape(len(t_axis), len(x_axis))

    test_inputs = np.hstack((np.array(data['x']).reshape(-1, 1), np.array(data['t']).reshape(-1, 1)))
    test_label = np.hstack(
        (np.array(data['u']).reshape(-1, 1), np.array(data['v']).reshape(-1, 1), np.array(data['uv']).reshape(-1, 1)))

    # 方程初始条件
    init_num = int(args.equation[7])
    print('init_num:',init_num)
    x_init = np.arange(-20, 20, 0.05).reshape(-1, 1)
    t_init = (x_init * 0).reshape(-1, 1)
    if init_num == 0:
        u_init = np.array(exp(-(x_init ** 2 / 10))).reshape(-1, 1)
        v_init = (u_init * 0).reshape(-1, 1)
    elif init_num == 1:
        u_init = np.array(sin(x_init) * 2.0 / (exp(x_init) + exp(-x_init))).reshape(-1, 1)
        v_init = (u_init * 0).reshape(-1, 1)
    elif init_num == 2:
        u_init = np.array(cos(x_init) * 2.0 / (exp(x_init) + exp(-x_init))).reshape(-1, 1)
        v_init = (u_init * 0).reshape(-1, 1)
    else:
        raise ValueError("缺少初始条件。")
    sumuv = sum(u_init ** 2 * 0.05 + v_init ** 2 * 0.05) ** 0.5
    u_init = u_init / sumuv  # 归一化
    points_init = np.hstack((x_init, t_init, u_init, v_init, u_init))

    # 边界条件
    t_boundary = np.arange(0, t_axis[-1], 0.0001).reshape(-1, 1)
    x_left_boundary = np.tile(x_axis[0], (len(t_boundary), 1))
    x_right_boundary = np.tile(x_axis[-1], (len(t_boundary), 1))
    u_left_boundary = np.tile(0, (len(t_boundary), 1))
    points_left_boundary = np.hstack((x_left_boundary, t_boundary, u_left_boundary, u_left_boundary))
    points_right_boundary = np.hstack((x_right_boundary, t_boundary, u_left_boundary, u_left_boundary))
    points_boundary = np.vstack((points_left_boundary, points_right_boundary))

    # 归一化条件
    if args.regularization:
        points_regularization = generate_regularization_points(args.discret_step)
    else:
        points_regularization = None

    # PDE配点

    points_collection = min_point + (max_point - min_point) * lhs(2, N_f)

    # 网络初始化数据
    if args.initialization:
        u_star = np.tile(u_init, (len(t_axis), 1))
        v_star = np.tile(v_init, (len(t_axis), 1))
        X, T = np.meshgrid(x_init, t_axis)
        x_star = X.flatten()[:, None]
        t_star = T.flatten()[:, None]
        points_initialization = np.hstack((x_star, t_star, u_star, v_star))
    else:
        points_initialization = None

    if args.observation:
        points_plot_observe, points_observation = generate_observation_data(x_axis, t_axis, RealU_t_x_metric,
                                                                            RealV_t_x_metric, args.x_proportion,
                                                                            args.t_proportion, args.t_range)
    else:
        points_plot_observe = None
        points_observation = None

    return points_initialization, points_init, points_boundary, points_collection, points_regularization, \
           points_observation, RealUV_t_x_metric, points_plot_observe, test_inputs, test_label, min_point, max_point, x_axis, t_axis

def generate_data2(args, ):
    data = pd.read_csv('/code/data/{}.csv'.format(args.equation))
    N_f = args.N_f
    # ############################ 数据准备 #############################
    x_axis = np.array(sorted(list(set(list(data['x']))), reverse=False)).reshape(-1, 1)
    t_axis = np.array(sorted(list(set(list(data['t']))), reverse=False)).reshape(-1, 1)
    # --生成x轴，t轴--
    min_point = np.array([-8.0, np.min(t_axis)])
    max_point = np.array([8.0, np.max(t_axis)])
    # --生成用于归一化的最大最小值--

    #  reference
    RealUV_t_x_metric = np.array(data['uv']).reshape(len(t_axis), len(x_axis))
    RealU_t_x_metric = np.array(data['u']).reshape(len(t_axis), len(x_axis))
    RealV_t_x_metric = np.array(data['v']).reshape(len(t_axis), len(x_axis))

    test_inputs = np.hstack((np.array(data['x']).reshape(-1, 1), np.array(data['t']).reshape(-1, 1)))
    test_label = np.hstack(
        (np.array(data['u']).reshape(-1, 1), np.array(data['v']).reshape(-1, 1), np.array(data['uv']).reshape(-1, 1)))

    # 方程初始条件
    init_num = int(args.equation[7])
    print('init_num:',init_num)
    x_init = np.arange(-20, 20, 0.05).reshape(-1, 1)
    t_init = (x_init * 0).reshape(-1, 1)
    if init_num == 0:
        u_init = np.array(exp(-(x_init ** 2 / 10))).reshape(-1, 1)
        v_init = (u_init * 0).reshape(-1, 1)
    elif init_num == 1:
        u_init = np.array(sin(x_init) * 2.0 / (exp(x_init) + exp(-x_init))).reshape(-1, 1)
        v_init = (u_init * 0).reshape(-1, 1)
    elif init_num == 2:
        u_init = np.array(cos(x_init) * 2.0 / (exp(x_init) + exp(-x_init))).reshape(-1, 1)
        v_init = (u_init * 0).reshape(-1, 1)
    else:
        raise ValueError("缺少初始条件。")
    sumuv = sum(u_init ** 2 * 0.05 + v_init ** 2 * 0.05) ** 0.5
    u_init = u_init / sumuv  # 归一化
    points_init = np.hstack((x_init, t_init, u_init, v_init, u_init))

    # 边界条件
    t_boundary = np.arange(0, t_axis[-1], 0.0001).reshape(-1, 1)
    x_left_boundary = np.tile(x_axis[0], (len(t_boundary), 1))
    x_right_boundary = np.tile(x_axis[-1], (len(t_boundary), 1))
    u_left_boundary = np.tile(0, (len(t_boundary), 1))
    points_left_boundary = np.hstack((x_left_boundary, t_boundary, u_left_boundary, u_left_boundary))
    points_right_boundary = np.hstack((x_right_boundary, t_boundary, u_left_boundary, u_left_boundary))
    points_boundary = np.vstack((points_left_boundary, points_right_boundary))

    # 归一化条件
    if args.regularization:
        points_regularization = generate_regularization_points(args.discret_step)
    else:
        points_regularization = None

    # PDE配点
    points_collection = min_point + (max_point - min_point) * lhs(2, N_f)

    # 网络初始化数据
    if args.initialization:
        u_star = np.tile(u_init, (len(t_axis), 1))
        v_star = np.tile(v_init, (len(t_axis), 1))
        X, T = np.meshgrid(x_init, t_axis)
        x_star = X.flatten()[:, None]
        t_star = T.flatten()[:, None]
        points_initialization = np.hstack((x_star, t_star, u_star, v_star))
    else:
        points_initialization = None

    if args.observation:
        points_plot_observe, points_observation = generate_observation_data(x_axis, t_axis, RealU_t_x_metric,
                                                                            RealV_t_x_metric, args.x_proportion,
                                                                            args.t_proportion, args.t_range)
    else:
        points_plot_observe = None
        points_observation = None

    return points_initialization, points_init, points_boundary, points_collection, points_regularization, \
           points_observation, RealUV_t_x_metric, points_plot_observe, test_inputs, test_label, min_point, max_point, x_axis, t_axis


class DataProcess():
    """
    后面再慢慢整理
    """

    def __init__(self, args):
        self.data = pd.read_csv('./data/{}.csv'.format(args.equation))
        self.x_axis = np.array(sorted(list(set(list(self.data['x_r']))), reverse=False)).reshape(-1, 1)
        self.t_axis = np.array(sorted(list(set(list(self.data['t_r']))), reverse=False)).reshape(-1, 1)
        self.RealUV_t_x_metric = np.array(self.data['uv']).reshape(len(self.t_axis), len(self.x_axis))
        self.RealU_t_x_metric = np.array(self.data['u']).reshape(len(self.t_axis), len(self.x_axis))
        self.RealV_t_x_metric = np.array(self.data['v']).reshape(len(self.t_axis), len(self.x_axis))

    def generate_reference(self, ):
        pass

    def generate_init(self):
        pass

    def generate_boundary(self):
        pass

    def generate_regularization(self):
        pass

    def generate_initialization(self):
        pass

    def generate_collection(self):
        pass

    def generate_observation(self):
        pass

    def generate_minmax(self):
        pass


class DateProcess_B():
    def __init__(self, args):
        pass

    def generate_Anchor(self):
        pass

    def generate_Scoring(self):
        pass

# x_axis = np.array([-1, 0, 1, 2, 3])
# t_axis = np.array([0, 1, 2, 3])
# RealV_t_x_metric = np.array(
#     [[1, 2, 3, 4, 5], [11, 22, 33, 44, 55], [111, 222, 333, 444, 555], [1111, 2222, 3333, 4444, 5555]])
# RealU_t_x_metric = np.array(
#     [[1, 2, 3, 4, 5], [11, 22, 33, 44, 55], [111, 222, 333, 444, 555], [1111, 2222, 3333, 4444, 5555]])
# x_proportion = 0.80
# t_proportion = 0.5
# t_range = 0.75
# points_plot_observe, points_observation = generate_observation_data(x_axis, t_axis, RealU_t_x_metric, RealV_t_x_metric,
#                                                                     x_proportion, t_proportion, t_range)
# points_regularization = generate_regularization_points(0.5)
# x_r = points_regularization['x_r']
# print(x_r)
# t_r = points_regularization['t_r']
# print(t_r)
# x_r = x_r.reshape((len(points_regularization['t_axis']), len(points_regularization['x_axis'])))
# print(x_r)
