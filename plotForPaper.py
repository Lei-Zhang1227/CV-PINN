
import os
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.autograd import Variable
from visualize import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as ticker
import torch
import math
import os
from matplotlib.ticker import AutoMinorLocator
import argparse
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def calculate_fig_size_2(Aspect_Ratio, word_width=190, word_margins=34, ):
    '''
    计算出图时图的大小
    :param Aspect_Ratio: 目标图的横纵比
    :param word_width: 文档宽度
    :param word_margins: 文档横向页边距
    :return:
    '''
    fig_width = (word_width - word_margins) / 25.4
    fig_lenth = fig_width / Aspect_Ratio
    return [fig_width, fig_lenth]


def calculate_error(referred_uv, outputs_uv):
    mae_uv = np.abs(referred_uv - outputs_uv)
    return mae_uv


def read_data(key, t_range, x_range):
    data = data_prepare(key)
    # read
    x = data['x_axis']
    t = data['t_axis']
    print('x.shape', x.shape)
    print('t.shape', t.shape)
    ref = data['ref'].reshape(len(t), len(x))[::4, ::2]
    C = data['C'].reshape(len(t), len(x))[::4, ::2]
    R = data['R'].reshape(len(t), len(x))[::4, ::2]
    x = x[::2, :]
    t = t[::4, :]

    # ERROR
    EC = np.abs(ref - C)
    ER = np.abs(ref - R)

    # norm
    A_max = ref.max()
    A_min = ref.min()
    C = (C - A_min) / (A_max - A_min)
    ref = (ref - A_min) / (A_max - A_min)
    R = (R - A_min) / (A_max - A_min)
    max_error = max(EC.max(), ER.max())

    # time slice
    if x_range:
        x_start = int(x_range[0] * (len(x) - 1))
        x_end = int(x_range[1] * (len(x) - 1))
        S_1_C = C[int(t_range * (len(t) - 1)), x_start:x_end]
        S_2_C = C[-1, x_start:x_end]
        S_1_R = R[int(t_range * (len(t) - 1)), x_start:x_end]
        S_2_R = R[-1, x_start:x_end]
        S_1_ref = ref[int(t_range * (len(t) - 1)), x_start:x_end]
        S_2_ref = ref[-1, x_start:x_end]
        slice_x = x[x_start:x_end]
    else:
        S_1_C = C[int(t_range * (len(t) - 1)), :]
        S_2_C = C[-1, :]
        S_1_R = R[int(t_range * (len(t) - 1)), :]
        S_2_R = R[-1, :]
        S_1_ref = ref[int(t_range * (len(t) - 1)), :]
        S_2_ref = ref[-1, :]
        slice_x = None

    data_new = {'C': C, 'R': R, 'REF': ref, 'S_1_C': S_1_C, 'S_2_C': S_2_C, 'S_1_R': S_1_R, 'S_2_R': S_2_R,
                'S_1_ref': S_1_ref, 'S_2_ref': S_2_ref, 'max_error': max_error, 'EC': EC, 'ER': ER, 'x': x, 't': t,
                'slice_x': slice_x}
    print(
        f'ref size: {ref.shape}, C size: {C.shape}, R size: {R.shape}, x size: {x.shape}, t size: {t.shape}, S_1_C size: {S_1_C.shape}')
    return data_new


def plot_A():
    '''
    :param name_2:
    :param data_1:
    :param name_1:
    :param data_2:
    :param path:
    :param label1:
    :param label2:
    :param save_svg:
    :return:
    '''

    plt.rcParams['font.family'] = 'DejaVu Serif'
    # Creating the figures
    Aspect_Ratio = 8 / 6.5  # 宽/长
    fig, ax = plt.subplots(
        figsize=(calculate_fig_size_2(Aspect_Ratio)[0] * 1.10, calculate_fig_size_2(Aspect_Ratio)[1] * 1.10))

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')  # 不显示x轴和y轴
    # ax.text(0.5, 0.33333 * 2, '(b)', ha='center', va='center', fontsize=5)
    # ax.text(0.5, 0.33333 * 3, '(c)', ha='center', va='center', fontsize=5)
    gs_error = gridspec.GridSpec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1])
    gs_error.update(top=1 - 0.055, bottom=0.055, left=0 + 0.05, right=0.154 * 2, wspace=0.08, hspace=0.3)
    gs_solution = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])
    gs_solution.update(top=1 - 0.055, bottom=0.055, left=0.154 * 2 + 0.025, right=0.154 * 5 + 0.025, wspace=0.08,
                       hspace=0.3)
    gs_slice_A = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    gs_slice_A.update(top=1 - 0.055, bottom=1 - 0.055 - 0.247, left=0.154 * 5 + 0.025 * 3, right=0.98, hspace=0.3)
    gs_slice_B = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    gs_slice_B.update(top=1 - 0.055 - 0.247 * 1.3, bottom=1 - 0.055 - 0.247 * 2.3, left=0.154 * 5 + 0.025 * 3,
                      right=0.98,
                      hspace=0.3)
    gs_slice_C = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    gs_slice_C.update(top=1 - 0.055 - 0.247 * 2.6, bottom=0.055, left=0.154 * 5 + 0.025 * 3, right=0.98,
                      hspace=0.3)
    # 设置全局字体
    plt.rcParams['font.family'] = 'DejaVu Serif'
    # 设置刻度标签大小
    plt.rcParams['xtick.labelsize'] = 5
    plt.rcParams['ytick.labelsize'] = 5
    # 设置标题大小
    plt.rcParams['axes.titlesize'] = 5
    plt.rcParams['axes.titlepad'] = 2
    # 设置刻度长度和宽度
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['xtick.major.size'] = 1
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['ytick.major.size'] = 1
    # 设置全图中刻度标签与坐标轴的距离
    plt.rcParams['xtick.major.pad'] = 1
    plt.rcParams['ytick.major.pad'] = 0.5
    # 设置轴标签大小
    plt.rcParams['axes.labelsize'] = 5
    # 设置轴标签距离
    plt.rcParams['axes.labelpad'] = 1
    # 设置边框粗细
    plt.rcParams['axes.spines.top'] = True  # 例如，全局隐藏顶部边框
    plt.rcParams['axes.spines.right'] = True  # 全局隐藏右侧边框
    plt.rcParams['axes.spines.left'] = True  # 显示左侧边框
    plt.rcParams['axes.spines.bottom'] = True  # 显示底部边框
    plt.rcParams['axes.linewidth'] = 0.5  # 设置全局轴线宽度

    print('-- 画布构建完成')
    # region equation A
    # 读取数据
    data = read_data(key='A', t_range=Config_A.A_trange, x_range=Config_A.A_xrange)
    x = np.squeeze(data['x'])
    t = np.squeeze(data['t'])
    print(f'x size: {x.shape}, t size: {t.shape}')
    print('-- model A 数据读取完成')
    # equation-A ERROR-C
    AEC = plt.subplot(gs_error[0, 0])
    h = AEC.imshow(data['EC'].T, interpolation='nearest',  # cmap='rainbow',
                   extent=[np.min(t), np.max(t), np.min(x), np.max(x)],
                   origin='lower', aspect='auto', vmin=0, vmax=data['max_error'])
    AEC.set_title('Error-Complex')
    AEC.set_xlabel('$t$')
    AEC.set_ylabel('$x$')
    AEC.xaxis.set_major_locator(MaxNLocator(5, prune=None))

    # equation-A ERROR-R
    AER = plt.subplot(gs_error[0, 1])
    h_aer = AER.imshow(data['ER'].T, interpolation='nearest',  # cmap='rainbow',
                       extent=[np.min(t), np.max(t), np.min(x), np.max(x)],
                       origin='lower', aspect='auto', vmin=0, vmax=data['max_error'])
    divider = make_axes_locatable(AER)
    cax = divider.append_axes("right", size="3%", pad=0.015)
    cbar = fig.colorbar(h_aer, cax=cax)
    cbar.set_ticks(np.linspace(0, data['max_error'], num=6))
    AER.set_title('Error-Real')
    AER.set_xlabel('$t$')
    AER.set_yticklabels([])
    AER.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    if data['max_error'] <= 1.0:
        cbar.ax.yaxis.set_major_formatter(CustomFormatter())
        cbar.ax.yaxis.get_offset_text().set_position((2.7, 1))  # 调整偏移文本的位置
        cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置

    # equation-A C
    AC = plt.subplot(gs_solution[0, 0])
    h3 = AC.imshow(data['C'].T, interpolation='nearest',  # cmap='rainbow',
                   extent=[np.min(t), np.max(t), np.min(x), np.max(x)],
                   origin='lower', aspect='auto', vmin=0, vmax=1)
    AC.set_xlabel('$t$')
    AC.set_title('Predicted Solution-Complex')
    AC.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    AC.set_yticklabels([])
    # equation-A REF
    AREF = plt.subplot(gs_solution[0, 1])
    h4 = AREF.imshow(data['REF'].T, interpolation='nearest',  # cmap='rainbow',
                     extent=[np.min(t), np.max(t), np.min(x), np.max(x)],
                     origin='lower', aspect='auto', vmin=0, vmax=1)
    AREF.set_xlabel('$t$')
    AREF.set_yticklabels([])
    AREF.set_title('Numerical Solution')
    AREF.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    # equation-A R
    AR = plt.subplot(gs_solution[0, 2])
    h_AR = AR.imshow(data['R'].T, interpolation='nearest',  # cmap='rainbow',
                     extent=[np.min(t), np.max(t), np.min(x), np.max(x)],
                     origin='lower', aspect='auto', vmin=0, vmax=1)
    AR.set_xlabel('$t$')
    AR.set_yticklabels([])
    AR.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    AR.set_title('Predicted Solution-Real')
    divider = make_axes_locatable(AR)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cbar2 = fig.colorbar(h_AR, cax=cax)
    cbar2.set_ticks(np.linspace(0, 1, num=6))
    cbar2.ax.tick_params(labelsize=5)

    # equation-A slice_A
    ASC1 = plt.subplot(gs_slice_A[0, 0])
    ASC1.plot(data['slice_x'], data['S_1_ref'], linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=0.6,
              label='Numerical Solution')
    ASC1.plot(data['slice_x'], data['S_1_C'], linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=0.6,
              label='Predicted Solution')
    ASC1.set_xticklabels([])
    ASC1.set_ylabel(r'$|\Psi(x,t)|$')
    ASC1.yaxis.set_ticks([0, 1])
    ASC1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ASC1.set_title(f'$t = {format(Config_A.A_trange * 5, ".1f")}$')
    plt.axis('equal')
    ASC1.set_xlim([-20 + Config_A.A_xrange[0] * 40, -20 + Config_A.A_xrange[1] * 40])
    ASC1.set_ylim([0, 1])
    ASC1.set_aspect('auto')

    ASR1 = plt.subplot(gs_slice_A[0, 1])
    ASR1.plot(data['slice_x'], data['S_1_ref'], linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=0.6,
              label='Numerical Solution')
    ASR1.plot(data['slice_x'], data['S_1_R'], linestyle='--', color=(254 / 255, 183 / 255, 5 / 255), linewidth=0.6,
              label='Predicted Solution')
    ASR1.set_xticklabels([])
    ASR1.set_yticklabels([])
    ASR1.set_title(f'$t = {format(Config_A.A_trange * 5, ".1f")}$')
    plt.axis('equal')
    ASR1.set_xlim([-20 + Config_A.A_xrange[0] * 40, -20 + Config_A.A_xrange[1] * 40])
    ASR1.set_ylim([0, 1])
    ASR1.set_aspect('auto')

    ASC2 = plt.subplot(gs_slice_A[1, 0])
    ASC2.plot(data['slice_x'], data['S_2_ref'], linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=0.6,
              label='Numerical Solution')
    ASC2.plot(data['slice_x'], data['S_2_C'], linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=0.6,
              label='Predicted Solution')
    ASC2.set_xlabel('$x$', fontsize=4)
    ASC2.tick_params(axis='x', labelsize=4)
    ASC2.set_ylabel(r'$|\Psi(x,t)|$')
    ASC2.yaxis.set_ticks([0, 1])
    ASC2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ASC2.set_title('$t = 5$')
    plt.axis('equal')
    ASC2.set_xlim([-20 + Config_A.A_xrange[0] * 40, -20 + Config_A.A_xrange[1] * 40])
    ASC2.set_ylim([0, 1])
    ASC2.set_aspect('auto')

    ASR2 = plt.subplot(gs_slice_A[1, 1])
    ASR2.plot(data['slice_x'], data['S_2_ref'], linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=0.6,
              label='Numerical Solution')
    ASR2.plot(data['slice_x'], data['S_2_R'], linestyle='--', color=(254 / 255, 183 / 255, 5 / 255), linewidth=0.6,
              label='Predicted Solution')
    ASR2.set_yticklabels([])
    ASR2.tick_params(axis='x', labelsize=4)
    ASR2.set_xlabel('$x$', fontsize=4)
    ASR2.set_title('$t = 5$')
    plt.axis('equal')
    ASR2.set_xlim([-20 + Config_A.A_xrange[0] * 40, -20 + Config_A.A_xrange[1] * 40])
    ASR2.set_ylim([0, 1])
    ASR2.set_aspect('auto')
    plt.tight_layout()
    # endregion
    # region equation B
    # 读取数据
    data = read_data(key='B', t_range=Config_A.B_trange, x_range=Config_A.B_xrange)
    x = np.squeeze(data['x'])
    t = np.squeeze(data['t'])
    print(f'x size: {x.shape}, t size: {t.shape}')
    print('-- model A 数据读取完成')

    # equation-B ERROR-C
    BEC = plt.subplot(gs_error[1, 0])
    h = BEC.imshow(data['EC'].T, interpolation='nearest',  # cmap='rainbow',
                   extent=[np.min(t), np.max(t), np.min(x), np.max(x)],
                   origin='lower', aspect='auto', vmin=0, vmax=data['max_error'])
    BEC.set_title('Error-Complex')
    BEC.set_xlabel('$t$')
    BEC.set_ylabel('$x$')
    BEC.xaxis.set_major_locator(MaxNLocator(5, prune=None))

    # equation-A ERROR-R
    BER = plt.subplot(gs_error[1, 1])
    h_aer = BER.imshow(data['ER'].T, interpolation='nearest',  # cmap='rainbow',
                       extent=[np.min(t), np.max(t), np.min(x), np.max(x)],
                       origin='lower', aspect='auto', vmin=0, vmax=data['max_error'])
    divider = make_axes_locatable(BER)
    cax = divider.append_axes("right", size="3%", pad=0.015)
    cbar = fig.colorbar(h_aer, cax=cax)
    cbar.set_ticks(np.linspace(0, data['max_error'], num=6))
    BER.set_title('Error-Real')
    BER.set_xlabel('$t$')
    BER.set_yticklabels([])
    BER.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    if data['max_error'] <= 1.0:
        cbar.ax.yaxis.set_major_formatter(CustomFormatter())
        cbar.ax.yaxis.get_offset_text().set_position((2.7, 1))  # 调整偏移文本的位置
        cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置

    # equation-B C
    BC = plt.subplot(gs_solution[1, 0])
    h3 = BC.imshow(data['C'].T, interpolation='nearest',  # cmap='rainbow',
                   extent=[np.min(t), np.max(t), np.min(x), np.max(x)],
                   origin='lower', aspect='auto', vmin=0, vmax=1)
    BC.set_xlabel('$t$')
    BC.set_title('Predicted Solution-Complex')
    BC.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    BC.set_yticklabels([])
    # equation-A REF
    BREF = plt.subplot(gs_solution[1, 1])
    h4 = BREF.imshow(data['REF'].T, interpolation='nearest',  # cmap='rainbow',
                     extent=[np.min(t), np.max(t), np.min(x), np.max(x)],
                     origin='lower', aspect='auto', vmin=0, vmax=1)
    BREF.set_xlabel('$t$')
    BREF.set_yticklabels([])
    BREF.set_title('Numerical Solution')
    BREF.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    # equation-A R
    BR = plt.subplot(gs_solution[1, 2])
    h_AR = BR.imshow(data['R'].T, interpolation='nearest',  # cmap='rainbow',
                     extent=[np.min(t), np.max(t), np.min(x), np.max(x)],
                     origin='lower', aspect='auto', vmin=0, vmax=1)
    BR.set_xlabel('$t$')
    BR.set_yticklabels([])
    BR.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    BR.set_title('Predicted Solution-Real')
    divider = make_axes_locatable(BR)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cbar2 = fig.colorbar(h_AR, cax=cax)
    cbar2.set_ticks(np.linspace(0, 1, num=6))
    cbar2.ax.tick_params(labelsize=5)

    # equation-B slice
    BSC1 = plt.subplot(gs_slice_B[0, 0])
    BSC1.plot(data['slice_x'], data['S_1_ref'], linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=0.6,
              label='Numerical Solution')
    BSC1.plot(data['slice_x'], data['S_1_C'], linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=0.6,
              label='Predicted Solution')
    BSC1.set_xticklabels([])
    BSC1.set_ylabel(r'$|\Psi(x,t)|$')
    BSC1.yaxis.set_ticks([0, 1])
    BSC1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    BSC1.set_title(f'$t = {format(Config_A.A_trange * 5, ".1f")}$')
    plt.axis('equal')
    BSC1.set_xlim([-20 + Config_A.A_xrange[0] * 40, -20 + Config_A.A_xrange[1] * 40])
    BSC1.set_ylim([0, 1])
    BSC1.set_aspect('auto')

    BSR1 = plt.subplot(gs_slice_B[0, 1])
    BSR1.plot(data['slice_x'], data['S_1_ref'], linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=0.6,
              label='Numerical Solution')
    BSR1.plot(data['slice_x'], data['S_1_R'], linestyle='--', color=(254 / 255, 183 / 255, 5 / 255), linewidth=0.6,
              label='Predicted Solution')
    BSR1.set_xticklabels([])
    BSR1.set_yticklabels([])
    BSR1.set_title(f'$t = {format(Config_A.A_trange * 5, ".1f")}$')
    plt.axis('equal')
    BSR1.set_xlim([-20 + Config_A.A_xrange[0] * 40, -20 + Config_A.A_xrange[1] * 40])
    BSR1.set_ylim([0, 1])
    BSR1.set_aspect('auto')

    BSC2 = plt.subplot(gs_slice_B[1, 0])
    BSC2.plot(data['slice_x'], data['S_2_ref'], linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=0.6,
              label='Numerical Solution')
    BSC2.plot(data['slice_x'], data['S_2_C'], linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=0.6,
              label='Predicted Solution')
    BSC2.set_xlabel('$x$')
    BSC2.tick_params(axis='x', labelsize=4)
    BSC2.set_ylabel(r'$|\Psi(x,t)|$')
    BSC2.yaxis.set_ticks([0, 1])
    BSC2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    BSC2.set_title('$t = 5$')
    plt.axis('equal')
    BSC2.set_xlim([-20 + Config_A.A_xrange[0] * 40, -20 + Config_A.A_xrange[1] * 40])
    BSC2.set_ylim([0, 1])
    BSC2.set_aspect('auto')

    BSR2 = plt.subplot(gs_slice_B[1, 1])
    BSR2.plot(data['slice_x'], data['S_2_ref'], linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=0.6,
              label='Numerical Solution')
    BSR2.plot(data['slice_x'], data['S_2_R'], linestyle='--', color=(254 / 255, 183 / 255, 5 / 255), linewidth=0.6,
              label='Predicted Solution')
    BSR2.set_yticklabels([])
    BSR2.set_xlabel('$x$')
    BSR2.tick_params(axis='x', labelsize=4)
    BSR2.set_title('$t = 5$')
    plt.axis('equal')
    BSR2.set_xlim([-20 + Config_A.A_xrange[0] * 40, -20 + Config_A.A_xrange[1] * 40])
    BSR2.set_ylim([0, 1])
    BSR2.set_aspect('auto')
    plt.tight_layout()
    # endregion
    # region equation C
    # 读取数据
    data = read_data(key='C', t_range=Config_A.C_trange, x_range=Config_A.C_xrange)
    x = np.squeeze(data['x'])
    t = np.squeeze(data['t'])
    print(f'x size: {x.shape}, t size: {t.shape}')
    print('-- model A 数据读取完成')

    # equation-B ERROR-C
    BEC = plt.subplot(gs_error[2, 0])
    h = BEC.imshow(data['EC'].T, interpolation='nearest',  # cmap='rainbow',
                   extent=[np.min(t), np.max(t), np.min(x), np.max(x)],
                   origin='lower', aspect='auto', vmin=0, vmax=data['max_error'])
    BEC.set_title('Error-Complex')
    BEC.set_xlabel('$t$')
    BEC.set_ylabel('$x$')
    BEC.xaxis.set_major_locator(MaxNLocator(5, prune=None))

    # equation-A ERROR-R
    BER = plt.subplot(gs_error[2, 1])
    h_aer = BER.imshow(data['ER'].T, interpolation='nearest',  # cmap='rainbow',
                       extent=[np.min(t), np.max(t), np.min(x), np.max(x)],
                       origin='lower', aspect='auto', vmin=0, vmax=data['max_error'])
    divider = make_axes_locatable(BER)
    cax = divider.append_axes("right", size="3%", pad=0.015)
    cbar = fig.colorbar(h_aer, cax=cax)
    cbar.set_ticks(np.linspace(0, data['max_error'], num=6))
    BER.set_title('Error-Real')
    BER.set_xlabel('$t$')
    BER.set_yticklabels([])
    BER.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    if data['max_error'] <= 1.0:
        cbar.ax.yaxis.set_major_formatter(CustomFormatter())
        cbar.ax.yaxis.get_offset_text().set_position((2.7, 1))  # 调整偏移文本的位置
        cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置

    # equation-B C
    BC = plt.subplot(gs_solution[2, 0])
    h3 = BC.imshow(data['C'].T, interpolation='nearest',  # cmap='rainbow',
                   extent=[np.min(t), np.max(t), np.min(x), np.max(x)],
                   origin='lower', aspect='auto', vmin=0, vmax=1)
    BC.set_xlabel('$t$')
    BC.set_title('Predicted Solution-Complex')
    BC.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    BC.set_yticklabels([])
    # equation-A REF
    BREF = plt.subplot(gs_solution[2, 1])
    h4 = BREF.imshow(data['REF'].T, interpolation='nearest',  # cmap='rainbow',
                     extent=[np.min(t), np.max(t), np.min(x), np.max(x)],
                     origin='lower', aspect='auto', vmin=0, vmax=1)
    BREF.set_xlabel('$t$')
    BREF.set_yticklabels([])
    BREF.set_title('Numerical Solution')
    BREF.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    # equation-A R
    BR = plt.subplot(gs_solution[2, 2])
    h_AR = BR.imshow(data['R'].T, interpolation='nearest',  # cmap='rainbow',
                     extent=[np.min(t), np.max(t), np.min(x), np.max(x)],
                     origin='lower', aspect='auto', vmin=0, vmax=1)
    BR.set_xlabel('$t$')
    BR.set_yticklabels([])
    BR.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    BR.set_title('Predicted Solution-Real')
    divider = make_axes_locatable(BR)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cbar2 = fig.colorbar(h_AR, cax=cax)
    cbar2.set_ticks(np.linspace(0, 1, num=6))
    cbar2.ax.tick_params(labelsize=5)

    # equation-B slice
    BSC1 = plt.subplot(gs_slice_C[0, 0])
    BSC1.plot(data['slice_x'], data['S_1_ref'], linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=0.6,
              label='Numerical Solution')
    BSC1.plot(data['slice_x'], data['S_1_C'], linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=0.6,
              label='Predicted Solution')
    BSC1.set_xticklabels([])
    BSC1.set_ylabel(r'$|\Psi(x,t)|$')
    BSC1.yaxis.set_ticks([0, 1])
    BSC1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    BSC1.set_title(f'$t = {format(Config_A.C_trange * 5, ".1f")}$')
    plt.axis('equal')
    BSC1.set_xlim([-20 + Config_A.A_xrange[0] * 40, -20 + Config_A.A_xrange[1] * 40])
    BSC1.set_ylim([0, 1])
    BSC1.set_aspect('auto')

    BSR1 = plt.subplot(gs_slice_C[0, 1])
    BSR1.plot(data['slice_x'], data['S_1_ref'], linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=0.6,
              label='Numerical Solution')
    BSR1.plot(data['slice_x'], data['S_1_R'], linestyle='--', color=(254 / 255, 183 / 255, 5 / 255), linewidth=0.6,
              label='Predicted Solution')
    BSR1.set_xticklabels([])
    BSR1.set_yticklabels([])
    BSR1.set_title(f'$t = {format(Config_A.C_trange * 5, ".1f")}$')
    plt.axis('equal')
    BSR1.set_xlim([-20 + Config_A.A_xrange[0] * 40, -20 + Config_A.A_xrange[1] * 40])
    BSR1.set_ylim([0, 1])
    BSR1.set_aspect('auto')

    BSC2 = plt.subplot(gs_slice_C[1, 0])
    BSC2.plot(data['slice_x'], data['S_2_ref'], linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=0.6,
              label='Numerical Solution')
    line3, = BSC2.plot(data['slice_x'], data['S_2_C'], linestyle='--', color=(255 / 255, 106 / 255, 125 / 255),
                       linewidth=0.6,
                       label='Predicted Solution-C')
    BSC2.set_xlabel('$x$')
    BSC2.tick_params(axis='x', labelsize=4)
    BSC2.set_ylabel(r'$|\Psi(x,t)|$')
    BSC2.yaxis.set_ticks([0, 1])
    BSC2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    BSC2.set_title('$t = 5$')
    plt.axis('equal')
    BSC2.set_xlim([-20 + Config_A.A_xrange[0] * 40, -20 + Config_A.A_xrange[1] * 40])
    BSC2.set_ylim([0, 1])
    BSC2.set_aspect('auto')

    BSR2 = plt.subplot(gs_slice_C[1, 1])
    line1, = BSR2.plot(data['slice_x'], data['S_2_ref'], linestyle='-', color=(45 / 255, 12 / 255, 126 / 255),
                       linewidth=0.6,
                       label='Numerical Solution')
    line2, = BSR2.plot(data['slice_x'], data['S_2_R'], linestyle='--', color=(254 / 255, 183 / 255, 5 / 255),
                       linewidth=0.6,
                       label='Predicted Solution-R')
    BSR2.set_yticklabels([])
    BSR2.set_xlabel('$x$')
    BSR2.tick_params(axis='x', labelsize=4)
    BSR2.set_title('$t = 5$')
    plt.axis('equal')
    BSR2.set_xlim([-20 + Config_A.A_xrange[0] * 40, -20 + Config_A.A_xrange[1] * 40])
    BSR2.set_ylim([0, 1])
    BSR2.set_aspect('auto')

    fig.legend(handles=[line1, line3, line2], loc='upper center', bbox_to_anchor=(0.75, 0.35), ncol=3, fontsize=5,
               frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')  # 不显示x轴和y轴
    ax.text(0.5, 0.66, '(a)', ha='center', va='center', fontsize=6)
    ax.text(0.5, 0.33, '(b)', ha='center', va='center', fontsize=6)
    ax.text(0.5, 0.02, '(c)', ha='center', va='center', fontsize=6)

    plt.tight_layout()

    # endregion

    # plt.show()
    plt.savefig(f'{Config_A.path}/5.1-2.png', dpi=300)


def ewma(data, alpha=0.3):
    return pd.Series(data).ewm(alpha=alpha).mean().to_numpy()


def plot_B():
    '''
    :param name_2:
    :param data_1:
    :param name_1:
    :param data_2:
    :param path:
    :param label1:
    :param label2:
    :param save_svg:
    :return:
    '''

    plt.rcParams['font.family'] = 'DejaVu Serif'
    # Creating the figures
    Aspect_Ratio = 2.2 / 1  # 宽/长
    fig, ax = plt.subplots(
        figsize=(calculate_fig_size_2(Aspect_Ratio)[0] * 1, calculate_fig_size_2(Aspect_Ratio)[1] * 1))

    # ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')  # 不显示x轴和y轴
    gs_error = gridspec.GridSpec(2, 7, width_ratios=[2, 0.15, 1, 1, 1, 1, 1.1], height_ratios=[1, 1])
    gs_error.update(top=1 - 0.055, bottom=0.1, left=0 + 0.05, right=1 - 0.05, wspace=0.08, hspace=0.3)
    # region 基础设置
    # 设置全局字体
    plt.rcParams['font.family'] = 'DejaVu Serif'
    # 设置刻度标签大小
    plt.rcParams['xtick.labelsize'] = 5
    plt.rcParams['ytick.labelsize'] = 5
    # 设置标题大小
    plt.rcParams['axes.titlesize'] = 5
    plt.rcParams['axes.titlepad'] = 2
    # 设置刻度长度和宽度
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['xtick.major.size'] = 1
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['ytick.major.size'] = 1
    # 设置全图中刻度标签与坐标轴的距离
    plt.rcParams['xtick.major.pad'] = 1
    plt.rcParams['ytick.major.pad'] = 0.5
    # 设置轴标签大小
    plt.rcParams['axes.labelsize'] = 5
    # 设置轴标签距离
    plt.rcParams['axes.labelpad'] = 0
    # 设置边框粗细
    plt.rcParams['axes.spines.top'] = True  # 例如，全局隐藏顶部边框
    plt.rcParams['axes.spines.right'] = True  # 全局隐藏右侧边框
    plt.rcParams['axes.spines.left'] = True  # 显示左侧边框
    plt.rcParams['axes.spines.bottom'] = True  # 显示底部边框
    plt.rcParams['axes.linewidth'] = 0.5  # 设置全局轴线宽度
    # 设置全局刻度线向内
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # endregion
    # region plot loss
    loss_list, log_list, x_axis = read_history('R')
    smoothed_loss_list = ewma(loss_list.reshape(-1), Config_B.alpha)
    print('loss_list.shape,log_list.shape,x_axis.shape:', loss_list.shape, log_list.shape, x_axis.shape)
    loss = fig.add_subplot(gs_error[0])
    # axins_first = loss.inset_axes([Config_B.fig1_x, Config_B.fig1_y, Config_B.fig1_width, Config_B.fig1_lenth])
    # axins_first.set_facecolor((1, 1, 1, 0.8))

    line_loss_R, = loss.semilogy(x_axis, smoothed_loss_list, linestyle='-',
                                 color=(181 / 255, 11 / 255, 11 / 255, 1),
                                 linewidth=0.4, label='PINN')
    # axins_first.plot(x_axis, smoothed_loss_list, linestyle='-.', color=(81 / 255, 11 / 255, 11 / 255, 0.8), alpha=0.8,
    #                  linewidth=0.5, label='PINN')
    loss_list, log_list, x_axis = read_history('C')
    smoothed_loss_list = ewma(loss_list.reshape(-1), Config_B.alpha)
    line_loss_C, = loss.semilogy(x_axis, smoothed_loss_list, linestyle='-',
                                 color=(45 / 255, 12 / 255, 126 / 255, 1),
                                 linewidth=0.4, label='CV-PINN')
    # axins_first.plot(x_axis, smoothed_loss_list, linestyle='-.', color=(45 / 255, 12 / 255, 126 / 255, 0.8), alpha=0.8,
    #                  linewidth=0.5, label='CV-PINN')
    loss.set_xlabel('Epochs($\\times 10^{3}$)')  # 设置x轴刻度标签
    loss.set_ylabel('Traing MAE')  # 设置y轴刻度标签
    loss.set_ylim(0, Config_B.max_mae)
    loss.set_xlim(0, Config_B.max_x)
    loss.tick_params(axis='x', )
    loss.tick_params(axis='y', )
    # 设置x轴和y轴的刻度密度
    # loss.locator_params(axis='x')  # 设置x轴的刻度线数量
    # loss.locator_params(axis='y')
    # offset_text = loss.xaxis.get_offset_text()
    # offset_text.set_position((0, 0))
    # if Config_B.max_mae < 1.0:
    #     loss.yaxis.set_major_formatter(CustomFormatter())  # 假设CustomFormatter是一个自定义格式化器
    #     loss.yaxis.get_offset_text().set_fontsize(6)
    #     loss.yaxis.get_offset_text().set_position((-0.03, 0))  # 调整偏移文本的位置
    #     loss.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置
    # # 绘制次刻度线
    # loss.xaxis.set_minor_locator(AutoMinorLocator(2))  # 在每个主刻度之间添加1个次刻度
    # loss.yaxis.set_minor_locator(AutoMinorLocator(2))  # 在每个主刻度之间添加1个次刻度
    # loss.tick_params(axis='x', direction='in', which='minor')
    # loss.tick_params(axis='y', direction='in', which='minor')
    # # 显示次刻度线
    # loss.minorticks_on()
    loss.legend(bbox_to_anchor=(0.95, 0.95), fontsize=5, frameon=False, markerscale=1)

    # axins_first.set_xlim(Config_B.fig1_x_range[0], Config_B.fig1_x_range[1])  # 设置放大区域的x轴范围
    # axins_first.set_ylim(0, round_down_auto(smoothed_loss_list[Config_B.fig1_x_range[1] * 20] * Config_B.R_1_H))  # 设置放大区域的y轴范围
    # axins_first.locator_params(axis='x', nbins=3)  # 设置x轴的刻度线数量
    # axins_first.locator_params(axis='y', nbins=4)
    # axins_first.tick_params(axis='x', labelsize=4, width=0.4, direction='in', length=1,
    #                         pad=1.5)
    # axins_first.tick_params(axis='y', labelsize=4, width=0.4, direction='in', length=1,
    #                         pad=1.5)
    # offset_text = axins_first.xaxis.get_offset_text()
    # offset_text.set_fontsize(6)
    # offset_text.set_position((0, 0))
    # #     axins_first.set_xlabel('Epochs($\\times 10^{3}$)', fontdict=fontdict, labelpad=1.5)  # 设置x轴刻度标签
    # #     axins_first.set_ylabel('MAE', fontdict=fontdict, labelpad=1.5)  # 设置y轴刻度标签
    # if round_down_auto(smoothed_loss_list[Config_B.fig1_x_range[1] * 20] * Config_B.R_1_H) < 1.0:
    #     axins_first.yaxis.set_major_formatter(CustomFormatter())  # 假设CustomFormatter是一个自定义格式化器
    #     axins_first.yaxis.get_offset_text().set_fontsize(3)
    #     axins_first.yaxis.get_offset_text().set_position((-0.03, 0))  # 调整偏移文本的位置
    #     axins_first.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置
    #
    # # axins_first.yaxis.get_offset_text().set_color = (157 / 255, 157 / 255, 161 / 255, 0.8)
    # axins_first.spines['top'].set_linewidth(0.5)  # 设置上边框的宽度
    # axins_first.spines['right'].set_linewidth(0.5)  # 设置右边框的宽度
    # axins_first.spines['bottom'].set_linewidth(0.5)  # 设置下边框的宽度
    # axins_first.spines['left'].set_linewidth(0.5)
    # axins_first.grid(True, color=(178 / 255, 178 / 255, 178 / 255, 0.6), linestyle='--', linewidth=0.25)
    #
    # rect = Rectangle((Config_B.fig1_x_range[0], 0), Config_B.fig1_x_range[1] - Config_B.fig1_x_range[0],
    #                  min(round_down_auto(smoothed_loss_list[Config_B.fig1_x_range[1] * 20] * Config_B.R_1_H),Config_B.y_max),
    #                  edgecolor='red',
    #                  facecolor=(222 / 255, 223 / 255, 217 / 255, 0.4), lw=0.5, zorder=3)
    # loss.add_patch(rect)

    # 添加连接线
    # ax.annotate("", xy=(Config_B.fig1_x + Config_B.fig1_width / 2 - 0.05, Config_B.fig1_y + 0.03),
    #             xycoords='figure fraction',
    #             xytext=((Config_B.fig1_x_range[0] + Config_B.fig1_x_range[1]) / 2,
    #                     min(round_down_auto(smoothed_loss_list[Config_B.fig1_x_range[1] * 20] * Config_B.R_1_H), Config_B.y_max)),
    #             textcoords='data', arrowprops=dict(arrowstyle='-|>', color='red', linewidth=0.4,
    #                                                linestyle='dashed'))
    # endregion
    # region plot log
    loss_list, log_list, x_axis = read_history('R')
    log = fig.add_subplot(gs_error[7])
    line_log_R, = log.plot(log_list[:, 0], log_list[:, 1], linestyle='-',
                           color=(181 / 255, 11 / 255, 11 / 255, 1),
                           linewidth=0.4, label='PINN')
    loss_list, log_list, x_axis = read_history('C')
    line_log_C, = log.plot(log_list[:, 0], log_list[:, 1], linestyle='-',
                           color=(45 / 255, 12 / 255, 126 / 255, 1),
                           linewidth=0.4, label='CV-PINN')
    log.set_xlabel('Epochs($\\times 10^{3}$)')  # 设置x轴刻度标签
    log.set_ylabel('Test MSE')  # 设置y轴刻度标签
    log.set_ylim(0, Config_B.max_mse)
    log.set_xlim(0, Config_B.max_x)
    log.tick_params(axis='x', direction='in')
    log.tick_params(axis='y', direction='in')
    # 设置x轴和y轴的刻度密度
    log.locator_params(axis='x', nbins=5)  # 设置x轴的刻度线数量
    log.locator_params(axis='y', nbins=5)
    offset_text = loss.xaxis.get_offset_text()
    offset_text.set_position((0, 0))
    if Config_B.max_mae < 1.0:
        log.yaxis.set_major_formatter(CustomFormatter())  # 假设CustomFormatter是一个自定义格式化器
        log.yaxis.get_offset_text().set_fontsize(6)
        log.yaxis.get_offset_text().set_position((-0.03, 0))  # 调整偏移文本的位置
        log.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置
    # 绘制次刻度线
    log.xaxis.set_minor_locator(AutoMinorLocator(2))  # 在每个主刻度之间添加1个次刻度
    log.yaxis.set_minor_locator(AutoMinorLocator(2))  # 在每个主刻度之间添加1个次刻度
    log.tick_params(axis='x', direction='in', which='minor')
    log.tick_params(axis='y', direction='in', which='minor')
    # 显示次刻度线
    log.minorticks_on()
    log.legend(bbox_to_anchor=(0.95, 0.95), fontsize=5, frameon=False, markerscale=1)
    # endregion
    inputs, ref, x_axis, t_axis = generate_input_B(Config_B.equation)
    ref = ref.reshape(len(t_axis), len(x_axis))
    A_max = ref.max()
    A_min = ref.min()
    ref_s = (ref - A_min) / (A_max - A_min)
    # region plot ref
    Fig_ref = fig.add_subplot(gs_error[6])
    h = Fig_ref.imshow(ref_s.T, interpolation='nearest',  # cmap='rainbow',
                       extent=[np.min(t_axis), np.max(t_axis), np.min(x_axis), np.max(x_axis)],
                       origin='lower', aspect='auto', vmin=0, vmax=1)
    divider = make_axes_locatable(Fig_ref)
    cax = divider.append_axes("right", size="3%", pad=0.015)
    cbar = fig.colorbar(h, cax=cax)
    cbar.set_ticks(np.linspace(0, 1, num=6))
    Fig_ref.set_title('Numerical Solution')
    Fig_ref.set_xlabel('$t$')
    Fig_ref.set_yticklabels([])
    Fig_ref.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    # endregion
    # region C-CHECK-1
    psi_2 = read_solutions('C', Config_B.check_point[0], inputs, ref, t_axis, x_axis)
    Fig_2 = fig.add_subplot(gs_error[2])
    h = Fig_2.imshow(psi_2.T, interpolation='nearest',  # cmap='rainbow',
                     extent=[np.min(t_axis), np.max(t_axis), np.min(x_axis), np.max(x_axis)],
                     origin='lower', aspect='auto', vmin=0, vmax=1)
    Fig_2.set_title(f'Complex_{Config_B.check_point[0]}')
    Fig_2.set_xlabel('$t$')
    Fig_2.set_ylabel('$x$', fontsize=4)
    Fig_2.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    # endregion
    # region C-CHECK-2
    psi_2 = read_solutions('C', Config_B.check_point[1], inputs, ref, t_axis, x_axis)
    Fig_2 = fig.add_subplot(gs_error[3])
    h4 = Fig_2.imshow(psi_2.T, interpolation='nearest',  # cmap='rainbow',
                      extent=[np.min(t_axis), np.max(t_axis), np.min(x_axis), np.max(x_axis)],
                      origin='lower', aspect='auto', vmin=0, vmax=1)
    Fig_2.set_xlabel('$t$')
    Fig_2.set_yticklabels([])
    Fig_2.set_title(f'Complex_{Config_B.check_point[1]}')
    Fig_2.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    # endregion
    # region C-CHECK-3
    psi_2 = read_solutions('C', Config_B.check_point[2], inputs, ref, t_axis, x_axis)
    Fig_2 = fig.add_subplot(gs_error[4])
    h4 = Fig_2.imshow(psi_2.T, interpolation='nearest',  # cmap='rainbow',
                      extent=[np.min(t_axis), np.max(t_axis), np.min(x_axis), np.max(x_axis)],
                      origin='lower', aspect='auto', vmin=0, vmax=1)
    Fig_2.set_xlabel('$t$')
    Fig_2.set_yticklabels([])
    Fig_2.set_title(f'Complex_{Config_B.check_point[2]}')
    Fig_2.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    # endregion
    # region C-CHECK-4
    psi_2 = read_solutions('C', Config_B.check_point[3], inputs, ref, t_axis, x_axis)
    Fig_2 = fig.add_subplot(gs_error[5])
    h4 = Fig_2.imshow(psi_2.T, interpolation='nearest',  # cmap='rainbow',
                      extent=[np.min(t_axis), np.max(t_axis), np.min(x_axis), np.max(x_axis)],
                      origin='lower', aspect='auto', vmin=0, vmax=1)
    Fig_2.set_xlabel('$t$')
    Fig_2.set_yticklabels([])
    Fig_2.set_title(f'Complex_{Config_B.check_point[3]}')
    Fig_2.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    # endregion
    # region plot ref
    Fig_ref = fig.add_subplot(gs_error[13])
    psi_2 = read_solutions('R', Config_B.check_point[-1], inputs, ref, t_axis, x_axis)
    h = Fig_ref.imshow(psi_2.T, interpolation='nearest',  # cmap='rainbow',
                       extent=[np.min(t_axis), np.max(t_axis), np.min(x_axis), np.max(x_axis)],
                       origin='lower', aspect='auto', vmin=0, vmax=1)
    divider = make_axes_locatable(Fig_ref)
    cax = divider.append_axes("right", size="3%", pad=0.015)
    cbar = fig.colorbar(h, cax=cax)
    cbar.set_ticks(np.linspace(0, 1, num=6))
    Fig_ref.set_title(f'Real_{Config_B.check_point[-1]}')
    Fig_ref.set_xlabel('$t$')
    Fig_ref.set_yticklabels([])
    Fig_ref.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    # endregion
    # region C-CHECK-1
    psi_2 = read_solutions('R', Config_B.check_point[0], inputs, ref, t_axis, x_axis)
    Fig_2 = fig.add_subplot(gs_error[9])
    h = Fig_2.imshow(psi_2.T, interpolation='nearest',  # cmap='rainbow',
                     extent=[np.min(t_axis), np.max(t_axis), np.min(x_axis), np.max(x_axis)],
                     origin='lower', aspect='auto', vmin=0, vmax=1)
    Fig_2.set_title(f'Real_{Config_B.check_point[0]}')
    Fig_2.set_xlabel('$t$')
    Fig_2.set_ylabel('$x$', fontsize=4)
    Fig_2.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    # endregion
    # region C-CHECK-2
    psi_2 = read_solutions('R', Config_B.check_point[1], inputs, ref, t_axis, x_axis)
    Fig_2 = fig.add_subplot(gs_error[10])
    h4 = Fig_2.imshow(psi_2.T, interpolation='nearest',  # cmap='rainbow',
                      extent=[np.min(t_axis), np.max(t_axis), np.min(x_axis), np.max(x_axis)],
                      origin='lower', aspect='auto', vmin=0, vmax=1)
    Fig_2.set_xlabel('$t$')
    Fig_2.set_yticklabels([])
    Fig_2.set_title(f'Real_{Config_B.check_point[1]}')
    Fig_2.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    # endregion
    # region C-CHECK-3
    psi_2 = read_solutions('R', Config_B.check_point[2], inputs, ref, t_axis, x_axis)
    Fig_2 = fig.add_subplot(gs_error[11])
    h4 = Fig_2.imshow(psi_2.T, interpolation='nearest',  # cmap='rainbow',
                      extent=[np.min(t_axis), np.max(t_axis), np.min(x_axis), np.max(x_axis)],
                      origin='lower', aspect='auto', vmin=0, vmax=1)
    Fig_2.set_xlabel('$t$')
    Fig_2.set_yticklabels([])
    Fig_2.set_title(f'Real_{Config_B.check_point[2]}')
    Fig_2.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    # endregion
    # region C-CHECK-4
    psi_2 = read_solutions('R', Config_B.check_point[3], inputs, ref, t_axis, x_axis)
    Fig_2 = fig.add_subplot(gs_error[12])
    h4 = Fig_2.imshow(psi_2.T, interpolation='nearest',  # cmap='rainbow',
                      extent=[np.min(t_axis), np.max(t_axis), np.min(x_axis), np.max(x_axis)],
                      origin='lower', aspect='auto', vmin=0, vmax=1)
    Fig_2.set_xlabel('$t$')
    Fig_2.set_yticklabels([])
    Fig_2.set_title(f'Real_{Config_B.check_point[3]}')
    Fig_2.xaxis.set_major_locator(MaxNLocator(5, prune=None))
    # endregion

    plt.savefig(f'{Config_B.path}/5.2-1.png', dpi=300)


def round_down_auto(number):
    if number == 0:
        return 0
    decimals = -int(math.floor(math.log10(abs(number))))
    factor = 10 ** decimals
    return math.floor(number * factor) / factor


def generate_input(key):
    equation = getattr(Config_A, f'{key}_E', None)
    data = pd.read_csv('./data/{}.csv'.format(equation))
    # ############################ 数据准备 #############################
    x_axis = np.array(sorted(list(set(list(data['x']))), reverse=False)).reshape(-1, 1)
    t_axis = np.array(sorted(list(set(list(data['t']))), reverse=False)).reshape(-1, 1)
    #  reference
    test_inputs = np.hstack((np.array(data['x']).reshape(-1, 1), np.array(data['t']).reshape(-1, 1)))
    inputs = torch.tensor(test_inputs, requires_grad=True).float().to(device)
    test_label = np.array(data['uv']).reshape(-1, 1)
    return inputs, test_label, x_axis, t_axis


def generate_input_B(key):
    print(key)
    data = pd.read_csv(f'./data/{key}.csv')
    # ############################ 数据准备 #############################
    x_axis = np.array(sorted(list(set(list(data['x']))), reverse=False)).reshape(-1, 1)
    t_axis = np.array(sorted(list(set(list(data['t']))), reverse=False)).reshape(-1, 1)
    #  reference
    test_inputs = np.hstack((np.array(data['x']).reshape(-1, 1), np.array(data['t']).reshape(-1, 1)))
    inputs = torch.tensor(test_inputs, requires_grad=True).float().to(device)
    test_label = np.array(data['uv']).reshape(-1, 1)
    return inputs, test_label, x_axis, t_axis


def data_prepare(key):
    """
    Load a model, define the optimizer, load optimizer state, load LogIter, and load loss_list.
    :param filename: The path to the directory containing the model and checkpoint files.
    :param key: The key used to retrieve epoch information from the Config.
    :param x: The input tensor x.
    :param t: The input tensor t.
    :return: The calculated psi values as a numpy array.
    """
    inputs, ref, x_axis, t_axis = generate_input(key)

    filename_C = getattr(Config_A, f'{key}_C', None)
    C_model = torch.load(f'{filename_C}/BestModel.pkl', map_location=torch.device('cpu'))
    C_model = C_model.to(device)
    epoch_value = getattr(Config_A, f'{key}_epoch', None)
    if epoch_value is not None:
        checkpoint_path = f'{filename_C}/checkpoint_{epoch_value}.pth.tar'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state = checkpoint['model']
        C_model.load_state_dict(state)
        C_model = C_model.to(device)
    out = C_model(inputs)
    u = out[:, 0:1]
    v = out[:, -1:]
    u = u.detach().cpu().numpy()
    v = v.detach().cpu().numpy()
    psi_C = np.sqrt(np.square(u) + np.square(v))

    filename_R = getattr(Config_A, f'{key}_R', None)
    R_model = torch.load(f'{filename_R}/BestModel.pkl', map_location=torch.device('cpu'))
    R_model = R_model.to(device)
    epoch_value = getattr(Config_A, f'{key}_epoch', None)
    if epoch_value is not None:
        checkpoint_path = f'{filename_R}/checkpoint_{epoch_value}.pth.tar'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state = checkpoint['model']
        R_model.load_state_dict(state)
        R_model = R_model.to(device)

    out = R_model(inputs)
    u = out[:, 0:1]
    v = out[:, -1:]
    u = u.detach().cpu().numpy()
    v = v.detach().cpu().numpy()
    psi_R = np.sqrt(np.square(u) + np.square(v))
    result = {'ref': ref, 'C': psi_C, 'R': psi_R, 'x_axis': x_axis, 't_axis': t_axis}
    return result


def fill_save_log(log, epoch):
    log = np.array(log)
    target_size = (int(epoch / 2500) + 1, 2)
    epochs = np.arange(0, target_size[0] * 2500 + 2500, 2500)
    filled_sequence = np.zeros(target_size)
    filled_sequence[:, 0] = epochs[:target_size[0]]
    for i in range(1, target_size[0]):
        if epochs[i] in log[:, 0]:
            filled_sequence[i, 1] = log[log[:, 0] == epochs[i], 1]
            filled_sequence[i, 0] = filled_sequence[i, 0] / 1000
        else:
            filled_sequence[i, 1] = filled_sequence[i - 1, 1]
            filled_sequence[i, 0] = filled_sequence[i, 0] / 1000
    filled_sequence[0, 1] = 0.1
    print('filled_sequence.shape:', filled_sequence.shape)
    return filled_sequence


def read_history(key):
    path = getattr(Config_B, f'{key}_path', None)
    model = torch.load(f'{path}/BestModel.pkl', map_location=torch.device('cpu'))
    checkpoint = torch.load(f'{path}/checkpoint_{Config_B.check_point[-1]}.pth.tar', map_location=torch.device('cpu'))
    loss_list = np.array(checkpoint['loss_list'])
    iterations_all = len(loss_list)
    x_axis = np.arange(1, iterations_all + 1, 1) * 50 / 1000
    if 'save_record' in checkpoint:
        save_record = np.array(checkpoint['save_record'])
        log_list = fill_save_log(save_record, iterations_all * 50)
    else:
        log_list = []
    all_loss = loss_list[:, 1:2] + loss_list[:, 2:3] + loss_list[:, 3:4]
    return all_loss, log_list, x_axis


def read_solutions(key, epoch, inputs, ref, t_axis, x_axis):
    path = getattr(Config_B, f'{key}_path', None)
    model = torch.load(f'{path}/BestModel.pkl', map_location=torch.device('cpu'))
    checkpoint = torch.load(f'{path}/checkpoint_{Config_B.check_point[-1]}.pth.tar', map_location=torch.device('cpu'))
    model = model.to(device)
    checkpoint_path = f'{path}/checkpoint_{epoch}.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state = checkpoint['model']
    model.load_state_dict(state)
    C_model = model.to(device)

    out = C_model(inputs)
    u = out[:, 0:1]
    v = out[:, -1:]
    u = u.detach().cpu().numpy()
    v = v.detach().cpu().numpy()
    psi = np.sqrt(np.square(u) + np.square(v)).reshape(len(t_axis), len(x_axis))
    A_max = ref.max()
    A_min = ref.min()
    psi = (psi - A_min) / (A_max - A_min)
    return psi


class Config_A():
    A_C = '/code/output/PDM/210/C-C'  # 第一个case C模型的项目地址
    A_R = '/code/output/PDM/210/R-C'  # 第一个case R模型的项目地址
    A_E = '1D-GPE-210'  # 第一个case的方程
    A_epoch = 70000
    A_trange = 0.75
    A_xrange = [0.25, 0.75]

    B_C = '/code/output/PDM/EXP15.C-A'
    B_R = '/code/output/PDM/EXP15.R-A'
    B_E = '1D-GPE-012'
    B_epoch = 50000
    B_trange = 0.75
    B_xrange = [0.25, 0.75]

    C_C = '/code/output/1.3/1.3.6.C'
    C_R = '/code/output/1.3/1.3.6.R_cut'
    C_E = '1D-GPE-13'
    C_epoch = None
    C_trange = 0.5
    C_xrange = [0.25, 0.75]

    path = '/code/output/PaperFig'


class Config_B():
    C_path = '/code/output/PDM/TEST110-C-7'
    R_path = '/code/output/PDM/TEST110-R-7'
    check_point = [10000,20000, 30000, 40000, 50000]
    equation = '1D-GPE-110'
    max_mae = 0.1
    max_mse = 0.05
    max_x = 50
    path = '/code/output/PaperFig/012'
    alpha = 0.4
    # loss曲线子图
    fig1_x = 0.45
    fig1_y = 0.25
    fig1_width = 0.35
    fig1_lenth = 0.4
    fig1_x_range = [30, 45]
    y_max = 0.03
    R_1_H = 20


path = Config_A.path
if not os.path.exists(path):
    os.makedirs(path)
path = Config_B.path
if not os.path.exists(path):
    os.makedirs(path)
plot_B()
