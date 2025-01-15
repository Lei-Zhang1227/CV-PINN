"""
Visualize outputs.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotting import newfig, savefig
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.font_manager as font_manager
import math


def round_down_auto(number):
    if number == 0:
        return 0
    decimals = -int(math.floor(math.log10(abs(number))))
    factor = 10 ** decimals
    return math.floor(number * factor) / factor


def calculate_aspect(x, t, aspe):
    '''
    计算确定的横纵比以使得imshow得到的图像与scatter大小一致
    :param x:
    :param t:
    :param aspe:
    :return:
    '''
    x_num = len(x)
    t_num = len(t)
    aspect = 1 * x_num / (aspe * t_num)
    return aspect


def calculate_fig_size(Aspect_Ratio, word_width=210, word_margins=25.4, ):
    '''
    计算出图时图的大小
    :param Aspect_Ratio: 目标图的横纵比
    :param word_width: 文档宽度
    :param word_margins: 文档横向页边距
    :return:
    '''
    fig_width = (word_width - 2 * word_margins) / 25.4
    fig_lenth = fig_width / Aspect_Ratio
    return [fig_width, fig_lenth]


class CustomFormatter(ticker.ScalarFormatter):
    '''
    设置cbar的科学计数法表示
    '''

    def __init__(self, useMathText=True, powerlimits=(-1, 1)):
        super().__init__(useMathText=useMathText)
        self.set_powerlimits(powerlimits)
        self.set_scientific(True)

    def __call__(self, x, pos=None):
        # 缩放数值以适应科学计数法的基数部分
        scale = np.power(10, -self.orderOfMagnitude)
        return f'{x * scale:.1f}'
    
    
def plot_solution_with_new_points3(path, x, t, diff, label2,save_svg=False):
    '''
    单独绘制分数热图
    :param data1:
    :param x:
    :param t:
    :param path:
    :param title:
    :param label2:
    :param save_svg:
    :return:
    '''

    Aspect_Ratio = 1 / 1.3
    plt.rcParams['font.family'] = 'DejaVu Serif'
    #  ################# 图标题 ##################
    # fig.suptitle(f'{title}{label2}', fontname='Times New Roman', fontweight='bold', size=15)
    fig = plt.figure(figsize=(calculate_fig_size(Aspect_Ratio)[0] / 3, calculate_fig_size(Aspect_Ratio)[1] / 3))
    ax3 = fig.add_subplot(111)
    # region error
    u_max = diff.max()
    u_min = diff.min()
    diff = (diff - u_min) / (u_max - u_min)
    h = ax3.imshow(diff, interpolation='nearest', cmap='viridis',
                   extent=[t.min(), t.max(), x.min(), x.max()],
                   origin='lower', aspect='auto', vmin=0, vmax=1)
    # ax3.set_title('(c) New Points', fontsize=10)
    ax3.set_xlabel('$t$', fontweight=400, size=6, labelpad=-5)
    ax3.set_xlim([0, 5])
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax3.set_ylim([-20, 20])
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax3.set_ylabel('$x$', fontweight=400, size=6, labelpad=-5)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax3.tick_params(axis='x', labelsize=5, direction='in', length=1)
    ax3.tick_params(axis='y', labelsize=5, direction='in', length=1)
    ax3.grid(True, which='major', color=(178 / 255, 178 / 255, 178 / 255, 0.6), linestyle='--', linewidth=0.25)
    # endregion
    if save_svg:
        plt.savefig(f'{path}/{label2}.svg', format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.savefig(f'{path}/{label2}.png', dpi=500, bbox_inches='tight', transparent=False)
    plt.close()
    
    

def plot_solution_with_new_points(data1, x, t, path, label2, x_f, t_f, pre_train, diff, save_svg=False):
    '''
    绘制全局的残差分布，与solution绘制不同的是，由于数值普遍较小，在cbar刻度上采用科学计数法的表示，保留了两位小数；
    :param data1:
    :param x:
    :param t:
    :param path:
    :param title:
    :param label2:
    :param save_svg:
    :return:
    '''
    Aspect_Ratio = 1 / 1.3
    plt.rcParams['font.family'] = 'DejaVu Serif'
    #  ################# 图标题 ##################
    # fig.suptitle(f'{title}{label2}', fontname='Times New Roman', fontweight='bold', size=15)
    fig, ax = plt.subplots(figsize=(calculate_fig_size(Aspect_Ratio)[0], calculate_fig_size(Aspect_Ratio)[1]))

    u_max = data1.max()
    u_min = data1.min()
    data1 = (data1 - u_min) / (u_max - u_min)
    pre_train = (pre_train - u_min) / (u_max - u_min)

    gs0 = gridspec.GridSpec(2, 2)
    gs0.update(top=1, bottom=0, left=0, right=1, wspace=0.2)
    #  region 数值解
    ax1 = plt.subplot(gs0[0, 0])
    h = ax1.imshow(data1.T, interpolation='nearest', cmap='viridis',
                   extent=[t.min(), t.max(), x.min(), x.max()],
                   origin='lower', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('(a) Numerical Solution', fontsize=10)
    #  ################# 坐标轴 ##################
    ax1.set_xlabel(r'$t$', fontweight=400, size=10, labelpad=-5)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax1.set_ylabel(r'$x$', fontweight=400, size=10, labelpad=-5)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.tick_params(labelsize=10)
    ax1.grid(True, which='major', color=(178 / 255, 178 / 255, 178 / 255, 0.6), linestyle='--', linewidth=0.25)
    # endregion
    #  region 预测解
    ax2 = plt.subplot(gs0[0, 1])
    h = ax2.imshow(pre_train.T, interpolation='nearest', cmap='viridis',
                   extent=[t.min(), t.max(), x.min(), x.max()],
                   origin='lower', aspect='auto', vmin=0, vmax=1)
    # h = ax2.scatter(t_f, x_f, c=pre_init, s=4, alpha=0.8, marker='o', cmap='viridis', edgecolors='none')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.set_ticks(np.linspace(0, 1, num=11))
    cbar.ax.tick_params(labelsize=10)
    #  ################# 坐标轴 ##################
    ax2.set_xlabel(r'$t$', fontweight=400, size=10, labelpad=-5)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax2.set_xlim([0, 5])
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax2.set_ylim([-20, 20])
    ax2.set_yticklabels([])
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax2.set_title('(b) Current Prediction', fontsize=10)
    ax2.tick_params(labelsize=10)
    ax2.grid(True, which='major', color=(178 / 255, 178 / 255, 178 / 255, 0.6), linestyle='--', linewidth=0.25)
    # endregion
    # region error
    ax3 = plt.subplot(gs0[1, 0])
    # h2 = ax3.imshow(pre_train.T, interpolation='nearest',  # cmap='rainbow',
    #                 extent=[t.min(), t.max(), x.min(), x.max()],
    #                 origin='lower', aspect='auto', vmin=0, vmax=1.0)
    h2 = ax3.scatter(t_f, x_f, c=(45 / 255, 12 / 255, 126 / 255, 0.5), s=1, marker='o', edgecolors='none')
    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = fig.colorbar(h2, cax=cax)
    # cbar.set_ticks(np.linspace(0, 0.3, num=9))
    ax3.set_title('(c) New Points', fontsize=10)
    ax3.set_xlabel('$t$', fontweight=400, size=10, labelpad=-5)
    ax3.set_xlim([0, 5])
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax3.set_ylim([-20, 20])
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax3.set_ylabel('$x$', fontweight=400, size=10, labelpad=-5)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    cbar.ax.yaxis.set_major_formatter(CustomFormatter())
    cbar.ax.yaxis.get_offset_text().set_position((2.7, 1))  # 调整偏移文本的位置
    cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置
    ax3.grid(True, which='major', color=(178 / 255, 178 / 255, 178 / 255, 0.6), linestyle='--', linewidth=0.25)
    # endregion

    # region weight
    # 绘制散点图
    ax4 = fig.add_subplot(gs0[1, 1], aspect=1 / 1.3)
    h3 = ax4.imshow(diff.T, interpolation='nearest',  # cmap='rainbow',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto', vmin=0, vmax=1.0)
    # h2 = ax3.scatter(t_f, x_f, c=pre_train, s=4, alpha=0.8, marker='o', cmap='viridis', edgecolors='none')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h3, cax=cax)
    cbar.set_ticks(np.linspace(0, 1, num=11))
    # scatter = ax4.scatter(t_f, x_f, c=diff, s=4, alpha=0.8, marker='o', cmap='viridis', edgecolors='none')
    #
    # divider = make_axes_locatable(ax4)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = fig.colorbar(scatter, cax=cax)
    # cbar.set_ticks(np.linspace(0, round_down_auto(max(diff)), num=11))
    # cbar.ax.tick_params(labelsize=10)
    #
    # cbar.ax.yaxis.set_major_formatter(CustomFormatter())
    # cbar.ax.yaxis.get_offset_text().set_position((2.7, 1))  # 调整偏移文本的位置
    # cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置
    # 设置标签、标题和限制
    ax4.set_xlabel(r'$t$', fontweight=400, size=10, labelpad=-5)
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax4.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax4.set_title('(d) Last Prediction', fontsize=10)
    ax4.set_xlim([0, 5])
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax4.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax4.set_ylim([-20, 20])
    ax4.set_yticklabels([])
    ax4.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax4.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax4.set_aspect('auto')
    ax4.spines['top'].set_visible(True)
    ax4.spines['right'].set_visible(True)
    ax4.grid(True, which='major', color=(178 / 255, 178 / 255, 178 / 255, 0.6), linestyle='--', linewidth=0.25)

    # ax3.set_title('(c) Absolute Error', fontsize=10)
    # ax3.set_xlabel('$t$', fontweight=400, size=10, labelpad=-5)
    # ax3.set_xlim([0, 5])
    # ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    # ax3.set_ylim([-20, 20])
    # ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    # ax3.set_ylabel('$x$', fontweight=400, size=10, labelpad=-5)
    # ax3.yaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax3.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    # cbar.ax.yaxis.set_major_formatter(CustomFormatter())
    # cbar.ax.yaxis.get_offset_text().set_position((2.7, 1))  # 调整偏移文本的位置
    # cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置
    # ax3.grid(True, which='major', color=(178 / 255, 178 / 255, 178 / 255, 0.6), linestyle='--', linewidth=0.25)

    # endregion
    if save_svg:
        plt.savefig(f'{path}/{label2}.svg', format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.savefig(f'{path}/{label2}.png', dpi=500, bbox_inches='tight', transparent=False)
    plt.close()


def plot_solution_with_new_points2(path, label2, x_f, t_f, save_svg=False):
    '''
    绘制全局的残差分布，与solution绘制不同的是，由于数值普遍较小，在cbar刻度上采用科学计数法的表示，保留了两位小数；
    :param data1:
    :param x:
    :param t:
    :param path:
    :param title:
    :param label2:
    :param save_svg:
    :return:
    '''

    Aspect_Ratio = 1 / 1.3
    plt.rcParams['font.family'] = 'DejaVu Serif'
    #  ################# 图标题 ##################
    # fig.suptitle(f'{title}{label2}', fontname='Times New Roman', fontweight='bold', size=15)
    fig = plt.figure(figsize=(calculate_fig_size(Aspect_Ratio)[0]/3, calculate_fig_size(Aspect_Ratio)[1]/3))
    ax3 = fig.add_subplot(111)
    # region error
    h2 = ax3.scatter(t_f, x_f, c=(45 / 255, 12 / 255, 126 / 255, 0.5), s=0.3, marker='o', edgecolors='none')
    # ax3.set_title('(c) New Points', fontsize=10)
    ax3.set_xlabel('$t$', fontweight=400, size=6, labelpad=-5)
    ax3.set_xlim([0, 5])
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax3.set_ylim([-20, 20])
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax3.set_ylabel('$x$', fontweight=400, size=6, labelpad=-5)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax3.tick_params(axis='x', labelsize=5, direction='in', length=1)
    ax3.tick_params(axis='y', labelsize=5, direction='in', length=1)
    ax3.grid(True, which='major', color=(178 / 255, 178 / 255, 178 / 255, 0.6), linestyle='--', linewidth=0.25)
    # endregion
#     if save_svg:
#         plt.savefig(f'{path}/{label2}.svg', format='svg', bbox_inches='tight', transparent=False)
#     else:
#         plt.savefig(f'{path}/{label2}.png', dpi=500, bbox_inches='tight', transparent=False)
#     plt.savefig(f'{path}/{label2}.svg', format='svg', bbox_inches='tight', transparent=False)
    plt.savefig(f'{path}/{label2}.png', dpi=500, bbox_inches='tight', transparent=False)
    plt.close()


def plot_single_residual(data, x, t, path, title, save_title, label2, save_svg=True, Aspect_Ratio=1.5,
                         show_title=False):
    '''
    绘制全局的残差分布，与solution绘制不同的是，由于数值普遍较小，在cbar刻度上采用科学计数法的表示，保留了两位小数；
    :param data:
    :param x:
    :param t:
    :param path:
    :param title:
    :param label2:
    :param save_svg:
    :return:
    '''
    #     font_path = r'/code/font/times.ttf'
    #     font_prop = font_manager.FontProperties(fname=font_path)
    #     plt.rcParams['font.family'] = font_prop.get_name()
    #     print('font_prop.get_name():',font_prop.get_name())
    #     font_paths = font_manager.findSystemFonts()
    #     print(font_paths)

    fig = plt.figure(figsize=(calculate_fig_size(Aspect_Ratio)[0] / 2, calculate_fig_size(Aspect_Ratio)[1]))
    plt.rcParams['font.family'] = 'DejaVu Serif'
    max_value = min(data.max(), 0.03)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max_value)
    ax = fig.add_subplot(111)
    data = data.reshape(len(t), len(x))
    #  0313修改：对于大于1的情况，不使用科学计数法的颜色条表示；
    data_max = data.max()
    #  0313修改--zhanglei
    h = ax.imshow(data.T, interpolation='nearest', cmap='viridis',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto', norm=norm)
    divider = make_axes_locatable(ax)
    #  ################# 设置颜色条 ##################
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.set_ticks(np.linspace(0, max_value, num=9))
    cbar.ax.tick_params(labelsize=8)
    if data_max <= 1.0:
        cbar.ax.yaxis.set_major_formatter(CustomFormatter())
        cbar.ax.yaxis.get_offset_text().set_position((2.7, 1))  # 调整偏移文本的位置
        cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置
    #  ################# 坐标轴 ##################
    ax.set_xlabel('t', fontweight=400, size=10, labelpad=-5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.set_ylabel('x', fontweight=400, size=10, labelpad=-5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    #  ################# 图标题 ##################
    if show_title:
        ax.set_title(f'{title}{label2}', fontweight='bold', size=12, pad=20)
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.9, -0.05),
            ncol=5,
            frameon=False,
            prop={'size': 10}
        )
    ax.tick_params(labelsize=8)
    #  ################# 保存 ##################
    if save_svg:
        plt.savefig(f'{path}{save_title}_{label2}.svg', format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.savefig(f'{path}{save_title}_{label2}.png', dpi=500, bbox_inches='tight', transparent=True)
    plt.close()
    return None


def plot_double_residual(data1, data2, x, t, path, save_title, label2, save_svg=True, Aspect_Ratio=1.7):
    '''
    绘制全局的残差分布，与solution绘制不同的是，由于数值普遍较小，在cbar刻度上采用科学计数法的表示，保留了两位小数；
    :param data1:
    :param x:
    :param t:
    :param path:
    :param title:
    :param label2:
    :param save_svg:
    :return:
    '''
    plt.rcParams['font.family'] = 'DejaVu Serif'
    fig = plt.figure(figsize=(calculate_fig_size(Aspect_Ratio)[0], calculate_fig_size(Aspect_Ratio)[1]))
    #  ################# 图标题 ##################
    # fig.suptitle(f'{title}{label2}', fontname='Times New Roman', fontweight='bold', size=15)
    ax = fig.add_subplot(111)
    ax.axis('off')

    max_value = max(data1.max(), data2.max())
    min_value = min(data1.min(), data2.min())
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max_value)
    #  ################# 左图 ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1, bottom=0, left=0, right=1, wspace=0.3)
    ax1 = plt.subplot(gs0[:, 0])
    data1 = data1.reshape(len(t), len(x))
    #  0313修改：添加了mask操作，将矩阵中大于1的值映射为1，以解决在切换科学计数法表示时出现valueerror的报错；
    data1_max = data1.max()
    #  0313修改--zhanglei
    h = ax1.imshow(data1.T, interpolation='nearest', cmap='viridis',
                   extent=[t.min(), t.max(), x.min(), x.max()],
                   origin='lower', aspect='auto', norm=norm)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.set_ticks(np.linspace(max(-0.999, min_value), min(max_value, 0.999), num=9))
    cbar.ax.tick_params(labelsize=8)
    if data1_max < 1.0:  # and data1_min > -1.0:
        cbar.ax.yaxis.set_major_formatter(CustomFormatter())
        cbar.ax.yaxis.get_offset_text().set_position((2.7, 1))  # 调整偏移文本的位置
        cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置
    ax1.set_title('(a)', fontsize=10)
    #  ################# 坐标轴 ##################
    ax1.set_xlabel('t', fontweight=400, size=10, labelpad=-5)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax1.set_ylabel('x', fontweight=400, size=10, labelpad=-5)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.tick_params(labelsize=8)
    #  ################# 右图 ##################
    ax2 = plt.subplot(gs0[:, 1])
    data2 = data2.reshape(len(t), len(x))
    #  0313修改：添加了mask操作，将矩阵中大于1的值映射为1，以解决在切换科学计数法表示时出现valueerror的报错；
    data2_max = data2.max()
    data2_min = data2.min()
    #  0313修改--zhanglei
    h = ax2.imshow(data2.T, interpolation='nearest', cmap='viridis',
                   extent=[t.min(), t.max(), x.min(), x.max()],
                   origin='lower', aspect='auto', norm=norm)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.set_ticks(np.linspace(max(-0.999, min_value), min(max_value, 0.999), num=9))
    cbar.ax.tick_params(labelsize=8)
    if data2_max < 1.0:  # and data2_min > -1.0:
        cbar.ax.yaxis.set_major_formatter(CustomFormatter())
        cbar.ax.yaxis.get_offset_text().set_position((2.7, 1))  # 调整偏移文本的位置
        cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置
    #  ################# 坐标轴 ##################
    ax2.set_xlabel('t', fontweight=400, size=10, labelpad=-5)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax2.set_ylabel('x', fontweight=400, size=10, labelpad=-5)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax2.set_title('(b)', fontsize=10)
    ax2.tick_params(labelsize=8)
    #  ################# 保存 ##################
    if save_svg:
        plt.savefig(f'{path}{save_title}_{label2}.svg', format='svg', bbox_inches='tight', transparent=True)

    else:
        plt.savefig(f'{path}{save_title}_{label2}.png', dpi=500, bbox_inches='tight', transparent=True)
    plt.close()
    return None


def u_predict(u_vals, U_pred, x, t, nu, beta, rho, seed, layers, N_f, L, source, lr, u0_str, system, path):
    """Visualize u_predicted."""
    plt.rcParams['font.family'] = 'DejaVu Serif'

    fig = plt.figure(figsize=(9, 5))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    ax = fig.add_subplot(111)

    # colorbar for prediction: set min/max to ground truth solution.
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='viridis',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto', vmin=u_vals.min(0), vmax=u_vals.max(0), norm=norm)
    plt.clim(0, 1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    line = np.linspace(x.min(), x.max(), 2)[:, None]

    ax.set_xlabel('t', fontweight='bold', size=30)
    ax.set_ylabel('x', fontweight='bold', size=30)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    ax.tick_params(labelsize=15)
    plt.savefig(
        f"{path}/upredicted_{system}_nu{nu}_beta{beta}_rho{rho}_Nf{N_f}_{layers}_L{L}_seed{seed}_source{source}_{u0_str}_lr{lr}.pdf")
    plt.close()
    return None


def plot_solutions_2(pred, real, x, t, path, label1, label2, save_svg=False):
    plt.rcParams['font.family'] = 'DejaVu Serif'
    u_max = real.max()
    u_min = real.min()
    real = (real - u_min) / (u_max - u_min)
    real = real.reshape(len(t), len(x))
    pred = (pred - u_min) / (u_max - u_min)
    # 此时real、pred均为[t_r,x_r]维的矩阵
    # Creating the figures
    fig, ax = newfig(1.4, 1.8)
    ax.axis('off')  # 不显示x轴和y轴
    low = int(0.25 * len(t))
    mid = int(0.5 * len(t))
    high = int(0.75 * len(t))
    ####### Row 0: u(t_r,x_r) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 2, left=0.1, right=0.9, wspace=0.2)

    ax = plt.subplot(gs0[:, 0])

    h = ax.imshow(real.T, interpolation='nearest',  # cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto', vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_title('real', fontsize=12)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax = plt.subplot(gs0[:, 1])

    h = ax.imshow(pred.T, interpolation='nearest',  # cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto', vmin=0, vmax=1)
    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)

    cbar.set_ticks(np.linspace(0, u_max, num=10))
    cbar.ax.tick_params(labelsize=12)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    cbar.ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel('$t$')
    ax.set_title(label1 + str(label2), fontsize=12)
    ax.set_yticklabels([])
    ####### Row 1: u(t_r,x_r) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 2, bottom=0, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, real[low, :], 'b-', linewidth=2, label='real')
    ax.plot(x, pred[low, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = 25%$', fontsize=10)
    plt.axis('equal')
    ax.set_xlim([-20, 20])
    ax.set_ylim([0, 1])
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, real[mid, :], 'b-', linewidth=2, label='real')
    ax.plot(x, pred[mid, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = 50%$', fontsize=10)
    ax.set_xlim([-20, 20])
    ax.set_ylim([0, 1])
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, real[high, :], 'b-', linewidth=2, label='real')
    ax.plot(x, pred[high, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_xlim([-20, 20])
    ax.set_ylim([0, 1])
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    ax.set_title('$t = 75%$', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, fontsize=12, frameon=False)

    if save_svg:
        # plt.savefig(f'{label2}.svg', format='svg', bbox_inches='tight',transparent=True)
        plt.savefig(f'{path}{label2}.svg', format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.savefig(f'{path}{label2}.png', dpi=500, bbox_inches='tight', transparent=True)
    plt.close()


def plot_solution_with_error_region(data1, data2, error, x, t, path, label2, region, trained_region, save_svg=False):
    '''
    绘制全局的残差分布，与solution绘制不同的是，由于数值普遍较小，在cbar刻度上采用科学计数法的表示，保留了两位小数；
    :param data1:
    :param x:
    :param t:
    :param path:
    :param title:
    :param label2:
    :param save_svg:
    :return:
    '''
    x = np.squeeze(x)
    t = np.squeeze(t)
    Aspect_Ratio = 1 / 1.3
    plt.rcParams['font.family'] = 'DejaVu Serif'
    #  ################# 图标题 ##################
    fig, ax = plt.subplots(figsize=(calculate_fig_size(Aspect_Ratio)[0], calculate_fig_size(Aspect_Ratio)[1]))
    x_min = min(x)
    x_max = max(x)
    t_min = min(t)
    t_max = max(t)

    u_max = data1.max()
    u_min = data1.min()
    data1 = (data1 - u_min) / (u_max - u_min)
    data2 = (data2 - u_min) / (u_max - u_min)
    t_1 = t[int(0.5 * (len(t) - 1))]
    t_2 = t[int(0.8 * (len(t) - 1))]
    t_3 = t[-1]

    gs0 = gridspec.GridSpec(2, 2)
    gs0.update(top=1, bottom=0, left=0, right=1, wspace=0.2)
    #  region 数值解
    ax1 = plt.subplot(gs0[0, 0])
    h = ax1.imshow(data1.T, interpolation='nearest', cmap='viridis',
                   extent=[min(t), max(t), min(x), max(x)],
                   origin='lower', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('Numerical Solution', fontsize=10)
    #  ################# 坐标轴 ##################
    ax1.set_xlabel(r'$t$', fontweight=400, size=10, labelpad=1)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax1.set_ylabel(r'$x$', fontweight=400, size=10, labelpad=-5)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.tick_params(labelsize=10)
    # endregion
    #  region 预测解
    ax2 = plt.subplot(gs0[0, 1])
    h = ax2.imshow(data2.T, interpolation='nearest', cmap='viridis',
                   extent=[min(t), max(t), min(x), max(x)],
                   origin='lower', aspect='auto', vmin=0, vmax=1)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.set_ticks(np.linspace(0, 1, num=10))
    cbar.ax.tick_params(labelsize=10)
    #  ################# 坐标轴 ##################
    ax2.set_xlabel(r'$t$', fontweight=400, size=10, labelpad=1)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax2.set_yticklabels([])
    ax2.set_title('Predicted Solution-' + str(label2), fontsize=10)
    ax2.tick_params(labelsize=10)
    #  ################# 网格线 ##################
    for i, v_lines, in enumerate(region):
        tt = i * ((t_max - t_min) / len(region))
        x_sub_region = (x_max - x_min) / v_lines[0]
        if t_min < tt < t_max:
            ax2.vlines(tt, x_min, x_max, colors=(178 / 255, 178 / 255, 178 / 255, 1), linestyle='--', linewidth=0.5)
        for xx in range(int(v_lines[0]) - 1):
            ax2.hlines((xx + 1) * x_sub_region + x_min, i * ((t_max - t_min) / len(region)),
                       (i + 1) * ((t_max - t_min) / len(region)), colors=(178 / 255, 178 / 255, 178 / 255, 0.8),
                       linestyle='--', linewidth=0.5)
    for rec in trained_region:
        if t_min < rec[0] < t_max:
            ax2.vlines(rec[0], rec[2], rec[3], colors=(181 / 255, 11 / 255, 11 / 255, 1), linestyle='--',
                       linewidth=0.5)
        if t_min < rec[1] < t_max:
            ax2.vlines(rec[1], rec[2], rec[3], colors=(181 / 255, 11 / 255, 11 / 255, 1), linestyle='--',
                       linewidth=0.5)
        if x_min < rec[2] < x_max:
            ax2.hlines(rec[2], rec[0], rec[1], colors=(181 / 255, 11 / 255, 11 / 255, 1),
                       linestyle='--', linewidth=0.5)
        if x_min < rec[3] < x_max:
            ax2.hlines(rec[3], rec[0], rec[1], colors=(181 / 255, 11 / 255, 11 / 255, 1),
                       linestyle='--', linewidth=0.5)

    # endregion
    # region error
    ax3 = plt.subplot(gs0[1, 0])
    h2 = ax3.imshow(error.T, interpolation='nearest',  # cmap='rainbow',
                    extent=[min(t), max(t), min(x), max(x)],
                    origin='lower', aspect='auto', vmin=0, vmax=0.3)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h2, cax=cax)
    cbar.set_ticks(np.linspace(0, 0.3, num=9))
    ax3.set_title('Absolute Error', fontsize=10)
    ax3.set_xlabel('$t$', fontweight=400, size=10, labelpad=1)
    ax3.set_ylabel('$x$', fontweight=400, size=10, labelpad=-4)
    cbar.ax.yaxis.set_major_formatter(CustomFormatter())
    cbar.ax.yaxis.get_offset_text().set_position((2.7, 1))  # 调整偏移文本的位置
    cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置
    #  ################# 网格线 ##################
    for i, v_lines, in enumerate(region):
        tt = i * ((t_max - t_min) / len(region))
        x_sub_region = (x_max - x_min) / v_lines[0]
        if t_min < tt < t_max:
            ax3.vlines(tt, x_min, x_max, colors=(178 / 255, 178 / 255, 178 / 255, 1), linestyle='--', linewidth=0.5)
        for xx in range(int(v_lines[0]) - 1):
            ax3.hlines((xx + 1) * x_sub_region + x_min, i * ((t_max - t_min) / len(region)),
                       (i + 1) * ((t_max - t_min) / len(region)), colors=(178 / 255, 178 / 255, 178 / 255, 0.8),
                       linestyle='--', linewidth=0.5)
    for rec in trained_region:
        if t_min < rec[0] < t_max:
            ax3.vlines(rec[0], rec[2], rec[3], colors=(181 / 255, 11 / 255, 11 / 255, 1), linestyle='--',
                       linewidth=0.5)
        if t_min < rec[1] < t_max:
            ax3.vlines(rec[1], rec[2], rec[3], colors=(181 / 255, 11 / 255, 11 / 255, 1), linestyle='--',
                       linewidth=0.5)
        if x_min < rec[2] < x_max:
            ax3.hlines(rec[2], rec[0], rec[1], colors=(181 / 255, 11 / 255, 11 / 255, 1),
                       linestyle='--', linewidth=0.5)
        if x_min < rec[3] < x_max:
            ax3.hlines(rec[3], rec[0], rec[1], colors=(181 / 255, 11 / 255, 11 / 255, 1),
                       linestyle='--', linewidth=0.5)
    # endregion

    # region solution_in_time
    #     gs_nested = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[1, 1])
    gs_nested = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs0[1, 1], wspace=0.6, hspace=0.3)
    solution_1_1 = data1[int(0.5 * (len(t) - 1)), :]
    solution_2_1 = data1[int(0.8 * (len(t) - 1)), :]
    solution_3_1 = data1[-1, :]

    solution_1_2 = data2[int(0.5 * (len(t) - 1)), :]
    solution_2_2 = data2[int(0.8 * (len(t) - 1)), :]
    solution_3_2 = data2[-1, :]

    ax4 = plt.subplot(gs_nested[0, 0])
    ax4.plot(x, solution_1_1, linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=2,
             label='Numerical Solution')
    ax4.plot(x, solution_1_2, linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=2,
             label='Predicted Solution')
    ax4.set_xticklabels([])
    ax4.set_ylabel(r'$|\Psi(x,t)|$')
    ax4.set_title(f'$t = {format(t_1, ".1f")}$', fontsize=10)
    plt.axis('equal')
    ax4.set_xlim([-20, 20])
    ax4.set_ylim([0, 1])
    ax4.set_aspect('auto')

    ax5 = plt.subplot(gs_nested[1, 0])
    ax5.plot(x, solution_2_1, linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=2,
             label='Numerical Solution')
    ax5.plot(x, solution_2_2, linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=2,
             label='Predicted Solution')
    ax5.set_xticklabels([])
    ax5.set_ylabel(r'$|\Psi(x,t)|$')
    ax5.set_title(f'$t = {format(t_2, ".1f")}$', fontsize=10)
    plt.axis('equal')
    ax5.set_xlim([-20, 20])
    ax5.set_ylim([0, 1])
    ax5.set_aspect('auto')

    ax6 = plt.subplot(gs_nested[2, 0])
    line1, = ax6.plot(x, solution_3_1, linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=2,
                      label='Numerical Solution')
    line2, = ax6.plot(x, solution_3_2, linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=2,
                      label='Predicted Solution')
    ax6.set_xlabel(r'$x$')
    ax6.set_ylabel(r'$|\Psi(x,t)|$')
    title = ax6.set_title(f'$t = {format(t_3, ".1f")}$', fontsize=10)
    title.set_position([0.5, 1.0])
    plt.axis('equal')
    ax6.set_xlim([-20, 20])
    ax6.set_ylim([0, 1])
    ax6.set_aspect('auto')
    for ax in [ax4, ax5, ax6]:
        pos = ax.get_position()
        pos.x0 += 0.1  # 向右移动
        ax.set_position(pos)
    plt.tight_layout()
    #     fig.legend(labels, loc='center', bbox_to_anchor=(0.5, 0.5))
    fig.legend(handles=[line1, line2], loc='center right', ncol=2, frameon=False)
    # endregion

    if save_svg:
        plt.savefig(f'{path}{label2}.svg', format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.savefig(f'{path}{label2}.png', dpi=500, bbox_inches='tight', transparent=False)
    plt.close()


def plot_solution_with_error(data1, data2, error, x, t, path, label2, save_svg=False):
    '''
    绘制全局的残差分布，与solution绘制不同的是，由于数值普遍较小，在cbar刻度上采用科学计数法的表示，保留了两位小数；
    :param data1:
    :param x:
    :param t:
    :param path:
    :param title:
    :param label2:
    :param save_svg:
    :return:
    '''
    x = np.squeeze(x)
    t = np.squeeze(t)
    Aspect_Ratio = 1 / 1.3
    plt.rcParams['font.family'] = 'DejaVu Serif'
    #  ################# 图标题 ##################
    fig, ax = plt.subplots(figsize=(calculate_fig_size(Aspect_Ratio)[0], calculate_fig_size(Aspect_Ratio)[1]))
    x_min = min(x)
    x_max = max(x)
    t_min = min(t)
    t_max = max(t)

    u_max = data1.max()
    u_min = data1.min()
    data1 = (data1 - u_min) / (u_max - u_min)
    data2 = (data2 - u_min) / (u_max - u_min)
    t_1 = t[int(0.5 * (len(t) - 1))]
    t_2 = t[int(0.8 * (len(t) - 1))]
    t_3 = t[-1]

    gs0 = gridspec.GridSpec(2, 2)
    gs0.update(top=1, bottom=0, left=0, right=1, wspace=0.2)
    #  region 数值解
    ax1 = plt.subplot(gs0[0, 0])
    h = ax1.imshow(data1.T, interpolation='nearest', cmap='viridis',
                   extent=[min(t), max(t), min(x), max(x)],
                   origin='lower', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('Numerical Solution', fontsize=10)
    #  ################# 坐标轴 ##################
    ax1.set_xlabel(r'$t$', fontweight=400, size=10, labelpad=1)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax1.set_ylabel(r'$x$', fontweight=400, size=10, labelpad=-5)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.tick_params(labelsize=10)
    # endregion
    #  region 预测解
    ax2 = plt.subplot(gs0[0, 1])
    h = ax2.imshow(data2.T, interpolation='nearest', cmap='viridis',
                   extent=[min(t), max(t), min(x), max(x)],
                   origin='lower', aspect='auto', vmin=0, vmax=1)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.set_ticks(np.linspace(0, 1, num=11))
    cbar.ax.tick_params(labelsize=10)
    #  ################# 坐标轴 ##################
    ax2.set_xlabel(r'$t$', fontweight=400, size=10, labelpad=1)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax2.set_yticklabels([])
    ax2.set_title('Predicted Solution-' + str(label2), fontsize=10)
    ax2.tick_params(labelsize=10)
    # endregion
    # region error
    ax3 = plt.subplot(gs0[1, 0])
    h2 = ax3.imshow(error.T, interpolation='nearest',  # cmap='rainbow',
                    extent=[min(t), max(t), min(x), max(x)],
                    origin='lower', aspect='auto', vmin=0, vmax=0.09)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.09)
    cbar = fig.colorbar(h2, cax=cax)
    cbar.set_ticks(np.linspace(0, 0.09, num=9))
    ax3.set_title('Absolute Error', fontsize=10)
    ax3.set_xlabel('$t$', fontweight=400, size=10, labelpad=1)
    ax3.set_ylabel('$x$', fontweight=400, size=10, labelpad=-4)
    cbar.ax.yaxis.set_major_formatter(CustomFormatter())
    cbar.ax.yaxis.get_offset_text().set_position((2.7, 1))  # 调整偏移文本的位置
    cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置
    # endregion

    # region solution_in_time
    #     gs_nested = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[1, 1])
    gs_nested = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs0[1, 1], wspace=0.6, hspace=0.3)
    solution_1_1 = data1[int(0.5 * (len(t) - 1)), :]
    solution_2_1 = data1[int(0.8 * (len(t) - 1)), :]
    solution_3_1 = data1[-1, :]

    solution_1_2 = data2[int(0.5 * (len(t) - 1)), :]
    solution_2_2 = data2[int(0.8 * (len(t) - 1)), :]
    solution_3_2 = data2[-1, :]

    ax4 = plt.subplot(gs_nested[0, 0])
    ax4.plot(x, solution_1_1, linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=2,
             label='Numerical Solution')
    ax4.plot(x, solution_1_2, linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=2,
             label='Predicted Solution')
    ax4.set_xticklabels([])
    ax4.set_ylabel(r'$|\Psi(x,t)|$')
    ax4.set_title(f'$t = {format(t_1, ".1f")}$', fontsize=10)
    plt.axis('equal')
    ax4.set_xlim([-20, 20])
    ax4.set_ylim([0, 1])
    ax4.set_aspect('auto')

    ax5 = plt.subplot(gs_nested[1, 0])
    ax5.plot(x, solution_2_1, linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=2,
             label='Numerical Solution')
    ax5.plot(x, solution_2_2, linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=2,
             label='Predicted Solution')
    ax5.set_xticklabels([])
    ax5.set_ylabel(r'$|\Psi(x,t)|$')
    ax5.set_title(f'$t = {format(t_2, ".1f")}$', fontsize=10)
    plt.axis('equal')
    ax5.set_xlim([-20, 20])
    ax5.set_ylim([0, 1])
    ax5.set_aspect('auto')

    ax6 = plt.subplot(gs_nested[2, 0])
    line1, = ax6.plot(x, solution_3_1, linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=2,
                      label='Numerical Solution')
    line2, = ax6.plot(x, solution_3_2, linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=2,
                      label='Predicted Solution')
    ax6.set_xlabel(r'$x$')
    ax6.set_ylabel(r'$|\Psi(x,t)|$')
    title = ax6.set_title(f'$t = {format(t_3, ".1f")}$', fontsize=10)
    title.set_position([0.5, 1.0])
    plt.axis('equal')
    ax6.set_xlim([-20, 20])
    ax6.set_ylim([0, 1])
    ax6.set_aspect('auto')
    for ax in [ax4, ax5, ax6]:
        pos = ax.get_position()
        pos.x0 += 0.1  # 向右移动
        ax.set_position(pos)
    plt.tight_layout()
    #     fig.legend(labels, loc='center', bbox_to_anchor=(0.5, 0.5))
    fig.legend(handles=[line1, line2], loc='center right', ncol=2, frameon=False)
    # endregion

    if save_svg:
        plt.savefig(f'{path}{label2}.svg', format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.savefig(f'{path}{label2}.png', dpi=500, bbox_inches='tight', transparent=False)
    plt.close()


def plot_solutions(data_1, data_2, x_1, t_1, x_2, t_2, path, label1, label2, name_1='Numerical Solution',
                   name_2='Predicted Solution', save_svg=True):
    '''
    :param name_2:
    :param data_1:
    :param name_1:
    :param data_2:
    :param x_1:
    :param t_1:
    :param x_2:
    :param t_2:
    :param path:
    :param label1:
    :param label2:
    :param save_svg:
    :return:
    '''

    plt.rcParams['font.family'] = 'DejaVu Serif'

    u_max = data_1.max()
    u_min = data_1.min()
    data_2 = (data_2 - u_min) / (u_max - u_min)
    data_1 = (data_1 - u_min) / (u_max - u_min)

    # Creating the figures
    fig, ax = newfig(1.4, 1.8)
    ax.axis('off')  # 不显示x轴和y轴

    low_2 = int(0.25 * (len(t_2) - 1))
    mid_2 = int(0.5 * (len(t_2) - 1))
    high_2 = int(0.75 * (len(t_2) - 1))

    low_1 = int(0.25 * (len(t_1) - 1))
    mid_1 = int(0.5 * (len(t_1) - 1))
    high_1 = int(0.75 * (len(t_1) - 1))

    ####### Row 0: u(t_r,x_r) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 2, left=0.1, right=0.9, wspace=0.2)
    ax = plt.subplot(gs0[:, 0])

    h = ax.imshow(data_1.T, interpolation='nearest',  # cmap='rainbow',
                  extent=[t_1.min(), t_1.max(), x_1.min(), x_1.max()],
                  origin='lower', aspect='auto', vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_title(name_1, fontsize=11)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')

    ax = plt.subplot(gs0[:, 1])

    h = ax.imshow(data_2.T, interpolation='nearest',  # cmap='rainbow',
                  extent=[t_1.min(), t_1.max(), x_1.min(), x_1.max()],
                  origin='lower', aspect='auto', vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = fig.colorbar(h, cax=cax)
    cbar.set_ticks(np.linspace(0, 1, num=6))
    cbar.ax.tick_params(labelsize=8)
    ax.set_xlabel('$t$')
    ax.set_title(name_2 + '-' + str(label1) + '-' + str(label2), fontsize=11)
    ax.set_yticklabels([])

    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 2, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x_1, data_1[low_1, :], 'b-', linewidth=2, label=name_1)
    ax.plot(x_2, data_2[low_2, :], 'r--', linewidth=2, label=name_2)
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$|\Psi(x,t)|$')
    ax.set_title('t = 25%', fontsize=10)
    plt.axis('equal')
    ax.set_xlim([-20, 20])
    ax.set_ylim([0, 1])
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x_1, data_1[mid_1, :], 'b-', linewidth=2, label=name_1)
    ax.plot(x_2, data_2[mid_2, :], 'r--', linewidth=2, label=name_2)
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$|\Psi(x,t)|$')
    ax.set_title(f't = 50%', fontsize=10)
    ax.set_xlim([-20, 20])
    ax.set_ylim([0, 1])
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x_1, data_1[high_1, :], 'b-', linewidth=2, label=name_1)
    ax.plot(x_2, data_2[high_2, :], 'r--', linewidth=2, label=name_2)
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$|\Psi(x,t)|$')
    ax.set_xlim([-20, 20])
    ax.set_ylim([0, 1])
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    ax.set_title('t = 75%', fontsize=10)

    ax.legend(loc='upper center', bbox_to_anchor=(-0.2, -0.35), ncol=5, fontsize=12, frameon=False)

    if save_svg:
        plt.savefig(f'{path}{label2}.svg', format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.savefig(f'{path}{label2}.png', dpi=500, bbox_inches='tight', transparent=True)
    plt.close()


def plot_PDE_Residual(residual, x, t, path, label1, label2, save_svg=False):
    """Visualize abs(u_pred - u_exact)."""
    plt.rcParams['font.family'] = 'DejaVu Serif'
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    max_residual = np.max(residual)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max_residual)
    residual = residual.reshape(len(t), len(x))
    h = ax.imshow(residual.T, interpolation='nearest', cmap='viridis',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto', norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.set_xlabel('r', fontweight='bold', size=20)
    ax.set_ylabel('x', fontweight='bold', size=20)
    ax.set_title(f'PDE Residual {label1}-{label2}', fontweight='bold', size=30)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.tick_params(labelsize=15)
    if save_svg:
        plt.savefig(f"{path}Residual_{label2}.svg", format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.savefig(f"{path}Residual_{label2}.png", dpi=500, bbox_inches='tight', transparent=True)
    plt.close()
    return None


def plot_diff(real, pred, x, t, path, label1, label2, relative_error=False, save_svg=False):
    """Visualize abs(u_pred - u_exact)."""
    plt.rcParams['font.family'] = 'DejaVu Serif'
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    real = real.reshape(len(t), len(x))
    pred = pred.reshape(len(t), len(x))
    if relative_error:
        h = ax.imshow(np.abs(real.T - pred.T) / np.abs(real.T), interpolation='nearest', cmap='viridis',
                      extent=[t.min(), t.max(), x.min(), x.max()],
                      origin='lower', aspect='auto', norm=norm)
    else:
        h = ax.imshow(np.abs(real.T - pred.T), interpolation='nearest', cmap='viridis',
                      extent=[t.min(), t.max(), x.min(), x.max()],
                      origin='lower', aspect='auto', norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    max_value = np.abs(real.T - pred.T).max()

    cbar.set_ticks(np.linspace(0, max_value, num=10))
    cbar.ax.tick_params(labelsize=12)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    cbar.ax.yaxis.set_major_formatter(formatter)

    # cbar.set_ticks()

    line = np.linspace(x.min(), x.max(), 2)[:, None]

    ax.set_xlabel('t_r', fontweight='bold', size=20)
    ax.set_ylabel('x_r', fontweight='bold', size=20)
    ax.set_title(f'Error {label1}-{label2}', fontweight='bold', size=30)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    ax.tick_params(labelsize=15)

    # plt.savefig(f"{path}/{i}_udiff_Exact_{name}_system_{system}_.{file_type}")
    if save_svg:
        plt.savefig(f"{path}Error_{label2}.svg", format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.savefig(f"{path}Error_{label2}.png", dpi=500, bbox_inches='tight', transparent=True)
    plt.close()


def plot_solution_with_error_and_weight(data1, data2, error, x, t, path, label2, x_f, t_f, values, save_svg=False):
    '''
    绘制全局的残差分布，与solution绘制不同的是，由于数值普遍较小，在cbar刻度上采用科学计数法的表示，保留了两位小数；
    :param data1:
    :param x:
    :param t:
    :param path:
    :param title:
    :param label2:
    :param save_svg:
    :return:
    '''
    Aspect_Ratio = 1 / 1.3
    plt.rcParams['font.family'] = 'DejaVu Serif'
    #  ################# 图标题 ##################
    # fig.suptitle(f'{title}{label2}', fontname='Times New Roman', fontweight='bold', size=15)
    fig, ax = plt.subplots(figsize=(calculate_fig_size(Aspect_Ratio)[0], calculate_fig_size(Aspect_Ratio)[1]))

    u_max = data1.max()
    u_min = data1.min()
    data1 = (data1 - u_min) / (u_max - u_min)
    data2 = (data2 - u_min) / (u_max - u_min)

    gs0 = gridspec.GridSpec(2, 2)
    gs0.update(top=1, bottom=0, left=0, right=1, wspace=0.2)
    #  region 数值解
    ax1 = plt.subplot(gs0[0, 0])
    h = ax1.imshow(data1.T, interpolation='nearest', cmap='viridis',
                   extent=[t.min(), t.max(), x.min(), x.max()],
                   origin='lower', aspect='auto', vmin=0, vmax=1)
    #     divider = make_axes_locatable(ax1)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     cbar = fig.colorbar(h, cax=cax)
    #     cbar.set_ticks(np.linspace(0, 1, num=9))
    #     cbar.ax.tick_params(labelsize=10)
    ax1.set_title('(a) Numerical Solution', fontsize=10)
    #  ################# 坐标轴 ##################
    ax1.set_xlabel(r'$t$', fontweight=400, size=10, labelpad=-5)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax1.set_ylabel(r'$x$', fontweight=400, size=10, labelpad=-5)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.tick_params(labelsize=10)
    ax1.grid(True, which='major', color=(178 / 255, 178 / 255, 178 / 255, 0.6), linestyle='--', linewidth=0.25)
    # endregion
    #  region 预测解
    ax2 = plt.subplot(gs0[0, 1])
    h = ax2.imshow(data2.T, interpolation='nearest', cmap='viridis',
                   extent=[t.min(), t.max(), x.min(), x.max()],
                   origin='lower', aspect='auto', vmin=0, vmax=1)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.set_ticks(np.linspace(0, 1, num=11))
    cbar.ax.tick_params(labelsize=10)
    #  ################# 坐标轴 ##################
    ax2.set_xlabel(r'$t$', fontweight=400, size=10, labelpad=-5)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax2.set_yticklabels([])
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax2.set_title('(b) Predicted Solution-' + str(label2), fontsize=10)
    ax2.tick_params(labelsize=10)
    ax2.grid(True, which='major', color=(178 / 255, 178 / 255, 178 / 255, 0.6), linestyle='--', linewidth=0.25)
    # endregion
    # region error
    ax3 = plt.subplot(gs0[1, 0])
    h2 = ax3.imshow(error.T, interpolation='nearest',  # cmap='rainbow',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto', vmin=0, vmax=0.3)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h2, cax=cax)
    cbar.set_ticks(np.linspace(0, 0.3, num=9))
    ax3.set_title('(c) Absolute Error', fontsize=10)
    ax3.set_xlabel('$t$', fontweight=400, size=10, labelpad=-5)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax3.set_ylabel('$x$', fontweight=400, size=10, labelpad=-5)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    cbar.ax.yaxis.set_major_formatter(CustomFormatter())
    cbar.ax.yaxis.get_offset_text().set_position((2.7, 1))  # 调整偏移文本的位置
    cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置
    ax3.grid(True, which='major', color=(178 / 255, 178 / 255, 178 / 255, 0.6), linestyle='--', linewidth=0.25)
    # endregion

    # region weight
    # 绘制散点图
    ax4 = fig.add_subplot(gs0[1, 1], aspect=1 / 1.3)
    scatter = ax4.scatter(t_f, x_f, c=values, s=4, alpha=0.8, marker='o', cmap='viridis', edgecolors='none')

    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_ticks(np.linspace(0, round_down_auto(max(values)), num=11))
    cbar.ax.tick_params(labelsize=10)

    cbar.ax.yaxis.set_major_formatter(CustomFormatter())
    cbar.ax.yaxis.get_offset_text().set_position((2.7, 1))  # 调整偏移文本的位置
    cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置
    # 设置标签、标题和限制
    ax4.set_xlabel(r'$t$', fontweight=400, size=10, labelpad=-5)
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax4.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax4.set_title('(d) Allocation Weight', fontsize=10)
    ax4.set_xlim([0, 5])
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax4.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax4.set_ylim([-20, 20])
    ax4.set_yticklabels([])
    ax4.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax4.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax4.set_aspect('auto')
    ax4.spines['top'].set_visible(True)
    ax4.spines['right'].set_visible(True)
    ax4.grid(True, which='major', color=(178 / 255, 178 / 255, 178 / 255, 0.6), linestyle='--', linewidth=0.25)
    # endregion

    if save_svg:
        plt.savefig(f'{path}{label2}.svg', format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.savefig(f'{path}{label2}.png', dpi=500, bbox_inches='tight', transparent=False)
    plt.close()


if __name__ == '__main__':
    from pylab import *
    from matplotlib.colors import LogNorm
    import matplotlib.pyplot as plt

    f = np.arange(0, 101)  # frequency
    t = np.arange(11, 245)  # time
    z = 20 * np.sin(f ** 0.56) + 22  # function
    z = np.reshape(z, (1, max(f.shape)))  # reshape the function
    Z = z * np.ones((max(t.shape), 1))  # make the single vector to a mxn matrix
    T, F = meshgrid(f, t)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(F, T, Z)
    plt.xlim((t.min(), t.max()))

    cbar = plt.colorbar()  # the mystery step ???????????
    # cbar.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1]) # add the labels
    plt.show()
