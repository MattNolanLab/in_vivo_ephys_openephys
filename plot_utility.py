import matplotlib.pylab as plt
import math
import numpy as np


def draw_reward_zone():
    for stripe in range(8):
        if stripe % 2 == 0:
            plt.axvline(91.25+stripe*2.5, color='limegreen', linewidth=5.5, alpha=0.4, zorder=0)
        else:
            plt.axvline(91.25+stripe*2.5, color='k', linewidth=5.5, alpha=0.4, zorder=0)


def draw_black_boxes():
    plt.axvline(15, color='k', linewidth=66, alpha=0.25, zorder=0)
    plt.axvline(185, color='k', linewidth=66, alpha=0.25, zorder=0)


def style_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    return plt, ax


def style_open_field_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off

    ax.set_aspect('equal')
    return ax


def style_polar_plot(ax):
    ax.spines['polar'].set_visible(False)
    ax.set_yticklabels([])  # remove yticklabels
    ax.grid(None)
    plt.xticks([math.radians(0), math.radians(90), math.radians(180), math.radians(270)])
    ax.axvline(math.radians(90), color='black', linewidth=1, alpha=0.6)
    ax.axvline(math.radians(180), color='black', linewidth=1, alpha=0.6)
    ax.axvline(math.radians(270), color='black', linewidth=1, alpha=0.6)
    ax.axvline(math.radians(0), color='black', linewidth=1, alpha=0.6)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2.0)
    return ax