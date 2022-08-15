import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

cm = 1/2.54

def attitude(trust: float, factor_average: float, logistic_growth: float = 10):
    return 1 - (1 / (1 + math.pow(math.e, -1 * logistic_growth * ((1 - trust) - factor_average))))


def symptomatic_contacts_linear(seen_symptomatic: int, kappa: float = 0.2):
    return max(0.0, 1 - kappa * seen_symptomatic)


def symptomatic_contacts_complex(seen_symptomatic: int, average_symptomatic: float, kappa: float = 0.2):
    """
    This one clearly does not make sense, so lets not use it
    """
    a = 1 - symptomatic_contacts_linear(seen_symptomatic, kappa)
    b = average_symptomatic
    return 1 - (((1 - b) * a) + b * b)


def group_size(seen: int, allowed: int, cutoff_constant: float = 0.4):
    return 0 if seen <= allowed else 1 - (1 / (cutoff_constant * (seen - allowed) + 1))


def plot_attitude(ax):
    x = np.arange(0, 1, 0.01)
    plot_ax(
        ax,
        x,
        [None] * len(x),
        0, 1,
        r"$\bar{f}$",
        r"$a(\beta)$",
        "(a) Attitude"
    )

    ax.set_prop_cycle(None)

    linestyles = ['dotted', 'solid', 'dashed']
    for linestyle, trust in zip(linestyles, [1, 0.5, 0]):
        ax.plot(x, [attitude(trust, _x) for _x in x], label=fr"$t = {trust}$", linestyle=linestyle)

    # ax.legend(prop={'size': 5}, loc='center left', bbox_to_anchor=(-0.5, 0.5))


def plot_group_size(ax):
    x = np.arange(0, 100, 1)
    plot_ax(
        ax,
        x,
        [group_size(_x, 10) for _x in x],
        8, 30,
        r"$\bar{n}^l_{\Delta d}$",
        r"$f_4$",
        "(c) Group size"
    )
    ax.set_xticks(np.arange(5, 31, 5))


def plot_symptomatic(ax):
    x = np.arange(0, 100, 1)
    plot_ax(
        ax,
        x,
        [symptomatic_contacts_linear(_x) for _x in x],
        0, 10,
        r"$s$", r"$f_3$",
        "(b) Symptomatic"
    )


def plot_ax(ax, x, y, x_min, x_max, x_label, y_label, title):
    ax.plot(x, y)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([0, 1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=8, y=0, pad=-40)
    ax.set_yticks([0, 0.5, 1])
    plt.setp(ax.get_xticklabels(), fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)


def plot_relevant_curves():
    # plt.style.use('seaborn')
    plt.rcParams.update(
        {
            'font.size': 4,
            'text.usetex': True,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
        }
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey='all')

    plot_attitude(ax1)
    plot_symptomatic(ax2)
    plot_group_size(ax3)

    fig.set_size_inches(12 / 2.54, 4 / 2.54)
    fig.set_dpi(300)
    fig.tight_layout()

    if sys.argv[1]:
        plt.savefig(sys.argv[1], dpi=500)

    fig.show()


def plot_one_curve(f):
    plt.style.use('tableau-colorblind10')
    plt.rcParams.update(
        {
            'text.usetex': True,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
        }
    )
    plt.rcParams['axes.linewidth'] = 0.1
    fig, ax = plt.subplots(1, 1, figsize=(5 * cm, 5 * cm))

    f(ax)

    fig.set_dpi(300)
    ax.set_title(None)

    fig.tight_layout()

    if sys.argv[1]:
        plt.savefig(sys.argv[1], dpi=500)

    fig.show()


def plot_attitude_as_one_curve():
    style.use('tableau-colorblind10')
    plt.rcParams.update(
        {
            'text.usetex': True,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
        }
    )
    plt.rcParams['axes.linewidth'] = 0.1

    fig, (ax, hidden_ax) = plt.subplots(1, 2, figsize=(10 * cm, 5 * cm))
    hidden_ax.axis('off')

    plot_attitude(ax)

    fig.set_dpi(300)
    ax.set_title(None)

    fig.legend(fontsize=7, loc='upper right', bbox_to_anchor=(0.8, 0.9))

    fig.tight_layout()

    # if sys.argv[1]:
    #     plt.savefig(sys.argv[1], dpi=500)

    fig.show()


if __name__ == "__main__":
    # plot_relevant_curves()
    # plot_one_curve(plot_group_size)
    plot_attitude_as_one_curve()
