import sys
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.scale import LogScale
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator, FormatStrFormatter

by_cores = defaultdict(dict)
by_agents = defaultdict(dict)

cpus = set()
agents = set()

# markers = ".v^1spPh+xd"
markers = ".1x234|*+"

# Colors generated with https://gka.github.io/palettes/#/9|d|00429d,96ffea,ffffe0|ffffe0,ff005e,93003a|1|1
# The tool says the palette is colorblind-safe
colors = ['#00429d', '#4771b2', '#73a2c6', '#a5d5d8', '#ffffe0', '#ffbcaf', '#f4777f', '#cf3759', '#93003a']

style.use('tableau-colorblind10')
matplotlib.rcParams.update(
    {
        'text.usetex': True,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)
plt.rcParams.update(
        {
            'text.usetex': True,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
        }
    )
plt.rcParams['legend.handlelength'] = 1.0
plt.rcParams['axes.linewidth'] = 0.1
plt.rcParams['ytick.major.size'] = 5.0

with open(sys.argv[1], 'r') as scaling_results_in:
    headers = scaling_results_in.readline()[:-1].split(',')
    for line in scaling_results_in:
        data = dict(zip(headers, line[:-1].split(",")))
        cpu = int(data['n_cpus'])
        persons = int(data['n_persons']) / 1000
        runtime = int(data['runtime_mean_sec'])
        by_cores[data['place']][cpu] = runtime
        by_agents[cpu][persons] = runtime
        cpus.add(cpu)
        agents.add(persons)


def do_common_plot(ax, x, y_series, max_x_tick_power, sort_kwargs):
    assert len(markers) >= len(y_series)

    for marker, color, (label, series) in zip(markers, colors, sorted(y_series.items(), **sort_kwargs)):
        y = [series[_x] if _x in series else float('nan') for _x in x]
        ax.plot(x, y, label=label, marker=marker)
    ax.set_xscale(LogScale(axis=ax.xaxis, base=2))
    ax.set_xticks([2 ** i for i in range(5, max_x_tick_power)])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yscale(LogScale(axis=ax.yaxis, base=2))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(FormatStrFormatter(""))
    all_y_ticks = [10**i * j for i in range(2, 6) for j in range(10) if 100 <= 10**i*j <= 100000]
    y_ticks = [y_tick for y_tick in all_y_ticks if str(y_tick)[0] in "12"]
    y_ticks_minor = all_y_ticks

    ax.set_yticks(y_ticks)
    ax.set_yticks(y_ticks_minor, minor=True)

    ax.set_ylabel("Runtime (seconds)")

    ax.grid(which='both', antialiased=True, animated=True, alpha=0.5, linewidth=0.3)


cm = 1/2.54
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20 * cm, 8 * cm), sharey='row')

do_common_plot(ax1, sorted(list(cpus)), by_cores, 11, sort_kwargs=dict(key=lambda x: max(x[1].values()), reverse=True))
ax1.set_xlabel(r"\# CPU cores")
ax1legend = ax1.legend(fontsize=7, title="region", ncol=2, loc='upper left', bbox_to_anchor=(0.01, 1.13))
ax1legend._legend_box.align = "left"
ax1.set_title("(a)", y=0, pad=-45)

do_common_plot(ax2, sorted(list(agents)), by_agents, 14, sort_kwargs=dict(key=lambda x: x, reverse=False))
ax2.set_xlabel(r"\# simulated individuals (thousands)")
ax2.set_title("(b)", y=0, pad=-45)
ax2.legend(fontsize=7, title=r"\#cpu cores")

plt.tick_params(axis='y', which='minor')

fig.set_dpi(300)
fig.tight_layout()
if len(sys.argv) > 1:
    plt.savefig(sys.argv[2], dpi=300)
plt.show()
