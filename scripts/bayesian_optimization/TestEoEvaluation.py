"""
Simple run-once script that can be used to test if the fitness function and assigned weights make sense
"""
import os.path
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from classes.ExecutiveOrderOptimizer.EOOptimization import EOOptimization
from utility.utility import get_project_root


def load_norm_weights(norm_weights_file, affected_agents_file):
    norm_weights = EOOptimization.load_norm_weights(norm_weights_file)
    affected_agents = EOOptimization.load_norm_application_counts_from_file(affected_agents_file)

    norms = defaultdict(dict)
    for norm, weight in norm_weights.items():
        norms[norm]['weight'] = weight
        norms[norm]['affected'] = affected_agents[norm]['affected_agents']
        norms[norm]['duration'] = affected_agents[norm]['affected_duration']
        norms[norm]['relative'] = weight * norms[norm]['affected']
        norms[norm]['relative_duration'] = weight * norms[norm]['duration']

    return norms


def plot(norms, subtitle="Default Weights"):
    sorted_norms = sorted(list(norms.keys()), key=lambda x: norms[x]['weight'])
    weights = [norms[norm]['weight'] for norm in sorted_norms]
    relative = [norms[norm]['relative'] for norm in sorted_norms]
    duration = [norms[norm]['relative_duration'] for norm in sorted_norms]

    bar_width = 0.2

    y_pos = np.arange(len(sorted_norms))

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    plt.xticks(y_pos, sorted_norms, rotation=90, ha='center')

    lines = [ax.bar(y_pos - bar_width, weights, bar_width)]
    ax2 = ax.twinx()
    ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
    ax2._get_patches_for_fill.prop_cycler = ax._get_patches_for_fill.prop_cycler
    ax3 = ax2.twinx()
    ax3._get_patches_for_fill.prop_cycler = ax._get_patches_for_fill.prop_cycler
    lines.append(ax2.bar(y_pos + 0.025, relative, bar_width))

    ax2.hlines(norms["SchoolsClosed[K12]"]['relative'], 0, y_pos[-1], color=lines[-1].patches[0].get_facecolor())
    ax2.hlines(norms["SchoolsClosed[K12;HIGHER_EDUCATION]"]['relative'], 0, y_pos[-1], color=lines[-1].patches[0].get_facecolor())

    lines.append(ax3.bar(y_pos + bar_width + 0.05, duration, bar_width))

    plt.suptitle("Norm weights & penalty per week active")
    plt.title(subtitle)

    ax.legend(lines, ["Weight", "Penalty per Week", "If using duration"])

    plt.tight_layout()
    plt.savefig(
        os.path.join(get_project_root(), "output", f"norm-weights-vs-weekly-penalty-{subtitle.replace(' ', '_')}.png"),
        dpi=300
    )
    plt.show()


if __name__ == "__main__":
    for weight, weight_file in [
        ("default weights", "norm_weights"),
        ("favour economy", "norm_weights_favour_economy"),
        ("schools open", "norm_weights_favour_schools_open")
    ]:
        norm_info = load_norm_weights(
            os.path.join(get_project_root(), "external", f"{weight_file}.csv"),
            os.path.join(get_project_root(), ".persistent", "affected-agents-per-norm-65-75-109-540.csv")
        )
        plot(norm_info, weight)



