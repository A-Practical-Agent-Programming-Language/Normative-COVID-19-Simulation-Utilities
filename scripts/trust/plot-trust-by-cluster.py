import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

simulation_output = sys.argv[1]
j = os.path.join
trust_files = [
    j(simulation_output, "trust", x)
    for x in os.listdir(j(simulation_output, "trust"))
    if "trust-timestep-" in x and x.endswith(".csv")
]


trust = defaultdict(dict)
for tf in trust_files:
    timestep_match = re.findall(r'trust-timestep-(\d+).csv', tf)
    with open(tf, 'r') as tf_in:
        for line in tf_in:
            agent, trust_val = line[:-1].split(";")
            trust[int(agent)][int(timestep_match[0])] = float(trust_val)

bucketed_agents = defaultdict(list)
for agent, curve in trust.items():
    t = curve[min(list(curve.keys()))]
    bucketed_agents[round(t, 1)].append(agent)

buckets = sorted(list(bucketed_agents.keys()))

fig, ax = plt.subplots()
x = sorted([int(re.findall(r'trust-timestep-(\d+).csv', x)[0]) for x in trust_files])
for bucket in buckets:
    vals = [[trust[agent][t] for agent in bucketed_agents[bucket]] for t in x]
    avg = np.array(list(map(np.mean, vals)))
    std = np.array(list(map(np.std, vals)))
    line = ax.plot(x, avg, label=r'$t_0 = {0}$'.format(bucket))[0]
    ax.fill_between(x, avg-std, avg+std, color=line.get_color(), alpha=0.2, antialiased=True)

plt.title("Evolution of trust bucketed by initial trust $t_0$")
plt.ylabel("Trust $t_x$")
plt.xlabel("Simulation time step")
plt.legend()
plt.show()

