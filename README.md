This library contains utility code for use with the 
[Normative COVID-19 Simulation repository](https://github.com/A-Practical-Agent-Programming-Language/Normative-COVID-19-Simulation).

This library has been used to automate the various processes for which repeated or multiple runs of the simulation
were required, amongst which such things as calibration of the behavior and disease model components, experiments,
scaling tests, and sensitivity analysis.

The automation deals with automatically creating all the files and arguments required for each individual simulation
by either the behavior model or by [PanSim](https://github.com/parantapa/pansim)

The library further contains a number of scripts that can be helpful in the analysis of results, or the production
of plots or paper diagrams. 

However, the library has been primarily created for personal use, so the code is not properly documented. 
This may limit this library's usefulness to others.

# Installation
Since this library is used to manage the integration of the behavior model and PanSim, 
it requires the 
[behavior model](https://github.com/A-Practical-Agent-Programming-Language/Normative-COVID-19-Simulation) to be set up according to the instructions in that repository, and
[PanSim](https://github.com/parantapa/pansim) to be available and installed as well.

All individual Python files can be invoked directly in your preferred Pythonesque way. 
The entire library can be installed by typing `pip install -e .` in the root directory of this (cloned) repository.
It is advised to use a virtual environment (e.g. venv or conda), and to share that virtual environment with PanSim.

After installation, type `calibration -h` (in the activated environment) for how this program can be invoked.

# License
This library contains free software; The code can be freely used under the Mozilla Public License 2.0. 
See the [license file](LICENSE) for details.
This code comes with ABSOLUTELY NO WARRANTY, to the extent permitted by applicable law.
