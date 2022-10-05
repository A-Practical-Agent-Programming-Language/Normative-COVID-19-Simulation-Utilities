"""Setup."""
"""Copied from Pansim and adapted for calibration tool"""

from setuptools import setup

package_name = "Disease Simulation Calibration Tool"
description = "Tool to calibrate the disease or behavior models of a 2APl/Pansim integrated simulation of behavior during novel pandemics"


classifiers = """
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Operating System :: POSIX :: Linux
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Topic :: Scientific/Engineering
Topic :: System :: Distributed Computing
""".strip().split(
    "\n"
)

setup(
    name=package_name,
    description=description,
    version="1.0",
    zip_safe=True,
    entry_points="""
        [console_scripts]
        calibration=calibration:start
        behavior_rmse=calibration:behavior_rmse
        disease_rmse=calibration:disease_rmse
    """,
    use_scm_version=True,
    install_requires=[
        "click",
        "click_completion",
        "toml",
        "scipy",
        "sklearn",
        "matplotlib",
        "numpy",
        "bayesian_optimization"
    ],
)
