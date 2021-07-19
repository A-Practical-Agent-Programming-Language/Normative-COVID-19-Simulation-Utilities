import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Callable, Dict

import click
import toml

from classes.EssentialLocations import EssentialDesignationExtractor


def make_file_absolute(relative_to: str, file: str) -> (str, bool):
    """
    Make a passed file path absolute; relative to some other file path if not already absolute
    Args:
            relative_to: File location for reference
            file:        Path to file to make absolute

    Returns:
            1) Absolute file path, where the path is interpreted relative to {relative_to} if not already absolute
            2) Boolean indicating whether the absolute path exists
    """
    if os.path.isabs(file):
        return file, os.path.exists(file)
    else:
        resolved_path = os.path.abspath(os.path.join(relative_to, file))
        return resolved_path, os.path.exists(resolved_path)


def load_toml_configuration(county_config_file):
    """
    Parse the county configuration TOML file
    Args:
            county_config_file: Location of TOML config file

    Returns:
            Dictionary containing the parsed configuration
    """
    missing = list()
    conf = toml.load(county_config_file)
    county_config_location = os.path.dirname(county_config_file)
    norms, norms_exist = make_file_absolute(
        county_config_location, conf["simulation"]["norms"]
    )
    if not norms_exist:
        missing.append(("simulation.norms", [conf["simulation"]["norms"]]))
    conf["simulation"]["norms"] = norms

    disease, disease_exists = make_file_absolute(
        county_config_location, conf["simulation"]["diseasemodel"]
    )
    if not disease_exists:
        missing.append(("simulation.disease", [conf["simulation"]["diseasemodel"]]))
    conf["simulation"]["diseasemodel"] = disease

    for county, cconf in conf["counties"].items():
        for k in [
            "activities",
            "households",
            "persons",
            "locations",
            "statefile",
            "locationDesignations",
        ]:
            if k not in cconf:
                continue
            missing_in_key = []
            updated = []
            for f in cconf[k] if type(cconf[k]) is list else [cconf[k]]:
                path, exists = make_file_absolute(county_config_location, f)
                updated.append(path)
                if not exists:
                    missing_in_key.append(f)
            if len(missing_in_key):
                missing.append((f"counties.{county}.{k}", missing_in_key))
            conf["counties"][county][k] = updated

        conf["counties"][county] = ensure_location_designation_present(
            conf["counties"][county]
        )

    if len(missing):
        error = f"Some paths specified in {county_config_location} could not be resolved:\n\n"
        for (m, m_) in missing:
            error += m + ":\n"
            for f in m_:
                error += "\t" + f + "\n"

        raise click.exceptions.BadParameter(error, param_hint="county-configuration")

    return conf


def ensure_location_designation_present(county):
    if "locationDesignations" not in county:
        county["locationDesignations"] = [
            EssentialDesignationExtractor().from_county(county)
        ]
    updated = []
    for locdes in county["locationDesignations"]:
        path, _ = make_file_absolute(os.curdir, locdes)
        updated.append(path)
    county["locationDesignations"] = updated
    return county


def test_code_available(java_location: os.PathLike) -> None:
    """
    Do a quick test to see if PanSim is available and accessible
    Args:
            java_location: Location of Java executable on this system

    Throws a bad argument exception if PanSim is not working
    """
    try:
        result = subprocess.run(
            [java_location, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        print("Using", str(result.stdout.splitlines()[0], "utf-8"))
    except FileNotFoundError as e:
        raise click.exceptions.BadParameter(
            f"The provided Java binary {java_location} could not be used to start a Java VM",
            param_hint="--java",
        )

    result = subprocess.run(
        ["pansim", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    if result.returncode == 0:
        print("Pansim is available")
    else:
        raise click.exceptions.BadArgumentUsage(
            f"PanSim was not found. Please install PanSim before attempting calibration"
        )


def get_expected_end_date(toml_county_configuration: str, num_steps: int) -> datetime:
    """
    Get the expected date of the last simulation date
    Args:
            toml_county_configuration: Location of county configuration file
            num_steps: Number of steps this simulation runs

    Returns:
            String object of date of last expected simulation day
    """
    config = toml.load(toml_county_configuration)
    last_date = config["simulation"]["startdate"] + timedelta(days=num_steps)
    return last_date


def extract_run_configuration_from_behavior_path(
    behavior_path: str,
) -> Dict[str, float]:
    """
    Extract all parameter values relevant to behavior calibration from the name of a path containing a run
    configuration. Also used in calibration.results.log
    Args:
        behavior_path: Path or name of run directory from which to extract parameters

    Returns:
        Dictionary containing the values for liberal, conservative, fatigue and fatigue_start
    """
    re_params = (
        r"(\d+(?:\.|e-)\d+)l-(\d+(?:\.|e-)\d+)c-(\d+(?:\.|e-)\d+)f-(\d+(?:\.|e-)\d+)fs"
    )
    params_match = re.findall(re_params, behavior_path)
    extracted = dict()
    if params_match is not None and len(params_match):
        liberal, conservative, fatigue, fatigue_start = params_match[0]
        extracted["liberal"] = float(liberal)
        extracted["conservative"] = float(conservative)
        extracted["fatigue"] = float(fatigue)
        extracted["fatigue_start"] = float(fatigue_start)
        return extracted


def extract_run_configuration_from_disease_path(disease_path: str) -> Dict[str, float]:
    """
    Extract all parameter values relevant to disease calibration from the name of a path containing a run
    configuration. Also used in calibration.results.log
    Args:
        disease_path: Path or name of run directory from which to extract parameters

    Returns:
        Dictionary containing the values for isymp, iasymp, scale, liberal, conservative,fatigue, and fatigue_start

    """
    re_params = (
        r"(\d+(?:\.|e-)\d+(?:e-\d+)?)isymp--?"
        r"(\d+(?:\.|e-)\d+(?:e-\d+)?)iasymp-(?:-?(\d+(?:\.|e-)\d+(?:e-\d+)?)scale-)?"
        r"(\d+(?:\.|e-)\d+(?:e-\d+)?)l-(\d+(?:\.|e-)\d+(?:e-\d+)?)c-(\d+(?:\.|e-)\d+(?:e-\d+)?)f-"
        r"(\d+(?:(?:\.|e-)\d+)?(?:e-\d+)?)fs"
    )
    extracted = dict()
    params_match = re.findall(re_params, disease_path)
    if params_match is not None and len(params_match):
        if len(params_match[0]) == 6:
            # Old naming
            isymp, iasymp, liberal, conservative, fatigue, fatigue_start = params_match[
                0
            ]
            scale = 30
        elif len(params_match[0]) == 7:
            # new naming
            (
                isymp,
                iasymp,
                scale,
                liberal,
                conservative,
                fatigue,
                fatigue_start,
            ) = params_match[0]
        extracted["isymp"] = float(isymp)
        extracted["iasymp"] = float(iasymp)
        extracted["scale"] = float(scale)
        extracted["liberal"] = float(liberal)
        extracted["conservative"] = float(conservative)
        extracted["fatigue"] = float(fatigue)
        extracted["fatigue_start"] = float(fatigue_start)

    return extracted


def find_runs(
    simulation_output_path: str,
    target_file_name: str,
    extract_config: Callable[[str], Dict[str, float]],
    to_tuple: Callable[[Dict[str, float]], tuple],
):
    """
    Get a dictionary of all simulations, multiple runs grouped by parameter configuration
    Args:
       simulation_output_path:  Path containing the simulation runs output
       target_file_name:        File that should be present to calculate specific RMSE (e.g. tick averages)
       extract_config:          Method that can extract configuration parameters from directory name
       to_tuple:                Method that converts a dictionary of configuration parameters to tuple of only values

    Returns:
        Dictionary where the keys are tuples of configuration parameters, and values are lists of directories with
        individual runs for those parameters
    """
    runs = defaultdict(list)
    if target_file_name in os.listdir(simulation_output_path):
        # Single run provided
        run_config = extract_config(str(simulation_output_path))
        if run_config is not None:
            runs[to_tuple(run_config)].append(str(simulation_output_path))
    else:
        # Collection of runs, all need to be judged
        for f in os.listdir(simulation_output_path):
            realpath = os.path.join(simulation_output_path, f)
            run_config = extract_config(f)
            if run_config is not None:
                if target_file_name in os.listdir(realpath):
                    runs[to_tuple(run_config)].append(realpath)
                else:
                    print("Missing", target_file_name, "file in", f)
    return runs
