import os
import re
from collections import defaultdict
from typing import Dict, Callable, List


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
        r"(\d+(?:\.|e-)\d+)l-(\d+(?:\.|e-)\d+)c-(\d+(?:\.|e-)\d+)f-(\d+(?:\.|e-)?\d+)fs"
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
        r"(\d+(?:\.|e-)\d+(?:e-\d+)?)isymp--?(\d+(?:\.|e-)\d+(?:e-\d+)?)iasymp-"
        r"(?:-?(\d+(?:\.|e-)\d+(?:e-\d+)?)scale-)?(\d+(?:\.|e-)\d+(?:e-\d+)?)l-"
        r"(\d+(?:\.|e-)\d+(?:e-\d+)?)c-(\d*(?:\.|e-)\d+(?:e-\d+)?)f-"
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
        try:
            extracted["scale"] = float(scale)
        except ValueError:
            extracted["scale"] = ''
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
) -> Dict[tuple, List[str]]:
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
            if run_config is not None and os.path.isdir(realpath):
                if target_file_name in os.listdir(realpath):
                    runs[to_tuple(run_config)].append(realpath)
                else:
                    print("Missing", target_file_name, "file in", f)
    return runs


def find_behavior_runs(simulation_output_path: str) -> Dict[tuple, List[str]]:
    return find_runs(
        simulation_output_path,
        "tick-averages.csv",
        extract_run_configuration_from_behavior_path,
        behavior_params_dict_to_tuple,
    )


def find_disease_runs(simulation_output_path: str) -> Dict[tuple, List[str]]:
    return find_runs(
        simulation_output_path,
        "epicurve.sim2apl.csv",
        extract_run_configuration_from_disease_path,
        disease_params_dict_to_tuple,
    )


def disease_params_dict_to_tuple(params_dct: Dict[str, float]) -> tuple:
    key_order = [
        "isymp",
        "iasymp",
        "scale",
        "liberal",
        "conservative",
        "fatigue",
        "fatigue_start",
    ]
    return tuple([params_dct[x] for x in key_order])


def behavior_params_dict_to_tuple(params_dct: Dict[str, float]) -> tuple:
    return (
        params_dct["liberal"],
        params_dct["conservative"],
        params_dct["fatigue"],
        params_dct["fatigue_start"],
    )


def runs_to_rmse_list(grouped_runs: List[str]) -> List[Dict[int, str]]:
    """
    Takes a list of runs, and creates a dictionary that can be passed to the calculate_RMSE functions.
    All paths in the list should refer to the same run configuration

    Args:
        grouped_runs: List of runs with the same run configuration

    Returns:
        Singleton list with dictionary with run number as key and path to run as value
    """
    runs = dict()
    for run in grouped_runs:
        match = re.findall(r"-run(\d+)$", run)
        run_number = int(match[0][0])
        runs[run_number] = run
    return [runs]
