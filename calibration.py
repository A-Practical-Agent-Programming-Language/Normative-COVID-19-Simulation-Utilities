#!/bin/python3
import random
import sys

import click
from scipy.optimize import minimize

from classes.Epicurve_RMSE import EpicurveRMSE
from classes.ExecutiveOrderOptimizer.EOOptimization import EOOptimization
from classes.Gyration import Gyration
from classes.execution.BehaviorCalibration import BehaviorCalibration
from classes.execution.DiseaseCalibration import DiseaseCalibration
from classes.execution.NormExperiment import NormExperiment
from classes.execution.RepeatedExecution import RepeatedExecution
from classes.execution.RunWithNormSchedules import RunWithNormSchedules
from classes.execution.ScalingTestExecution import ScalingTestExecution
from classes.execution.SensitityAnalysis import SensitivityAnalysis
from classes.execution.SlaveCodeExecution import SlaveCodeExecution
from classes.execution.testTrustDiscountFactor import TestTrustDiscountFactor
from classes.rmse_backlog.DiseaseRMSEOnBacklog import DiseaseRMSEOnBacklog
from classes.rmse_backlog.MobilityRMSEOnBacklog import MobilityRMSEOnBacklog
from utility.utility import *


@click.group()
@click.option(
    "--county-configuration",
    "-c",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, resolve_path=True),
    help="Specify the TOML file containing the county configuration for both the agent model and the calibration process",
    required=True,
)
@click.option(
    "--behavior-model",
    "-b",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, resolve_path=True),
    help="Specify the location of the behavior model .JAR file",
    required=True,
)
@click.option(
    "--pansim-num-cpus",
    type=int,
    default=1,
    help="Specify the number of CPUs (threads) to use with PanSim",
)
@click.option(
    "--java",
    "-j",
    type=str,
    default="java",
    help="Specify the Java binary to use for code execution",
)
@click.option(
    "--java-num-cpus",
    type=int,
    default=1,
    help="Specify the number of CPUs (threads) to use with Java",
)
@click.option(
    "-Xmx",
    type=str,
    default="64g",
    help="Specify the maximum heap size following Java argument formatting (see Java man pages for help, Java argument is -Xmx",
)
@click.option(
    "-Xms",
    type=str,
    default="55g",
    help="Specify the initial heap size following Java argument formatting (see Java man pages for help, Java argument is -Xms",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(dir_okay=True, file_okay=True, resolve_path=True),
    default="output/agent-runs",
    help="Specify an alternative directory where the behavior model output files will be stored",
)
@click.option(
    "--number-of-runs",
    "-r",
    type=int,
    default=10,
    help="How many times should the simulation be run for each configuration",
)
@click.option(
    "--number-of-steps",
    "-n",
    type=int,
    default=120,
    help="How many time steps should the simulation comprise of (starting March 1th 2020)",
)
@click.option("--name", type=str, default=None, help="Short descriptive name")
@click.option(
    "--is-master",
    type=bool,
    default=False,
    help="If true, this code will leave instructions for other compute nodes to run a specific configuration, instead"
    "of starting the calibration run itself, except if the run is the nth number with n = --number-of-runs",
)
@click.pass_context
def start(ctx, **kwargs):
    """
    This is a tool for calibrating the disease or behavior model of a Covid-19 simulation using Nelder Mead minimization.

    This tool requires being run in an environment with PanSim installed.
    This tool requires a JAR file of the behavior model. The easiest way to obtain the latest version of the behavior
    model is to build it from the source. Follow the instructions in the GIT repo:
            https://bitbucket.org/goldenagents/sim2apl-episimpledemics/src/master/

    The configuration of the Java behavior model can be re-used for this calibration, but requires the addition
    of the disease model.

    If the Java JAR file is located in a different directory than this tool, make sure to specify all path locations
    in the configuration file as absolute paths.

    Logging in Java can be controlled by adding a (or symlinking the default) logging.properties file in this
    directory.

    This tool makes use of the following Python scripts developed by other authors:
            * gyration_radius_calculator.py from Samarth (make sure to place in same directory)

    The same command line call contains arguments for how large the Java heap size
            should be. DOn't make this too small, or Java will crash with an out-of-memory
            error, or be SIGKILL'd by the OS

    If changing the county, make sure to update the config files.
    """
    test_code_available(kwargs["java"])
    seed = random.randrange(sys.maxsize)
    county_configuration = load_toml_configuration(kwargs["county_configuration"])
    county_configuration["simulation"]["seed"] = seed
    conf_file_tmp = os.path.join(".persistent", ".tmp", "java_model_config.toml")
    if not os.path.exists(os.path.dirname(conf_file_tmp)):
        os.makedirs(os.path.dirname(conf_file_tmp))
    with open(conf_file_tmp, "w") as fout:
        toml.dump(county_configuration, fout)

    counties = county_configuration["counties"]
    disease_model = county_configuration["simulation"]["diseasemodel"]
    norm_schedule = county_configuration["simulation"]["norms"]

    ctx.obj = dict(
        counties=counties,
        args=dict(
            county_configuration_file=conf_file_tmp,
            seed=seed,
            sim2apl_jar=kwargs["behavior_model"],
            counties=counties,
            disease_model_file=disease_model,
            norm_schedule=norm_schedule,
            n_cpus=kwargs["pansim_num_cpus"],
            java_home=kwargs["java"],
            java_threads=kwargs["java_num_cpus"],
            java_heap_size_max=kwargs["xmx"],
            java_heap_size_initial=kwargs["xms"],
            output_dir=kwargs["output_dir"],
            n_runs=kwargs["number_of_runs"],
            n_steps=kwargs["number_of_steps"],
            name=kwargs["name"],
            is_master=kwargs["is_master"],
        ),
    )


def print_dict(dct, ident=0):
    for k, v in dct.items():
        if type(v) == type(dct):
            print("\t" * ident, k)
            print_dict(v, ident + 1)
        else:
            print("\t" * ident, k, v)


@start.command(
    name="behavior",
    help="Start the behavior model calibration with the disease model fixed",
)
@click.option(
    "--mobility-index-file",
    "-m",
    type=click.Path(exists=True),
    help="Specify the location of the file containing the mobility index for each (relevant) county",
    default=os.path.join("external", "va_county_mobility_index.csv"),
)
@click.option(
    "--tick-averages-file",
    type=str,
    default="tick-averages.csv",
    help="Specify the name (not the directory) the tickaverages file generated by the behavior model will get",
)
@click.option(
    "--sliding-window-size",
    type=int,
    default=7,
    help="Specify the size of the sliding window with which the mobility index will be smoothed when calculating the "
    "RMSE. Note that for the baseline, the actual size of the window used may be smaller, because some "
    "dates are missing",
)
@click.pass_context
def behavior(ctx, mobility_index_file, tick_averages_file, sliding_window_size):
    click.echo("Starting Behavior Calibration")

    gyration = Gyration(
        mobility_index_file,
        tick_averages_file,
        ctx.obj["counties"],
        sliding_window_size=sliding_window_size,
    )

    bc = BehaviorCalibration(
        gyration=gyration, tick_averages_file=tick_averages_file, **ctx.obj["args"]
    )

    initial_simplex = [
        [0.5, 0.5, 0.0125, 60],
        [0.7, 0.5, 0.0120, 40],
        [0.5, 0.7, 0.0130, 70],
        [0.3, 0.5, 0.0120, 60],
        [0.5, 0.3, 0.130, 50],
    ]

    calibrate(bc.calibrate, initial_simplex)


@start.command(
    name="optimization",
    help="Start the optimization process with a behavior and disease models",
)
@click.option(
    "--alpha",
    "-a",
    type=float,
    help="Specify the alpha value for the Bayesian optimization",
    default=1e-2
)
@click.option(
    "--init_points",
    type=int,
    default=20,
    help="The number of random samples the Bayesian optimization should try before iteratively improving"
)
@click.option(
    "--n_iter",
    type=int,
    default=160,
    help="The number of iterations the Bayesian optimization should perform after the initial random exploration before"
         "optimization ends"
)
@click.option(
    "--weight",
    "-w",
    type=float,
    help="Specify global societal impact weight (used to weigh impact of norms vs duration of norms)",
    default=1.0
)
@click.option(
    "--norm-weights",
    type=click.Path(exists=True),
    default=os.path.join(get_project_root(), "external", "norm_weights.csv"),
    help="Specify the file containing the weights for each norm (used to weigh the impact of a norm vs. the number "
         "of agents affected by that norm",
    required=True
)
@click.option(
    "--mode-liberal",
    "-ml",
    default=0.5,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.option(
    "--mode-conservative",
    "-mc",
    default=0.5,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.option(
    "--fatigue",
    type=float,
    help="The fatigue factor with which the agents' trust attitude will decrease each day",
)
@click.option(
    "--fatigue-start",
    type=int,
    help="The start time step for decreasing the agents' trust attitude with fatigue",
)
@click.option(
    "--log-location",
    type=click.Path(exists=False),
    help="Specify where the history (json format) will be logged",
    required=False
)
@click.option(
    "--n-slaves",
    type=int,
    default=0,
    help="Specify the number of instantiated slaves",
    required=False
)
@click.option(
    "--slave-number",
    type=int,
    default=0,
    help="Specify the unique number of this slave run. Must be between 0 and number of slaves specified to master",
    required=False
)
@click.pass_context
def optimization(
        ctx,
        alpha,
        init_points,
        n_iter,
        weight,
        norm_weights,
        mode_liberal,
        mode_conservative,
        fatigue,
        fatigue_start,
        log_location,
        n_slaves,
        slave_number
):
    click.echo("Starting policy optimization")
    EOOptimization(
        alpha=alpha,
        init_points=init_points,
        n_iter=n_iter,
        norm_weights=norm_weights,
        societal_global_impact_weight=weight,
        mode_liberal=mode_liberal,
        mode_conservative=mode_conservative,
        fatigue=fatigue,
        fatigue_start=fatigue_start,
        log_location=log_location,
        n_slaves=n_slaves,
        slave_number=slave_number,
        **ctx.obj['args']
    )


@start.command(
    name="disease",
    help="Start the disease model calibration with the liberal and conservative trust values fixed",
)
@click.option(
    "--mode-liberal",
    "-ml",
    default=0.5,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.option(
    "--mode-conservative",
    "-mc",
    default=0.5,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.option(
    "--fatigue",
    help="The fatigue factor with which the agents' trust attitude will decrease each day",
)
@click.option(
    "--fatigue-start",
    help="The start time step for decreasing the agents' trust attitude with fatigue",
)
@click.option(
    "--case-data-file",
    help="The file with actual case data",
    default="external/va-counties-estimated-covid19-cases.csv",
)
@click.option(
    "--epicurve-file",
    type=str,
    default="epicurve.sim2apl.csv",
    help="Specify the name (not the directory) the epicurve file generated by the behavior model will get",
)
@click.pass_context
def disease(
    ctx,
    mode_liberal,
    mode_conservative,
    fatigue,
    fatigue_start,
    case_data_file,
    epicurve_file,
):
    click.echo("Disease calibration started")
    args = ctx.obj["args"]
    args["mode_liberal"] = mode_liberal
    args["mode_conservative"] = mode_conservative
    args["fatigue"] = fatigue
    args["fatigue_start"] = fatigue_start
    args["epicurve_file"] = epicurve_file

    args["epicurve_rmse"] = EpicurveRMSE(
        ctx.obj["counties"], epicurve_file, case_data_file
    )
    dc = DiseaseCalibration(**args)

    initial_simplex = [
        [1, .5],
        [0.25, 0.075],
        [0.0625, 0.03125],
    ]

    calibrate(dc.calibrate, initial_simplex)


@start.command(
    name="disease_initial_guess_finder",
    help="Start the disease model calibration with the liberal and conservative trust values fixed",
)
@click.option(
    "--mode-liberal",
    "-ml",
    default=0.5,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.option(
    "--mode-conservative",
    "-mc",
    default=0.5,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.option(
    "--fatigue",
    help="The fatigue factor with which the agents' trust attitude will decrease each day",
)
@click.option(
    "--fatigue-start",
    help="The start time step for decreasing the agents' trust attitude with fatigue",
)
@click.option(
    "--case-data-file",
    help="The file with actual case data",
    default="external/va-counties-estimated-covid19-cases.csv",
)
@click.option(
    "--epicurve-file",
    type=str,
    default="epicurve.sim2apl.csv",
    help="Specify the name (not the directory) the epicurve file generated by the behavior model will get",
)
@click.pass_context
def disease_initial_guess_finder(
    ctx,
    mode_liberal,
    mode_conservative,
    fatigue,
    fatigue_start,
    case_data_file,
    epicurve_file,
):
    click.echo("Disease calibration started")
    args = ctx.obj["args"]
    args["mode_liberal"] = mode_liberal
    args["mode_conservative"] = mode_conservative
    args["fatigue"] = fatigue
    args["fatigue_start"] = fatigue_start
    args["epicurve_file"] = epicurve_file

    args["epicurve_rmse"] = EpicurveRMSE(
        ctx.obj["counties"], epicurve_file, case_data_file
    )
    dc = DiseaseCalibration(**args)
    dc.rundirectory_template = [
        "disease_initial_guess",
        "{ncounties}counties-fips-{fips}",
        "{isymp}isymp-{iasymp}iasymp-{liberal}l-{conservative}c-{fatigue}f-{fatigue_start}fs-run{run}",
    ]
    dc.progress_format = "[DISEASE INITIAL GUESS] [{time}] {ncounties} counties ({fips}): {score} for isymp {x[0]} and iasymp {x[1]}, disease calibration (dir={output_dir})\n"
    dc.csv_log = os.path.join("output", "calibration.disease_initial_guess_finder.csv")
    dc.rundirectory_template.insert(0, args["output_dir"])
    exists = os.path.exists(dc.csv_log)
    if not exists:
        with open(dc.csv_log, "a") as fout:
            fout.write(
                "fips,#counties,score,isymp,iasymp,time_finished,calibration_start_time\n"
            )

    symp = 1
    asymp = .5

    for i in range(50):
        dc.calibrate([symp / (2 ** i), asymp / (2 ** i)])

@start.command(
    name="simplerepeat",
    help="Run the experiment n_runs number of times with a fixed disease model and behavior parameters, "
    "and exit after finishing",
)
@click.option(
    "--mode-liberal",
    "-ml",
    default=0.2,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.option(
    "--mode-conservative",
    "-mc",
    default=0.2,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.option(
    "--fatigue",
    help="The fatigue factor with which the agents' trust attitude will decrease each day",
)
@click.option(
    "--fatigue-start",
    help="The start time step for decreasing the agents' trust attitude with fatigue",
)
@click.pass_context
def simplerepeat(ctx, mode_liberal, mode_conservative, fatigue, fatigue_start):
    click.echo("Running experiment {0} times with liberal={1},conservative={2}")
    args = ctx.obj["args"]
    args["mode_liberal"] = mode_liberal
    args["mode_conservative"] = mode_conservative
    args["fatigue"] = fatigue
    args["fatigue_start"] = fatigue_start

    re = RepeatedExecution(**args)
    re.calibrate(None)


@start.command(
    name="run-with-norm-schedules",
    help="Simple method to run with specific norm schedules few times"
)
@click.option(
    "--mode-liberal",
    "-ml",
    default=0.2,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.option(
    "--mode-conservative",
    "-mc",
    default=0.2,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.option(
    "--fatigue",
    help="The fatigue factor with which the agents' trust attitude will decrease each day",
)
@click.option(
    "--fatigue-start",
    help="The start time step for decreasing the agents' trust attitude with fatigue",
)
@click.argument(
    "norm_schedules",
    nargs=-1,
    type=click.Path(dir_okay=False, file_okay=True, exists=True)
)
@click.pass_context
def run_norm_schedules(ctx, mode_liberal, mode_conservative, fatigue, fatigue_start, norm_schedules):
    args = ctx.obj["args"]
    args["mode_liberal"] = mode_liberal
    args["mode_conservative"] = mode_conservative
    args["fatigue"] = fatigue
    args["fatigue_start"] = fatigue_start
    RunWithNormSchedules(norm_schedules, **args)


@start.command(
    name="sensitivity",
    help="Perform simulation with the most extreme parameters to test sensitivity to reasoning",
)
@click.option(
    "--mobility-index-file",
    "-m",
    type=click.Path(exists=True),
    help="Specify the location of the file containing the mobility index for each (relevant) county",
    default=os.path.join("external", "va_county_mobility_index.csv"),
)
@click.option(
    "--tick-averages-file",
    type=str,
    default="tick-averages.csv",
    help="Specify the name (not the directory) the tickaverages file generated by the behavior model will get",
)
@click.option(
    "--sliding-window-size",
    type=int,
    default=7,
    help="Specify the size of the sliding window with which the mobility index will be smoothed when calculating the "
    "RMSE. Note that for the baseline, the actual size of the window used may be smaller, because some "
    "dates are missing",
)
@click.pass_context
def sensitivity(ctx, mobility_index_file, tick_averages_file, sliding_window_size):
    click.echo("Starting sensitivity runs")

    gyration = Gyration(
        mobility_index_file,
        tick_averages_file,
        ctx.obj["counties"],
        sliding_window_size=sliding_window_size,
    )

    sa = SensitivityAnalysis(
        gyration=gyration, tick_averages_file=tick_averages_file, **ctx.obj["args"]
    )


@start.command(
    name="slave",
    help="Run a slave to another calibration process. At least one process with --is-master set to true should be provided,"
    "and exactly N slave processes should be started, with N being one less than the --number-of-runs specified to the"
    "master process. "
    "The --name and --output-dir MUST BE THE SAME on all these runs",
)
@click.option(
    "--run", type=int, required=True, help="The run number this node should execute"
)
@click.pass_context
def slave_calibration(ctx, run):
    args = ctx.obj["args"]
    SlaveCodeExecution(run=run, **args)


@start.command(
    name="scaling",
    help="Run a scaling test by running simulations with varying numbers of processor cores",
)
@click.option(
    "--cores",
    type=str,
    required=True,
    default="1,2,4,8,12",
    help="Integers, seperated by comma, no spaces",
)
@click.option(
    "--suppress-calculations",
    type=bool,
    required=False,
    default=False,
    help="Test without calculating radius of gyration or writing files",
)
@click.pass_context
def scaling_test(ctx, cores, suppress_calculations):
    args = ctx.obj["args"]
    se = ScalingTestExecution(suppress_calculations, **args)

    for x in [int(x) for x in cores.split(",")]:
        se.calibrate(x)


@start.command(name="experiment", help="Run the norm experiment master")
@click.option(
    "--mode-liberal",
    "-ml",
    default=0.5,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.option(
    "--mode-conservative",
    "-mc",
    default=0.5,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.option(
    "--fatigue",
    help="The fatigue factor with which the agents' trust attitude will decrease each day",
)
@click.option(
    "--fatigue-start",
    help="The start time step for decreasing the agents' trust attitude with fatigue",
)
@click.pass_context
def experiment(ctx, mode_liberal, mode_conservative, fatigue, fatigue_start):
    click.echo("Experiment started")
    args = ctx.obj["args"]
    args["mode_liberal"] = mode_liberal
    args["mode_conservative"] = mode_conservative
    args["fatigue"] = fatigue
    args["fatigue_start"] = fatigue_start

    experiment = NormExperiment(**args)
    experiment.initiate()


@start.command(
    name="remove-norms-experiment",
    help="Run the remove-norms-experiment master, which runs a passed norm schedule multiple times, "
         "with in each simulation, one of the norms or EOs removed"
)
@click.option(
    "--mode-liberal",
    "-ml",
    default=0.5,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.option(
    "--mode-conservative",
    "-mc",
    default=0.5,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.option(
    "--fatigue",
    help="The fatigue factor with which the agents' trust attitude will decrease each day",
)
@click.option(
    "--fatigue-start",
    help="The start time step for decreasing the agents' trust attitude with fatigue",
)
@click.option(
    "--by-eo/--no-by-eo",
    help="Should the effect of an individual norm be tested, or the effect of an entire EO?",
    default=False

)
@click.pass_context
def experiment(ctx, mode_liberal, mode_conservative, fatigue, fatigue_start, by_eo):
    click.echo("Experiment started")
    args = ctx.obj["args"]
    args["mode_liberal"] = mode_liberal
    args["mode_conservative"] = mode_conservative
    args["fatigue"] = fatigue
    args["fatigue_start"] = fatigue_start

    experiment = NormExperiment(**args)
    if by_eo:
        experiment.run_by_removing_EOs()
    else:
        experiment.run_by_removing_norms()


@start.command(name="test-trust", help="Run various simulations comparing the trust discount factor impact")
@click.option(
    "--mode-liberal",
    "-ml",
    default=0.5,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.option(
    "--mode-conservative",
    "-mc",
    default=0.5,
    help="Specify the mode of the government trust factor for the liberal voting agents",
)
@click.pass_context
def test_trust_discount(ctx, mode_liberal, mode_conservative):
    click.echo("Starting of testing trust discount factor")
    args = ctx.obj["args"]
    args["mode_liberal"] = mode_liberal
    args["mode_conservative"] = mode_conservative
    TestTrustDiscountFactor(**args)


@click.command(
    name="behavior_rmse",
    help="Run the RMSE for behavior (i.e. mobility) on a simulation output directory, or a directory containing"
    "multiple simulation output directories",
)
@click.option(
    "--simulation-output-dir",
    "-s",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True),
    help="Specify the output directory to analyse (may contain multiple simulation run outputs)",
    required=True,
)
@click.option(
    "--county-configuration",
    "-c",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, resolve_path=True),
    help="Specify the TOML file containing the county configuration for both the agent model and the calibration process",
    required=True,
)
@click.option(
    "--average-runs",
    "-a",
    type=bool,
    help="If this flag is set to true, the average score of multiple runs will be used. Otherwise, the RMSE will be"
    "calculated over all participating runs",
    default=False,
    required=False,
)
@click.option(
    "--mobility-index-file",
    "-m",
    type=click.Path(exists=True),
    help="Specify the location of the file containing the mobility index for each (relevant) county",
    default=os.path.join("external", "va_county_mobility_index.csv"),
)
@click.option(
    "--tick-averages-file",
    type=str,
    default="tick-averages.csv",
    help="Specify the name (not the directory) the tickaverages file generated by the behavior model will get",
)
@click.option(
    "--sliding-window-size",
    type=int,
    default=7,
    help="Specify the size of the sliding window with which the mobility index will be smoothed when calculating the "
    "RMSE. Note that for the baseline, the actual size of the window used may be smaller, because some "
    "dates are missing",
)
def behavior_rmse(
    simulation_output_dir,
    county_configuration,
    mobility_index_file,
    tick_averages_file,
    sliding_window_size,
    average_runs,
):
    MobilityRMSEOnBacklog(
        county_configuration,
        simulation_output_dir,
        mobility_index_file,
        tick_averages_file,
        sliding_window_size,
        average_runs,
    )


@click.command(
    name="disease_rmse",
    help="Run the RMSE for disease model (i.e. epicurve) on a simulation output directory, or a directory containing"
    "multiple simulation output directories",
)
@click.option(
    "--simulation-output-dir",
    "-s",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True),
    help="Specify the output directory to analyse (may contain multiple simulation run outputs)",
    required=True,
)
@click.option(
    "--county-configuration",
    "-c",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, resolve_path=True),
    help="Specify the TOML file containing the county configuration for both the agent model and the calibration process",
    required=True,
)
@click.option(
    "--average-runs",
    "-a",
    type=bool,
    help="If this flag is set to true, the average score of multiple runs will be used. Otherwise, the RMSE will be"
    "calculated over all participating runs",
    default=False,
    required=False,
)
@click.option(
    "--case-data-file",
    help="The file with actual case data",
    default="external/va-counties-estimated-covid19-cases.csv",
)
@click.option(
    "--default-epicurve-file-name",
    "-e",
    help="Specify the default name of the epicurve file that should be present in each directory",
    default="epicurve.sim2apl.csv",
)
def disease_rmse(
    simulation_output_dir,
    county_configuration,
    average_runs,
    case_data_file,
    default_epicurve_file_name,
):
    DiseaseRMSEOnBacklog(
        county_configuration,
        simulation_output_dir,
        default_epicurve_file_name,
        case_data_file,
        average_runs,
    )


def calibrate(fitness_function, initial_simplex):
    options = {"xatol": 0.1, "disp": True, "initial_simplex": initial_simplex}
    minimize(
        fitness_function, x0=initial_simplex[0], method="nelder-mead", options=options
    )


if __name__ == "__main__":
    start()
