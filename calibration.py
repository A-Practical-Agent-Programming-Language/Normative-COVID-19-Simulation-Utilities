#!/bin/python3
import os
import subprocess

import click
import toml

from scipy.optimize import minimize

from classes.execution.BehaviorCalibration import BehaviorCalibration
from classes.execution.DiseaseCalibration import DiseaseCalibration
from classes.Gyration import Gyration


# TODO
#  - Move all generated files not directly relevant for user to another subdir, to avoid clutter
from classes.execution.RepeatedExecution import RepeatedExecution


@click.group()
@click.option('--county-configuration', '-c', type=click.Path(exists=True, dir_okay=True, file_okay=True, resolve_path=True),
              help='Specify the TOML file containing the county configuration for both the agent model and the calibration process',
              required=True)
@click.option('--behavior-model', '-b', type=click.Path(exists=True, dir_okay=True, file_okay=True, resolve_path=True),
              help="Specify the location of the behavior model .JAR file", required=True)
@click.option('--pansim-num-cpus', type=int, default=1, help="Specify the number of CPUs (threads) to use with PanSim")
@click.option('--java', '-j', type=str, default="java", help="Specify the Java binary to use for code execution")
@click.option('--java-num-cpus', type=int, default=1, help="Specify the number of CPUs (threads) to use with Java")
@click.option('-Xmx', type=str, default="64g",
              help="Specify the maximum heap size following Java argument formatting (see Java man pages for help, Java argument is -Xmx")
@click.option('-Xms', type=str, default="55g",
              help="Specify the initial heap size following Java argument formatting (see Java man pages for help, Java argument is -Xms")
@click.option('--output-dir', '-o', type=click.Path(dir_okay=True, file_okay=True, resolve_path=True), default="output/agent-runs",
              help="Specify an alternative directory where the behavior model output files will be stored")
@click.option('--number-of-runs', '-r', type=int, default=10,
              help="How many times should the simulation be run for each configuration")
@click.option("--epicurve-file", type=str, default="epicurve.csv",
              help="Specify the name (not the directory) the epicurve file should get")
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
	county_configuration = load_toml_configuration(kwargs["county_configuration"])
	conf_file_tmp = os.path.join(".persistent", ".tmp", "java_model_config.toml")
	if not os.path.exists(os.path.dirname(conf_file_tmp)):
		os.makedirs(os.path.dirname(conf_file_tmp))
	with open(conf_file_tmp, 'w') as fout:
		toml.dump(county_configuration, fout)

	counties = county_configuration["counties"]
	disease_model = county_configuration["simulation"]["diseasemodel"]

	ctx.obj = dict(
		counties=counties,
		args=dict(
			county_configuration_file=conf_file_tmp,
			sim2apl_jar=kwargs["behavior_model"],
			counties=counties,
			disease_model_file=disease_model,
			n_cpus=kwargs["pansim_num_cpus"],
			java_home=kwargs["java"],
			java_threads=kwargs["java_num_cpus"],
			java_heap_size_max=kwargs["xmx"],
			java_heap_size_initial=kwargs["xms"],
			output_dir=kwargs["output_dir"],
			n_runs=kwargs["number_of_runs"],
			epicurve_file=kwargs["epicurve_file"]
		)
	)

def print_dict(dct, ident=0):
	for k,v in dct.items():
		if type(v) == type(dct):
			print("\t"*ident, k)
			print_dict(v, ident + 1)
		else:
			print("\t"*ident, k, v)


@start.command(name="behavior", help="Start the behavior model calibration with the disease model fixed")
@click.option('--mobility-index-file', '-m', type=click.Path(exists=True),
              help="Specify the location of the file containing the mobility index for each (relevant) county",
              default=os.path.join("external", "va_county_mobility_index.csv"))
@click.option("--tick-averages-file", type=str, default="tick-averages.csv",
              help="Specify the name (not the directory) the tickaverages file generated by the behavior model will get")
@click.option(
	"--sliding-window-size",
	type=int,
	default=7,
	help="Specify the size of the sliding window with which the mobility index will be smoothed when calculating the "
	     "RMSE. Note that for the baseline, the actual size of the window used may be smaller, because some "
	     "dates are missing"
)
@click.pass_context
def behavior(ctx, mobility_index_file, tick_averages_file, sliding_window_size):
	click.echo("Starting Behavior Calibration")
	gyration = Gyration(
		mobility_index_file,
		tick_averages_file,
		ctx.obj["counties"],
		sliding_window_size=sliding_window_size
	)

	bc = BehaviorCalibration(
		gyration=gyration,
		tick_averages_file=tick_averages_file,
		**ctx.obj["args"]
	)

	initial_simplex = [
		[0.5, 0.5],
		[0.5, 0.75],
		[0.75, 0.5]
	]
	calibrate(bc.calibrate, initial_simplex)


@start.command(name="disease",
               help="Start the disease model calibration with the liberal and conservative trust values fixed")
@click.option('--mode-liberal', '-ml', default=0.2,
              help="Specify the mode of the government trust factor for the liberal voting agents")
@click.option('--mode-conservative', '-mc', default=0.2,
              help="Specify the mode of the government trust factor for the liberal voting agents")
@click.pass_context
def disease(ctx, mode_liberal, mode_conservative):
	click.echo("Disease calibration started")
	args = ctx.obj["args"]
	args["mode_liberal"] = mode_liberal
	args["mode_conservative"] = mode_conservative

	dc = DiseaseCalibration(**args)

	initial_simplex = [
		[1/100, 0.05, 0.010],
		[1/200, 0.05, 0.015],
		[1/70, 0.05, 0.05],
		[1 / 100, 0.05, 0.05],
	]

	calibrate(dc.calibrate, initial_simplex)


@start.command(name="simplerepeat", help="Run the experiment n_runs number of times with a fixed disease model and behavior parameters, and exit after finishing")
@click.option('--mode-liberal', '-ml', default=0.2,
              help="Specify the mode of the government trust factor for the liberal voting agents")
@click.option('--mode-conservative', '-mc', default=0.2,
              help="Specify the mode of the government trust factor for the liberal voting agents")
@click.pass_context
def simplerepeat(ctx, mode_liberal, mode_conservative):
	click.echo("Running experiment {0} times with liberal={1},conservative={2}")
	args = ctx.obj["args"]
	args["mode_liberal"] = mode_liberal
	args["mode_conservative"] = mode_conservative

	re = RepeatedExecution(**args)

	calibrate(re.calibrate(None))

def calibrate(fitness_function, initial_simplex):
	options = {'xatol': 0.1, 'disp': True, "initial_simplex": initial_simplex}
	minimize(fitness_function, x0=initial_simplex[0], method="nelder-mead", options=options)


def make_file_absolute(county_config_location, file):
	if os.path.isabs(file):
		return file, os.path.exists(file)
	else:
		resolved_path = os.path.abspath(os.path.join(county_config_location, file))
		return resolved_path, os.path.exists(resolved_path)


def load_toml_configuration(county_config_file):
	missing = list()
	conf = toml.load(county_config_file)
	county_config_location = os.path.dirname(county_config_file)
	norms, norms_exist = make_file_absolute(county_config_location, conf["simulation"]["norms"])
	if not norms_exist:
		missing.append(("simulation.norms", [conf["simulation"]["norms"]]))
	conf["simulation"]["norms"] = norms

	disease, disease_exists = make_file_absolute(county_config_location, conf["simulation"]["diseasemodel"])
	if not disease_exists:
		missing.append(("simulation.disease", [conf["simulation"]["diseasemodel"]]))
	conf["simulation"]["diseasemodel"] = disease

	for county, cconf in conf["counties"].items():
		for k in ["activities", "households", "persons", "locations", "statefile"]:
			if not k in cconf:
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

	if len(missing):
		error = f"Some paths specified in {county_config_location} could not be resolved:\n\n"
		for (m, m_) in missing:
			error += m + ":\n"
			for f in m_:
				error += "\t" + f + "\n"

		raise click.exceptions.BadParameter(error, param_hint="county-configuration")

	return conf


def test_code_available(java_location):
	try:
		result = subprocess.run([java_location, "--version"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
		print("Using", str(result.stdout.splitlines()[0], 'utf-8'))
	except FileNotFoundError as e:
		raise click.exceptions.BadParameter(
			f"The provided Java binary {java_location} could not be used to start a Java VM", param_hint="--java")

	result = subprocess.run(["pansim", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
	if result.returncode == 0:
		print("Pansim is available")
	else:
		raise click.exceptions.BadArgumentUsage(
			f"PanSim was not found. Please install PanSim before attempting calibration")


if __name__ == "__main__":
	start()
