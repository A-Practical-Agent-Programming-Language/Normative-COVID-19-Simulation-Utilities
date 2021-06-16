import os
import subprocess

import click
import toml

from classes.EssentialLocations import EssentialDesignationExtractor


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
		for k in ["activities", "households", "persons", "locations", "statefile", "locationDesignations"]:
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

		conf["counties"][county] = ensure_location_designation_present(conf["counties"][county])

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
		county["locationDesignations"] = [EssentialDesignationExtractor().from_county(county)]
	return county


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