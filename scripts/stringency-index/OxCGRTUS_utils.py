import json
import os
import subprocess
import sys

import pandas as pd
import requests
import xlrd


def assert_excel_compatible_or_convert_to_ods(filename_base):
    """Sometimes, Pandas fails to read Excel files. In that case we will have to convert to ODS, but we can
    first try and see if the Excel reading is supported

    Returns the filename (filename base with .ods or .xlsx extension)
    """
    filename = filename_base + ".xlsx"
    save_excel_file_from_github(filename)
    try:
        pd.read_excel(filename, sheet_name=0)
    except xlrd.biffh.XLRDError:
        filename = f'{filename_base}.ods'
        while not os.path.exists(filename):
            print("Trying to convert Excel file to OpenOffice format")
            try:
                try_convert_to_ods(filename_base + ".xlsx")
            except Exception as e:
                sys.stderr.write(
                    f"{filename_base}.xlsx needs to be converted to ODS, but this could not be done "
                    f"automatically because of the following error\n")
                sys.stderr.flush()
                print(e)
                input(f"Please convert {filename_base}.xlsx manually to {filename_base}.ods, and press enter when done")

    return filename


def save_excel_file_from_github(filename):
    """
    Pulls the Excel workbook containing all the data for the states of Virginia from Github, and saves
    the file on the current file system as an Excel document.
    """
    if not os.path.exists(filename) or not os.path.isfile(filename):
        print(f"{filename} not yet present. Downloading")
        r = requests.get(f'https://github.com/OxCGRT/USA-covid-policy/raw/master/data/{filename}')
        open(filename, 'wb').write(r.content)


def try_convert_to_ods(filename):
    """
    Only supported on POSIX systems. Tries to convert an XLSX document to ODS using LibreOffice
    """
    if not os.name == 'posix':
        raise IOError('Automatically converting to Open Office file format not supported')
    try:
        libreoffice = subprocess.check_output(['which', 'libreoffice']).decode("utf-8").replace('\n', '')
    except subprocess.CalledProcessError:
        raise Exception("LibreOffice not found")
    result = subprocess.run([
        libreoffice, '--headless', '--invisible', '--convert-to', 'ods',
        filename
    ])
    result.check_returncode()


def store_as_json(values, filename_base):
    """Dumps the values dictionary to <filename>.json"""
    with open(f'{filename_base}.json', 'w') as json_out:
        json.dump(values, json_out, indent=4)
    print(f"Stored extracted values in {filename_base}.json for faster access on next run")
