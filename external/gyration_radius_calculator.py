"""This script calculates radius of gyration for each ID"""
#Author: Samarth Swarup
#Date: Mar 23, 2020.

import argparse
import logging
from datetime import datetime
import time
import csv
import os
import glob
from math import radians, cos, sin, asin, sqrt
import numpy as np

starttime = datetime.now()

logger = logging.getLogger(__name__)

# Parse arguments
parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-i', '--inputdir', help='must specify an input directory', required=True)
requiredNamed.add_argument('-o', '--outputdir', help='must specify an output directory', required=True)

parser.add_argument('-v', '--verbose', help='print extra information while executing', action="store_true")
parser.add_argument('-g', '--debug', help='print debugging information', action="store_true")

args = parser.parse_args()
inputdir = args.inputdir
outputdir = args.outputdir

if (not inputdir.endswith('/')):
    inputdir += '/'
    
if (not outputdir.endswith('/')):
    outputdir += '/'
    
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

logfile = 'gyration_radius_calculator_' + os.path.basename(os.path.dirname(inputdir)) + '.log'
#Specifying both -d and -v is equivalent to just specifying -d
if args.debug:
    logging.basicConfig(filename=logfile, filemode='w', format='%(message)s', level=logging.DEBUG)
elif args.verbose:
    logging.basicConfig(filename=logfile, filemode='w', format='%(message)s', level=logging.INFO)
else:
    logging.basicConfig(filename=logfile, filemode='w', format='%(message)s', level=logging.WARNING)

logger.info('Input dir: %s', inputdir)
logger.info('Output dir: %s', outputdir)
logger.info('Log file: %s', logfile)

#This function from: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6372.8 # Radius of the earth in km. Use 3959.87433 for miles.
    return c * r

filelist = glob.glob(inputdir + "*.csv")

numZeroes = 0 #Number of records with (0,0) for lat, lon
for filename in filelist:
    logger.info("Now working on " + filename)
    basename = os.path.basename(filename)
    outputfilename = outputdir + basename.split('.')[0] + "_gyration_radii.csv"
    
    # Open the file with located activities
    csvfile = open(filename, 'r')
    inputreader = csv.reader(csvfile, delimiter='\t')
    
    # Open the outputfile for writing the gyration radii
    opw = open(outputfilename, "w")
    opw.write("ID,CentroidLat,CentroidLon,NumPoints,GyrationRadiusKm\n")
    
    currentID = ''
    lats = []
    lons = []
    for row in inputreader:
        if (float(row[3]) == 0.0 and float(row[4]) == 0.0): #Some lats and lons are set to 0.0; we are skipping those
            numZeroes += 1
            continue
        if (row[1] != currentID):
            if (len(lats) > 0):
                #Calculate the centroid of the locations for the currentID
                clat = np.mean(np.array(lats))
                clon = np.mean(np.array(lons))
                sqsum=0.0
                #Calculate the Haversine distance for each location from the centroid and sum it.
                for i in range(len(lats)):
                    dist = haversine(clon, clat, lons[i], lats[i])
                    sqsum += dist*dist
                sqsum /= len(lats)
                #Calculate the radius of gyration as the squareroot of the sum
                radius = sqrt(sqsum)
                opw.write(currentID + "," + str(clat) + "," + str(clon) + "," + str(len(lats)) + "," + str(radius) + "\n")
            currentID = row[1]
            lats = [float(row[3])]
            lons = [float(row[4])]
        else:
            lats.append(float(row[3]))
            lons.append(float(row[4]))
    opw.close()
    csvfile.close()
            

logger.info("Number of records with (0,0) for lat, lon: " + str(numZeroes))

endtime = datetime.now()
logger.info("Start time: " + str(starttime))
logger.info("End time: " + str(endtime))
logger.info("Run time: " + str(endtime - starttime))
    
    
    
    
    
    
    
    
    
    
    
    