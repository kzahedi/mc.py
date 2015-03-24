#!/usr/bin/env python

import argparse
import numpy
import math

from functions import *

parser = argparse.ArgumentParser(description="Arguments:")

parser.add_argument("-i", type=str, default=None, help="Input file.")
parser.add_argument("-o", type=str, default=None, help="Output file.")

args = parser.parse_args()

print "Reading data."
data = get_positions(args.i)
print "done."

print "Extracting domains."
domains = get_domain_for_each_marker(data)
print "done."

print "Scaled data."
scaled_data = scale_data_for_each_marker(data, domains)
print "done."

print "Bin data."
binned_data = bin_scaled_data_for_each_marker(scaled_data, 10)
print "done."

print "Bin data."
combined_binned_data = combine_bins_for_each_marker(binned_data, 10)
print "done."

print combined_binned_data.keys()

print max(combined_binned_data["2 PIP"])
print max(combined_binned_data["2 DIP"])
print max(combined_binned_data["3 MCP"])

print "Calculating joint distribution"
jd = emperical_joint_distribution(combined_binned_data["2 PIP"],
    combined_binned_data["2 DIP"], combined_binned_data["3 MCP"])
print "done."

print calculate_concept_one(jd)
print calculate_concept_two(jd)
