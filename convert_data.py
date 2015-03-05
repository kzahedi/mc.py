#!/usr/bin/env python

import argparse
import btk
import numpy
import os
import functions

parser = argparse.ArgumentParser(description="Arguments:")

parser.add_argument("-i", type=str, default=None, help="Input file.")
parser.add_argument("-o", type=str, default=None, help="Output file.")

args = parser.parse_args()

a    = get_positions(args.i)

for b in a:
  print b
