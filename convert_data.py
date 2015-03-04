#!/usr/local/bin/python

import argparse
import btk

parser = argparse.ArgumentParser(description="Arguments:")

parser.add_argument("-i", type=str, default=None, help="Input file.")
parser.add_argument("-o", type=str, default=None, help="Output file.")

args = parser.parse_args()


