import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("fname", metavar="FNAME", type=str, help="Input file name")
parser.add_argument("-o", "--output", metavar="F", type=str, default=None, action="store", help="Output filename")

args = parser.parse_args(sys.argv[1:])
fname = args.fname
out_name = args.output
if out_name is None:
    out_name = "{}.npz".format(fname)

_ = np.savez_compressed(out_name, np.loadtxt(fname))
