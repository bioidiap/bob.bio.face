#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""
This script parses through the given directory, collects all results of 
experiments using bob.db.ijba and dumps a nice copy/paste ReStructuredText (RST).

It supports comparison (--report-type comparison) and search  (--report-type search) experiments

For comparison, the following command dumps:

./bin/bob_ijba_collect_results.py <comparison-path> -r comparison


+-----------------+-----------------+-----------------+-----------------+--------------------------+
|    RR           | TPIR% (FAR=0.1) | TPIR% (FAR=0.01)|TPIR% (FAR=0.001)| split                    |
+=================+=================+=================+=================+==========================+
|   00000         |0000000          |00000000         |00000            |split 0                   |
+-----------------+-----------------+-----------------+-----------------+--------------------------+
|                                                                                                  |

For search, the following command dumps:

./bin/bob_ijba_collect_results.py <search-path> -r search

+-------------------------+-------------------------+--------------------------+
| DIR% (rank=1, FAR=0.1)  | DIR% (rank=1, FAR=0.01) | split                    |
+=========================+=========================+==========================+
|   00000                 |000000                   |00000000                  |


"""


from __future__ import print_function
import sys, os,  glob
import argparse
import numpy

import bob.measure
import bob.core
from argparse import RawTextHelpFormatter
from bob.bio.base.script.collect_results import Result, recurse, add_results
far_thresholds = [0.1, 0.01, 0.001]
logger = bob.core.log.setup("bob.bio.base")


def command_line_arguments(command_line_parameters):
  """Parse the program options"""

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=RawTextHelpFormatter)

  parser.add_argument('directory', default=".", help = "The directory where the results should be collected from.")
  parser.add_argument('-r', '--report-type', default="comparison", choices=("comparison", "search"), help = "Type of the report. For `comparison`, RR and TPIR (FAR=[0.1, 0.01 and 0.001] ) will be reported. For the search DIR (rank=1, under FAR=[0.1, 0.01]) is reported")

  bob.core.log.add_command_line_option(parser)

  # parse arguments
  args = parser.parse_args(command_line_parameters)

  bob.core.log.set_verbosity_level(logger, args.verbose)

  return args


def search_results(args, directories):
  """
  Navigates throught the directories collection evaluation results
  """
  
  results = []
  for directory in directories:
    r = recurse(args, directory)
    if r is not None:
      results += r

  return results
  

def compute_mean_std(results):
    return numpy.mean([r.nonorm_dev for r in results]), numpy.std([r.nonorm_dev for r in results])

    
def compute_comparison(args, directories):
  """
  Plot evaluation table for the comparison protocol
  """

  def plot_comparison_table(cmc_r1, fnmr):
  
    grid =  "+-----------------+-----------------+-----------------+-----------------+--------------------------+\n"
    grid += "|        RR       | TPIR% (FAR=0.1) | TPIR% (FAR=0.01)|TPIR% (FAR=0.001)| split                    |\n"
    grid += "+=================+=================+=================+=================+==========================+\n"
    
    for cmc, fnmr_0, fnmr_1, fnmr_2, split in zip(cmc_r1, fnmr[0], fnmr[1], fnmr[2], range(len(cmc_r1))):
      grid += "|{:17s}|{:17s}|{:17s}|{:17s}|{:26s}|\n".format(str(round(cmc.nonorm_dev,5)*100),
                                                             str(round(fnmr_0.nonorm_dev,5)*100),
                                                             str(round(fnmr_1.nonorm_dev,5)*100),
                                                             str(round(fnmr_2.nonorm_dev,5)*100),
                                                             "split {0}".format(split))
      grid +=  "+-----------------+-----------------+-----------------+-----------------+--------------------------+\n"

    cmc = compute_mean_std(cmc_r1)
    fnmr_0 = compute_mean_std(fnmr[0])
    fnmr_1 = compute_mean_std(fnmr[1])
    fnmr_2 = compute_mean_std(fnmr[2])   
    grid += "|**{:6s}({:5s})**|**{:6s}({:5s})**|**{:6s}({:5s})**|**{:6s}({:5s})**|{:26s}|\n".format(
                                                           str(round(cmc[0],4)*100),str(round(cmc[1],4)*100),
                                                           str(round(fnmr_0[0],4)*100),str(round(fnmr_0[1],4)*100),
                                                           str(round(fnmr_1[0],4)*100),str(round(fnmr_1[1],4)*100),
                                                           str(round(fnmr_2[0],4)*100),str(round(fnmr_2[1],4)*100),
                                                           "mean(std)")
    grid +=  "+-----------------+-----------------+-----------------+-----------------+--------------------------+\n"    
                  
    return grid

  def compute_fnmr(args, directories):
    fnmr = []  
    for f in far_thresholds:
      args.criterion = "FAR"
      args.far_threshold = f
      
      # Computing TPIR
      frr = search_results(args, directories)
      for rr in frr:
        rr.nonorm_dev = 1.-rr.nonorm_dev
      fnmr.append(frr)

    return fnmr


  args = args
  args.rank = 1
  args.criterion = "RR"
  cmc_r1 = search_results(args, directories)
  fnmr = compute_fnmr(args, directories)
  return plot_comparison_table(cmc_r1, fnmr)


def compute_search(args, directories):
  """
  Plot evaluation table for the search protocol
  """

  def plot_search_table(dira):
    grid =  "+----------------------+----------------------+--------------------------+\n"
    grid += "| DIR% (R=1, FAR=0.1)  | DIR% (R=1, FAR=0.01) | split                    |\n"
    grid += "+======================+======================+==========================+\n"
  
    n_splits = len(dira[0])
    for dira_0, dira_1, split in zip(dira[0], dira[1], range(n_splits)):
      grid += "|{:22s}|{:22s}|{:26s}|\n".format(str(round(dira_0.nonorm_dev,5)*100),
                                                str(round(dira_1.nonorm_dev,5)*100),
                                                "split {0}".format(split))
      grid += "+----------------------+----------------------+--------------------------+\n"

    dira_0 = compute_mean_std(dira[0])
    dira_1 = compute_mean_std(dira[1])
    grid += "|**{:6s}({:6s})**    |**{:6s}({:6s})**    |{:26s}|\n".format(
                                                                      str(round(dira_0[0],4)*100),str(round(dira_0[1],4)*100),
                                                                      str(round(dira_1[0],4)*100),str(round(dira_1[1],4)*100),
                                                                      "mean(std)")
    grid += "+----------------------+----------------------+--------------------------+\n"

    return grid

  def compute_dir(args, directories):
    dira = []  
    for f in far_thresholds[0:2]:
      args.criterion = "DIR"
      args.far_threshold = f      
      dira.append(search_results(args, directories))

    return dira


  args = args
  args.rank = 1
  args.criterion = "DIR"
  dira = compute_dir(args, directories)
  return plot_search_table(dira)


def main(command_line_parameters = None):
  """Iterates through the desired directory and collects all result files."""
  args = command_line_arguments(command_line_parameters)

  # collect results
  if not os.path.exists(args.directory):
    raise ValueError("The directory `%s` does not exists"%args.directory)
  
  # Injecting some variables
  args.dev = "scores-dev"
  args.eval = "scores-eval"  
  args.nonorm = "nonorm"
  args.ztnorm = "ztnorm"
    
  if args.report_type == "comparison":
    print(compute_comparison(args, [args.directory]))
  else:
    print(compute_search(args, [args.directory]))

