#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
This script runs some face recognition baselines under some face databases

Run `bob bio face baselines --help` to list all the baselines and databases available

Examples:

    To run the LDA baseline in the mobio-male database do:
       bob bio face baselines --database mobio-male --baseline lda
    
    To run the GMM baseline in the mobio-male database do:
       bob bio face baselines --database mobio-male --baseline gmm
 
"""

from bob.extension.scripts.click_helper import verbosity_option
from bob.extension.scripts.click_helper import (
            verbosity_option, ConfigCommand, ResourceOption)
from click_plugins import with_plugins
import click
import os
import pkg_resources
import bob.bio.base
from bob.bio.base.script.verify import main as verify

def get_available_databases():
    """
    Get all the available databases through the database entry-points
    """
    
    available_databases = dict()
    all_databases = bob.bio.base.resource_keys('database')
    for database in all_databases:        
        try:               
            database_entry_point = bob.bio.base.load_resource(database, 'database')

            available_databases[database] = dict()

            # Checking if the database has data for the ZT normalization
            available_databases[database]["has_zt"] = hasattr(database_entry_point, "zobjects") and hasattr(database_entry_point, "tobjects")
            available_databases[database]["groups"] = []
            # Searching for database groups
            try:
                groups = list(database_entry_point.groups())
                for g in ["dev", "eval"]:
                    available_databases[database]["groups"] += [g] if g in groups else []
            except:
                # In case the method groups is not implemented
                available_databases[database]["groups"] = ["dev"]
        except:
            pass
    return available_databases

# LIST OF REGISTERED BASELINES
BASELINES = {
        'eigenface': dict(
                  preprocessor = 'face-crop-eyes',
                  extractor    = 'linearize',
                  algorithm    = 'pca',
                ),

        'lda': dict(
                  preprocessor = 'face-crop-eyes',
                  extractor    = 'eigenface',
                  algorithm    = 'lda',                  
                ),
        'plda': dict(
                  preprocessor = 'face-crop-eyes',
                  extractor    = 'linearize',
                  algorithm    = 'pca+plda',
                  grid         = 'demanding',
                ),
        'gabor-graph': dict(
                  preprocessor = 'inorm-lbp-crop',
                  extractor    = 'grid-graph',
                  algorithm    = 'gabor-jet',
                ),
        'lgbphs': dict(
                  preprocessor = 'tan-triggs-crop',
                  extractor    = 'lgbphs',
                  algorithm    = 'histogram',
                ),
        'gmm': dict(
                  preprocessor = 'tan-triggs-crop',
                  extractor    = 'dct-blocks',
                  algorithm    = 'gmm',
                  grid         = 'demanding',
                ),
        'isv': dict(
                  preprocessor = 'tan-triggs-crop',
                  extractor    = 'dct-blocks',
                  algorithm    = 'isv',
                  grid         = 'demanding',
                ),
        'ivector': dict(
                  preprocessor = 'tan-triggs-crop',
                  extractor    = 'dct-blocks',
                  algorithm    = 'ivector',
                  grid         = 'demanding',
                ),
        }


DATABASES = [d for d in get_available_databases().keys()]

@with_plugins(pkg_resources.iter_entry_points('bob.bio.cli.bio'))
@click.group()
def face():
    """Entry point to run Face Recognition Algorithms
    
    Check it out https://www.idiap.ch/software/bob/docs/bob/bob.bio.face/stable/index.html 

    """
    pass


@face.command(entry_point_group='bob.bio.config', cls=ConfigCommand)
@click.option('--baseline', '-b', required=True, cls=ResourceOption, help="Registered baseline", type=click.Choice(BASELINES))
@click.option('--database', '-d', required=True, cls=ResourceOption, help="Registered database", type=click.Choice(DATABASES))
@click.option('--temp-directory', '-T', required=False, cls=ResourceOption, help="The directory to write temporary the data of the experiment into. If not specified, the default directory of the verify.py script is used (see verify.py --help).")
@click.option('--result-directory', '-R', required=False, cls=ResourceOption, help="The directory to write the resulting score files of the experiment into. If not specified, the default directories of the verify.py script are used (see verify.py --help).")
@click.option('--grid', '-g', help="Execute the algorithm in the SGE grid.", is_flag=True)
@click.option('--zt-norm', '-z', help="Enable the computation of ZT norms (if the database supports it).", is_flag=True)
def baselines(baseline, database, temp_directory, result_directory, grid, zt_norm, **kwargs):
    """
     This script runs some face recognition baselines under some face databases

     Run `bob bio face baselines --help` to list all the baselines and databases available

     Examples:

       To run the LDA baseline in the mobio-male database do:
           bob bio face baselines --database mobio-male --baseline lda

       To run the GMM baseline in the mobio-male database do:
           bob bio face baselines --database mobio-male --baseline gmm  
    """

    # this is the default sub-directory that is used
    sub_directory = os.path.join(database, baseline)
    database_data = get_available_databases()[database]
    parameters = [
        '-p', BASELINES[baseline]["preprocessor"],
        '-e', BASELINES[baseline]["extractor"],
        '-d', database,
        '-a', BASELINES[baseline]["algorithm"],
        '-vvv',
        '--temp-directory', temp_directory,
        '--result-directory', result_directory,
        '--sub-directory', sub_directory
    ]

    parameters += ['--groups'] + database_data["groups"]

    if grid:
        if 'grid' in BASELINES[baseline].keys():
            parameters += ['-g', BASELINES[baseline]["grid"]]
        else:
            parameters += ['-g', 'grid']

    if zt_norm and 'has_zt' in database_data.keys():
        parameters += ['--zt-norm']

    verify(parameters)
