"""
This script utalises task spooler (downloadable from https://vicerveza.homeunix.net/~viric/soft/ts/) to
manage scheduling the running of multiple hyperparameter and model experiments set out in a csv file.

requirements:
    - task spooler
"""

import csv
import os
import subprocess
import argparse
import sys

def run_bash(cmd):
    """
    Method to run a bash command
    derived from (https://stackoverflow.com/questions/42426960/how-does-one-train-multiple-models-in-a-single-script-in-tensorflow-when-there-a)
    :param cmd: bash command
    :return: stdout from shell
    """
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, executable='/bin/bash')
    out = p.stdout.read().strip()
    return out  # This is the stdout from the shell command

def string_to_bool(string):
    if string.lower() == 'false':
        return False
    elif string.lower() == 'true':
        return True
    else:
        raise ValueError
# ==============================================================================================================================================
# Set up arguments
parser = argparse.ArgumentParser(description='Script that utalises task spooler to run multiple hyperparameter experiments')
parser.add_argument('python_file', type=str, help='The version of wnut_pytorchtransformers.py to run')
parser.add_argument('config_csv', type=str, help='CSV File of hyperparameters')
args = parser.parse_args()

if not os.path.isfile(args.python_file):
    sys.stderr.write('Hyperparamter file not found' + args.config + ' not found\n')
    raise SystemExit

if not os.path.isfile(args.config_csv):
    sys.stderr.write('Hyperparamter file not found' + args.config + ' not found\n')
    raise SystemExit


# ==============================================================================================================================================
with open(args.config_csv, mode='r') as csv_in:
    reader = csv.reader(csv_in)

    for row in reader:

        dict = {
            "model_class"  : row[0],
            "model"        : row[1],
            "random_seed"  : int(row[2]),
            "max_len"      : int(row[3]),
            "epochs"       : int(row[4]),
            "learning_rate": float(row[5]),
            "batch_size"   : int(row[6]),
            "dropout_prob" : float(row[7]),
            "test_size"    : float(row[8]),
            "preprocessed" : string_to_bool(row[9]),
            "ensemble"     : string_to_bool(row[10]),
            "num_labels"   : int(row[11]),
            "max_feats"    : int(row[12]),
            "save_model"   : string_to_bool(row[13]),
            "split_train"  : string_to_bool(row[14])
        }

        job_cmd = "python " + args.python_file + " '{model_class}' '{model}' '{random_seed}'"\
		    "  '{max_len}' '{epochs}' '{learning_rate}' '{batch_size}' '{dropout_prob}' '{test_size}'"\
		     " '{preprocessed}' '{ensemble}' '{num_labels}' '{max_feats}' '{save_model}' '{split_train}'".format(model_class=dict['model_class'],model=dict['model'],random_seed=dict['random_seed'],max_len=dict['max_len'], epochs=dict['epochs'], learning_rate=dict['learning_rate'],batch_size=dict['batch_size'],dropout_prob=dict['dropout_prob'],test_size=dict['test_size'],preprocessed=dict['preprocessed'],ensemble=dict['ensemble'], num_labels=dict['num_labels'],max_feats=dict['max_feats'], save_model=dict['save_model'], split_train=dict['split_train'] )

        submit_cmd = "ts %s" %job_cmd

        run_bash(submit_cmd)