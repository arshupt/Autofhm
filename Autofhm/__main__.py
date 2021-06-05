import argparse
from Autofhm import Autofhm
import time
from datetime import datetime
from pathlib import Path
import sys
import os
import warnings
from rich import print

parser = argparse.ArgumentParser(description='An Automated Machine Learning tool. Models can be saved as sav file.')
parser.add_argument('-c', '--config', type=str, nargs=1,
                    help='path to the configruation file')
parser.add_argument('-n', '--name', type=str, nargs=1,
                    help='name of model to be saved')
args = parser.parse_args()
if args.config:
    if os.path.exists(args.config[0]):
        _temp = Autofhm(config=args.config[0])
        _temp.get_features()
        _temp.fit()
        if args.name:
            _temp.save_model(args.name[0],os.getcwd())
        else:    
            c = input("Save model?[Y/n]") or "y"
            if c in "yY":
                name = input("Enter name:")
                _temp.save_model(name,os.getcwd())
