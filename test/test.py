from Autofhm import Autofhm
import time
from datetime import datetime
from pathlib import Path
import sys
import warnings
from rich import print

to_test = ["admission_predict", "Boston", "fish_species", "fish_weight",
           "insurance", "iris", "realestate", "Sonar", "thyroid", "titanic"]

curr_directory = Path(__file__).resolve().parent
if len(sys.argv) == 3:
    if "-c" in sys.argv[1]:
        print(f"Only testing: {sys.argv[2]}")
        to_test = [sys.argv[2]]

print("Starting Test...")
test_susscess = []
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    for i in to_test:
        print("\n\n")
        print("Testing: [bold blue]"+i+"[/bold blue]")
        st = time.time()
        _temp = Autofhm(config=str(curr_directory)+"/"+i+"/config.json")
        _temp.get_features()
        _temp.fit()
        ti = time.time()-st
        res = _temp.test()
        del _temp
        print("time_taken in sec    " +f"{' = ':^15} {str(ti):<10}")

        res["Time to Build"] = ti
        with open(str(curr_directory)+"/results/"+i+".log.csv", "a") as df:
            _st = str(datetime.today()).split(".")[0]
            for metric in res.keys():
                _st = _st + "," + str(res[metric])
            _st = _st + "\n"
            df.write(_st)
