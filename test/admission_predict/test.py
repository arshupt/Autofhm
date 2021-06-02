from Autofhm import Autofhm
import time
from pathlib import Path

curr_directory = Path(__file__).resolve().parent

st = time.time()
_temp = Autofhm(config=str(curr_directory)+"/config.json")
_temp.get_features()
_temp.fit()
ti = time.time()-st
res = _temp.test()
print("Time Taken in seconds:" + str(ti))

with open(str(curr_directory) + "res.log", "a") as df:
    _st = "\n========================\n"+str(curr_directory.stem)
    for metric in res.items():
        _st = _st + "\n\t "+ metric+":\t"+res[metric]
    df.write(_st)
