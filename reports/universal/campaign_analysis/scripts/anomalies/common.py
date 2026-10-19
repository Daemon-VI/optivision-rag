import json, glob, os, sys
import numpy as np
PK="C:/Users/Lenovo/projects/optivision-package"
CAMP=PK+"/reports/universal/campaign"
OUT="C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/anomalies"
def measured():
    return json.load(open(PK+"/src/optivision/presets/measured.json"))["models"]
def outcomes():
    return json.load(open(CAMP+"/outcomes.json"))
def wide_files():
    d={}
    for f in glob.glob(CAMP+"/*/wide/*.json"):
        folder=f.replace("\\","/").split("/")[-3]
        d[(folder,os.path.basename(f))]=json.load(open(f))
    return d
