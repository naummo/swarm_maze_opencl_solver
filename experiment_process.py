"""
experiment_process.py contains some post-processing functionality.
Not used anymore.
"""

import csv
import os
import numpy as np

import configs as cfg


if not os.path.exists(cfg.reporting_dir):
    os.makedirs(cfg.reporting_dir)
# Collisions
results = [0 for i in range(cfg.seconds)]
for filename in os.listdir(cfg.reporting_dir):
    if "collisions.csv" in filename:
        csvfile = open(os.path.join(cfg.reporting_dir, filename))
        csvreader = csv.reader(csvfile, delimiter=',', lineterminator='\n',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in csvreader:
            for value in row:
                if value != '':
                    results[int(np.trunc(float(value) / cfg.framespersecond))] += 1
        csvfile.close()

print(results)
csvfile = open(os.path.join(".", "collisions_total.csv"), 'w')
csvwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
csvwriter.writerow([(str(val) if val != 0 else "") for val in results])
csvfile.close()
