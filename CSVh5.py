import csv
import numpy as np
import sys
import h5py
import string

if len(sys.argv) != 3:
    print("Improper Input: python3 CSVh5.py <input_file> <output_location>")
    sys.exit()

dataStore = []


output = sys.argv[2]

fh5 = h5py.File(output, 'w')
fh5 = fh5.create_group("Group")


with open(sys.argv[1]) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        length = len(row)
        for cols in range(length - 1):

            print(row[cols])
            dataStore.append(float(row[cols]))

dataStore = np.array(dataStore)

fh5.create_dataset("result", data=dataStore)


        