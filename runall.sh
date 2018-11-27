# #!/bin/bash

# 7.1
python code/main.py -n NIST36 -m FCN
python code/main.py -n MNIST -m CNN
python code/main.py -n NIST36 -m CNN
python code/main.py -n EMNIST -m CNN
# 7.2
python code/main.py -n flowers17 -p True
python code/main.py -n flowers17 -m CNN
# plot results in /log
python code/plot.py
mkdir ./figure
mv *.png ./figure/
