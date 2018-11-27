# #!/bin/bash

python code/main.py -n NIST36 -m FCN
python code/main.py -n MNIST -m CNN
python code/main.py -n NIST36 -m CNN
python code/main.py -n EMNIST -m CNN
python code/plot.py
mkdir ./figure
mv *.png ./figure/
