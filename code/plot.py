import os
import pandas as pd
import numpy as np
import pylab as pl

log_dir = r"./log/"
root_dir = os.path.abspath('')
colors = ["lightcoral", "dodgerblue"]

def plot(name, log_csv):
    epoch = log_csv["epoch"]
    avg_loss = log_csv["avg_loss"]
    avg_accuracy = log_csv["avg_accuracy"]
    # loss
    pl.title(name + "_loss")
    pl.xlabel('epoch')
    pl.ylabel('avg_loss')
    pl.plot(epoch, avg_loss, color= colors[0], linestyle='-')
    pl.legend(loc='lower right')
    pl.grid(color='gray', linestyle="--")
    pl.savefig(root_dir + r"/figures/" + name.lower() + "_loss" + '.png')
    pl.close()
    # accuracy
    pl.title(name + "_accuracy")
    pl.xlabel('epoch')
    pl.ylabel('avg_accuracy')
    pl.plot(epoch, avg_accuracy, color= colors[1], linestyle='-')
    pl.legend(loc='lower right')
    pl.grid(color='gray', linestyle="--")
    pl.savefig(root_dir + r"/figures/" + name.lower() + "_accuracy" + '.png')
    pl.close()

def plot_all(dir):
    for file in os.listdir(dir):
        file_name = dir + file + "/progress.csv"
        log_csv = pd.read_csv(file_name)
        plot(file, log_csv)

if __name__ == '__main__':
    plot_all(log_dir)





