import os
import pandas as pd
import numpy as np
import pylab as pl

colors = ["lightcoral", "dodgerblue"]

def plot(name, log_csv):
    epoch = log_csv["epoch"][0:21]
    avg_loss = log_csv["avg_loss"][0:21]
    avg_accuracy = log_csv["avg_accuracy"][0:21]
    # loss
    pl.title(name + "_loss")
    pl.xlabel('epoch')
    pl.ylabel('avg_loss')
    pl.plot(epoch, avg_loss, color= colors[0], linestyle='-')
    pl.legend(loc='lower right')
    pl.grid(color='gray', linestyle="--")
    pl.savefig(name.lower() + "_loss" + '.png')
    pl.close()
    # accuracy
    pl.title(name + "_accuracy")
    pl.xlabel('epoch')
    pl.ylabel('avg_accuracy')
    pl.plot(epoch, avg_accuracy, color= colors[1], linestyle='-')
    pl.legend(loc='lower right')
    pl.grid(color='gray', linestyle="--")
    pl.savefig(name.lower() + "_accuracy" + '.png')
    pl.close()


if __name__ == '__main__':
    log_csv = pd.read_csv("./progress.csv")
    plot("flowers17-cnn-pretrain-True",log_csv)





