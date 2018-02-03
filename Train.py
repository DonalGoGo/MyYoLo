import numpy as np
import cv2
import tensorflow as tf
from ReadData import *
from Yolo import *
from config import *
from Solver import Solver


def train():
    yolo = Yolo()
    solver=Solver(yolo)
    solver.train()
if __name__=="__main__":
    train()