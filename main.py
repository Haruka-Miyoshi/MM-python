import numpy as np
from mm import MarkovModel

data = np.loadtxt('./data/dataset.txt')
categorys = np.loadtxt('./data/categorys.txt')

"""実行文"""
if __name__ == '__main__':
    model = MarkovModel(3, 2)
    A, B, row = model.parameter_inference(categorys, data)