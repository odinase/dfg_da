from scipy.io import loadmat


DATA_PATH = "./data/at612"
MAT_FILE = "priorLikelihood612.mat"


if __name__ == "__main__":
    ws = loadmat(DATA_PATH + "/" + MAT_FILE)
    pass