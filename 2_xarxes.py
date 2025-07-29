import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random

#-- FUNCIONS ------------------------------------------------------------------------------------------------------------------------------------------------  

def cargar_matrius(carpeta, mascara):
    arxius = [f for f in os.listdir(carpeta) if f.endswith('.csv')]
    matrius = []
    for arxiu in arxius:
        matriu = pd.read_csv(os.path.join(carpeta, arxiu), header=None).to_numpy()
        matriu *= mascara
        matrius.append(matriu)
    matrius_np = np.array(matrius)
    return matrius_np


def construccio_multilayer(GM, RS, FA):
    num_subjs = FA.shape[0]
    num_nodes = FA.shape[1]
    em = np.zeros((num_subjs, num_nodes*2, num_nodes*2), dtype=float)
    for i in range(num_subjs):
                em[i,:76,:76] = GM[i,:,:]
                em[i,76:,76:] = RS[i,:,:]
                em[i,76:,:76] = FA[i,:,:]
                em[i,:76,76:] = FA[i,:,:]
    return em


def construccio_multiplex(GM, RS, FA):
    num_subjs = FA.shape[0]
    num_nodes = FA.shape[1]
    em = np.zeros((num_subjs, num_nodes*3, num_nodes*3), dtype=float)
    diagonal = np.zeros((num_subjs, num_nodes, num_nodes), dtype=float)
    diagonal[:, np.arange(num_nodes), np.arange(num_nodes)] = 1
    for i in range(num_subjs):
                em[i,:76,:76] = GM[i,:,:]
                em[i,76:76*2,76:76*2] = FA[i,:,:]
                em[i,76*2:,76*2:] = RS[i,:,:]
                em[i,76:76*2,:76] = diagonal[i,:,:]
                em[i,76*2:,:76] = diagonal[i,:,:]
                em[i,:76,76:76*2] = diagonal[i,:,:]
                em[i,76*2:,76:76*2] = diagonal[i,:,:]
                em[i,:76,76*2:] = diagonal[i,:,:]
                em[i,76:76*2,76*2:] = diagonal[i,:,:]
    return em


def construccio_monoplex(GM, RS, FA):
    em = (GM + RS + FA) / 3
    return em


def plot_matrius(em, titol):
    num_subjs = em.shape[0]
    n_rows = 1
    n_cols = 4
    _, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    individuals = random.sample(range(num_subjs), 4)
    for i, ind in enumerate(individuals):
        axs[i].imshow(em[ind,:,:], cmap='hot', interpolation='nearest')
        axs[i].set_title("Subject {}".format(ind))
    plt.savefig(str(titol) + ".png") 
    plt.close()


def plot_ml_matrix(A, ind):
    plt.imshow(A, origin='lower', cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title("Subject {}".format(ind))
    plt.savefig("MultiLayer_test_subj_"+ str(ind) +".png")
    plt.close()


#--------------------------------------------------------------------------------------------------------------------------------------------------  

GM_path = '/Users/aina/Desktop/TFG/codi/data/GM'
RS_path = '/Users/aina/Desktop/TFG/codi/data/RS'
FA_path = '/Users/aina/Desktop/TFG/codi/data/FA'

mascara_GM = np.load("mascares/mascara_GM.npy")
mascara_RS = np.load("mascares/mascara_RS.npy")
mascara_FA = np.load("mascares/mascara_FA.npy")

GM = cargar_matrius(GM_path, mascara_GM)
RS = cargar_matrius(RS_path, mascara_RS)
FA = cargar_matrius(FA_path, mascara_FA)

multilayer = construccio_multilayer(GM, RS, FA)
multiplex = construccio_multiplex(GM, RS, FA)
monoplex = construccio_monoplex(GM, RS, FA)

plot_matrius(multilayer, 'multilayer')
plot_matrius(multiplex, 'multiplex')
plot_matrius(monoplex, 'monoplex')

np.save("xarxes/multilayer.npy", multilayer)
np.save("xarxes/multiplex.npy", multiplex)
np.save("xarxes/monoplex.npy", monoplex)



