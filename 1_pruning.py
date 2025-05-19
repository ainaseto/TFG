import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

#-- FUNCIONS ------------------------------------------------------------------------------------------------------------------------------------------------  

def positius():
    pacients = pd.read_csv('/Users/aina/Desktop/TFG/codi/data/demographics.csv')
    pacients_zero = pacients[pacients["mstype"] >= 0]["ID"].astype(str).str.zfill(4).tolist()
    return pacients_zero


def mitjana_xarxes(path, pacients_zero):
    matriu = np.zeros((76,76), dtype=float)
    for pacient in os.listdir(path):
        if pacient[:4] in pacients_zero:
            matriu += pd.read_csv(os.path.join(path, pacient), header=None).to_numpy() 
    matriu /= len(pacients_zero)
    return matriu


def histograma_pesos(matriu, tipus):
    plt.hist(matriu.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel('Pes del graf')
    plt.ylabel('Freqüència')
    plt.title(f'Histograma dels pesos del graf - {tipus}')
    plt.savefig(f"plots/informe/histogrames/{tipus}.png")
    plt.show()


def netejar_graf(matriu, threshold):
    arestes_originals = np.count_nonzero(matriu)
    print(f'\nNúmero arestes originals: {arestes_originals}')
    matriu_neta = (matriu >= threshold).astype(float)
    arestes_thresholds = np.sum(matriu >= threshold)
    print(f'Número darestes >= {threshold}:', arestes_thresholds)
    return matriu_neta


def veins_FA(veins, matriu):
    df = pd.read_csv(veins, delimiter=';')
    df['veins'] = df['veins'].astype(str).apply(lambda x: list(map(int, x.split(','))))
    matriu_veins = np.zeros((76, 76), dtype=int)
    for _, row in df.iterrows():
        region = row['region_id']
        veins = row['veins']
        for vei in veins:
            matriu_veins[region-1, vei-1] = 1
            matriu_veins[vei-1, region-1] = 1 
    arestes_originals = np.count_nonzero(matriu)
    print(f'\nNúmero arestes originals: {arestes_originals}')
    arestes_veins = np.count_nonzero(matriu_veins)
    print('Número darestes regions veïnes:', arestes_veins)
    return matriu_veins


#--------------------------------------------------------------------------------------------------------------------------------------------------  

GM_path = '/Users/aina/Desktop/TFG/codi/data/GM'
RS_path = '/Users/aina/Desktop/TFG/codi/data/RS'
FA_path = '/Users/aina/Desktop/TFG/codi/data/FA'
veins = '/Users/aina/Desktop/TFG/codi/data/veins_nodes.csv'

pacients_zero = positius()
gm_xarxa = mitjana_xarxes(GM_path, pacients_zero)
rs_xarxa = mitjana_xarxes(RS_path, pacients_zero)
fa_xarxa = mitjana_xarxes(FA_path, pacients_zero)

histograma_pesos(gm_xarxa, 'GM')
histograma_pesos(rs_xarxa, 'RS')
histograma_pesos(fa_xarxa, 'FA')

mascara_GM = netejar_graf(gm_xarxa, 0.59)
mascara_RS = netejar_graf(rs_xarxa, 0.44)
mascara_FA = veins_FA(veins, fa_xarxa)

np.save("mascares/mascara_GM.npy", mascara_GM)
np.save("mascares/mascara_RS.npy", mascara_RS)
np.save("mascares/mascara_FA.npy", mascara_FA)
