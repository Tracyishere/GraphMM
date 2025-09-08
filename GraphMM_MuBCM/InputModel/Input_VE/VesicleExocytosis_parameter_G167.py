'''    
	Author: Chenxi Wang(chenxi.wang@salilab.org) 
	Date: 2021.01.22
	refer to Bertuzzi, A., Salinari, S. & Mingrone, G. Insulin granule trafficking in beta-cells: mathematical model of glucose-induced insulin secretion. Am. J. Physiol. Endocrinol. Metab. 293, E396–409 (2007).
'''

G_ex = 16.7 # mM

def _get_G_dependent_param(G_ex):
    if G_ex == 0:
        k1_ass = 1.447*10**(-5) #min-1
        k1_dis = 0.10375 #min-1
        CT = 500

    else:
        k1_ass = 5.788*10**(-5) #min-1
        k1_dis = 0.255 #min-1
        CT = 300

    return k1_ass, k1_dis, CT

''' I. Inuslin granule dynamics '''

k = 10**(-2) #min-1
aI = 0.3 #min-1
bI = 4*8 #min-1, for 16.7
aV = 0.6 #min-1
bV = 6 #min-1, changable
sigma = 30 #min-1
tR = 5 #min
k1_ass, k1_dis, CT = _get_G_dependent_param(G_ex)

''' II. rate coefficients '''

eta = 4 # min-1
epsilon = 4 # min-1
r_basal = 10**(-4) #min-1
p_basal = 0.02 #min-1
h_hat = 3.93 * 10**(-3) # min-1
tG = 1 # min
G_hat = 10  # mM
G_star = 4.58 # mM
kp = 350
p_hat = 1 # min-1
sp = 4*10**(-3) # min-2
tv = 5 # min

''' III. beta-cell population '''

I0 = 1.6 # amol for rat
Nc = 1000 # cell per islet
f_basal = 0.05
Kf = 3.43 #mM
Ni = 1 # islet per pancreas
am2m = 1e-18 # amol to mol
g2pg = 1e+12  # g to pg
n_insulin = 5734 # g/mol

''' Initial values '''

def VesicleExocytosisInit(G_ex):
    
    # Initial state
    I_0=10.000001693497563
    V_0=9.995060410323925
    F_0=0.014349656276487302
    R_0=10000.003120945406
    D_0=949.9968625400597
    D_IR_0=49.93274600807214
    gamma_0=0.0001001318616122718
    rho_0=0.0004969323156936569
    ISG_0=0.1502677464561559*10**(-6)
    if G_ex == 0:
        G_ex_0=1e-2
    else:
        G_ex_0=G_ex

    # x_0 = [I_0, V_0, F_0, R_0, D_0, D_IR_0, gamma_0, rho_0, ISG_0, G_ex_0]
    x_0 = [I_0, V_0, F_0, R_0, D_0, D_IR_0, gamma_0, rho_0, ISG_0]
    
    return x_0


def VesicleExocytosisInit_pm(G_ex):
    
    # Initial state
    I_0=10
    V_0=10
    F_0=0.
    R_0=10000
    D_0=950
    D_IR_0=50
    gamma_0=0.
    rho_0=0.
    ISG_0=9.28*10**(-6)
    if G_ex == 0:
        G_ex_0=1e-2
    else:
        G_ex_0=G_ex

    x_0 = [I_0, V_0, F_0, R_0, D_0, D_IR_0, gamma_0, rho_0, ISG_0, G_ex_0]
    # x_0 = [I_0, V_0, F_0, R_0, D_0, D_IR_0, gamma_0, rho_0, ISG_0]
    
    return x_0