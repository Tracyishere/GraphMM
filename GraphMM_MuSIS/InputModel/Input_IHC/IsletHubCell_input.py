# %%

import scipy.io
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy.interpolate import interp1d
import numpy as np


def silencingSSCM(param, hub):

    # %This function runs the SSCM model with or without the silencing of some 
    # %cells. 
    # %
    # %Inputs: hub- the number of the hub cell that will be silenced, 0 if no hub
    # %               cell is to be silenced
    # %        filename- the name of the .mat file containing the parameters
    # %           (e.g. 'parameters.mat')
    # %
    # %Outputs: t- associated time vector
    # %         x- solutions of the differential system organized as
    # %               Columns 1:N- Voltage
    # %               Columns N+1:2*N- n
    # %               Columns 2*N+1:3*N- s
    # %               Columns 3*N+1:4*N- Calcium

    # %STEP 1: Parameter set up
    SimTime=300 # simulation time in seconds
    
    gkatp = param['gkatp'].reshape(-1,)
    # gkatp_mean = 15
    # gkatp_std = gkatp_mean*0.5
    # gkatp = np.random.normal(gkatp_mean, gkatp_std, 57).reshape(-1,)

    gca = param['gca'].reshape(-1,)
    gk = param['gk'].reshape(-1,)
    gs = param['gs'].reshape(-1,)
    kca = param['kca'].reshape(-1,)
    x0 = param['x0'].reshape(-1,)
    M = param['M']
    N = M.shape[0] ##the number of cell

    # adj_matrix = np.where(param['M'] == 0, 0, 1)
    # np.fill_diagonal(adj_matrix, 0)

    # gc_mean = 145
    # gc_std = gc_mean*0.5
    # M = []

    # for i in range(57):
    #     M_i = np.random.normal(gc_mean, gc_std, 57)
    #     M_i *= adj_matrix[i]
    #     M_i[i] = sum(M_i[0:i]) + sum(M_i[i+1:])
    #     M_i *= -1
    #     M_i[i] *= -1
    #     M += [M_i]
    # M = np.array(M)


    # x0[:57] = [-70]*57
    
    # %STEP 2: Define model equations and solve system
    time = np.linspace(0, SimTime, 10000)
    result = odeint(odefunc, x0, time, args=(kca, gca, gkatp, gk, gs, N, M, hub))
                
    return result



def odefunc(x, t, kca, gca, gkatp, gk, gs, N, M, hub):

    # Parameters
    Cm = 5.3;           
    Vm = -20;           
    theta_m = 12;       
    Vn = -16;           
    theta_n = 5.6;      
    Vs = -52;           
    theta_s = 10;       
    Vca = 25;           
    Vk = -75;           
    tau_n = 20e-3;      
    tau_s = 20000e-3;   
    f = 0.01;           
    alph = 4.5e-3;      
    gcl=100;
    Vcl=-70;
    sigma=np.zeros((N,));
    if hub != 0: sigma[hub]=1;
    # print(sigma)
        
        
    # Declaration of variable
    V = x[:N]; # mV
    n = x[N:2*N]; # unitless
    s = x[2*N:3*N]; # unitless
    Ca = x[3*N:4*N]; # uM    
        
        
    # Calculate expressions
    minf = (1+np.exp((Vm-V)/theta_m))**(-1);
    ninf = (1+np.exp((Vn-V)/theta_n))**(-1);
    sinf = (1+np.exp((Vs-V)/theta_s))**(-1);
    Ica = gca*minf*(V-Vca);
    Ikatp = gkatp*(V-Vk);
    Ik = gk*n*(V-Vk);
    Is = gs*s*(V-Vk);
    Icl = gcl*sigma*(V-Vcl);
    Icoup = np.dot(M,V); #Multiply two matrices

    # RHS
    dVdt = (Ica + Ikatp + Ik + Is + Icl + Icoup)/(-Cm); # mV/s
    dndt = (ninf - n)/tau_n; # 1/s
    dsdt = (sinf - s)/tau_s; # 1/s
    dCadt = f*(-alph*Ica - kca*Ca); # uM/s
    #print('test:',(Ica + Ikatp + Ik + Is + Icl + Icoup)/(-Cm))
    #print('V=',V)
    #print("V_shape=",V.shape)
    
    dxdt = list(dVdt) + list(dndt) + list(dsdt) +list(dCadt)
    #dparameters = np.array([minf_mean,ninf_mean,sinf_mean,Ica_mean,Ikatp_mean,Ik_mean,Is_mean,Icl_mean,Icoup_mean])
    
    return np.array(dxdt)



def computeFuncConn(t, Ca, N):
    
    #     %This function calculates the functional connectivity of an islet using the
    #     %algorithm described in the Functional Connectivity section of "Flipping
    #     %the switch on the hub cell: Islet desynchronization through cell silencing"
    #     %
    #     %Inputs: t- time vector (does not need to be equally spaced)
    #     %        Ca- corresponding calcium traces
    #     %
    #     %Output: F- a square matrix where the i,jth entry is 1 is cells i,j are
    #     %           functionally connected or 0 otherwise


    # %STEP 1: Interpolation to get evenly spaced data & binarizing using 0.15 uM
    Caint=interp1d(t, Ca, axis=0)(t);
    c=np.where(Caint>0.15, 1, 0);

    
    # %STEP 2: Calculate time active and coactive
    T=np.zeros((N,N)); # coactivity time
    for i in range(N):
        for j in range(i, N):
            count = 0
            for k in range(c.shape[0]):
                if c[k,i]+c[k,j]==2:
                    count += 1
            T[i,j]=count

    
    # %STEP 3: Calculate C
    F=np.zeros((N,N)); # adjacency matrix
    for i in range(N):
        for j in range(i+1,N):
            C=T[i,j]/np.sqrt(T[i,i]*T[j,j]);
            if C>0.85: F[i,j]=1;
                
    return F.transpose()+F


# %%
FigPara = scipy.io.loadmat('./Fig7Parameters.mat')
result = silencingSSCM(FigPara, []) # no hub cell
# result = silencingSSCM(Fig6Para, []) is equal to result = silencingSSCM(Fig6Para, 0)

for i in range(57):
    fout = open('./Input_IHC_R05/input_IHC_cell_{}.csv'.format(i), 'w')
    print(*['V', 'n', 's', 'Ca'], file=fout, sep=',')
    for line in result:
        print(*[line[i], line[i+57], line[i+57*2], line[i+57*3]], file=fout, sep=',')
    fout.close()





















# %%
for i in range(57):
    fout = open('./Input_IHC_R05/input_IHC_cell_{}.csv'.format(i), 'w')
    print(*['V', 'n', 's', 'Ca'], file=fout, sep=',')
    for line in result:
        print(*[line[i], line[i+57], line[i+57*2], line[i+57*3]], file=fout, sep=',')
    fout.close()



# %%
FigPara = scipy.io.loadmat('./InputModel/Input_IHC/Fig7Parameters.mat')
Ca_t = np.genfromtxt('/Users/tracy/Desktop/coupling_graph_param_ISK_IHC_300s_cell_0.csv', delimiter=',')
Ca_t = Ca_t[::50, 0]
result = silencingSSCM(FigPara, [], Ca_t) # no hub cell



# %%
Conn = computeFuncConn(np.linspace(0,300,10000), result[:, 57*3:57*4], 57)
# %%


import numpy as np
from scipy import fftpack

def compute_synchronization_index(data, sampling_rate=1.0):
    prominent_periods = []

    # 1. 对每个细胞迹线执行FFT。
    for cell_trace in data:
        spectrum = fftpack.fft(cell_trace)
        frequencies = fftpack.fftfreq(len(cell_trace), d=sampling_rate)
        positive_freq_idx = np.where(frequencies > 0)
        magnitudes = np.abs(spectrum[positive_freq_idx])
        
        # 2. 确定每个细胞的主要周期。
        prominent_frequency = frequencies[positive_freq_idx][np.argmax(magnitudes)]
        prominent_period = 1.0 / prominent_frequency
        prominent_periods.append(prominent_period)

    # 3. 找出众数周期。
    mode_period = np.median(prominent_periods)  # 使用中位数作为众数

    # 4. 计算活动周期在众数周期±某个范围内的细胞的百分比。
    tolerance = mode_period * 0.1  # 例如，设定10%的公差
    synchronized_cells = np.sum((prominent_periods > (mode_period - tolerance)) & 
                                (prominent_periods < (mode_period + tolerance)))
    synchronization_index = synchronized_cells / len(data) * 100

    return synchronization_index

# 假设data是一个二维数组，每行是一个细胞的活动迹线。
data = np.random.rand(100, 1000)  # 这只是一个随机数据示例
print(compute_synchronization_index(result[:, 3*57:4*57]))

# %%
