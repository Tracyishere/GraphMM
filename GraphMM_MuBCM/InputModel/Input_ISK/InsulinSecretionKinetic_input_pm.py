# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy.interpolate import interp1d

# % Model of the combined calcium + granule compartments, as per Yi-der Chen,
# % Shaokun Wang, and Arthur Sherman, doi:10.1529/biophysj.107.124990

# % Identifying the Targets of the Amplifying Pathway for Insulin Secretion
# % in Pancreatic β-Cells by Kinetic Modeling of Granule Exocytosis

# % Based on the original XPP model,
# % http://lbm.niddk.nih.gov/sherman/gallery/beta/Vesicle/

# % Toni Giorgino  toni.giorgino@isib.cnr.it

# % State space has a dimension of 11. Correspondences shown below
# %  Here		XPP	Paper	Note
# %  x(1)		Cmd	 	Microdomain [Ca++]
# %  x(2)		Ci		Cytosol [Ca++]
# % --------------------------------------------
# %  x(3)		N1		Primed (II)
# %  x(4)		N2		Bound
# %  x(5)		N3		Triggered
# %  x(6)		N4		Fused
# %  x(7)		N5		Primed (I)
# %  x(8)		N6		Just arrived from Reserve Pool
# %  x(9)		NF		Fused (II)
# %  x(10)	    NR		Releasing 
# % --------------------------------------------
# %  x(11)	    SE		Secretion (integrated)

# % Note that equations for Cmd, Ci are a nonlinear 2-ODE system coupled to
# % the glucose (voltage) input. Granule compartments are a linear 10-ODE
# % system with time-varying coefficients. Secretion is a simple integral.

# % Voltage (stepped) input V 
# %   --> fraction of opened channels minf(V), eq. (5)
# %   --> ion flux through L-type channels IL ~ minf(V)
# %   --> ion flux through R-type channels IR ~ .25 * IL
# %   --> Ci and Cmd are replenished through these channels
# %   --> Ci leaks through three currents (see expression for L)

# % Ci  gates priming (r2) and resupply (r3) rates for pre-docked granules
# % Cmd gates binding and fusion

# % Parameters as given in the paper. 

def ExocytosisModel(x,t):

    ts=60 # sec to Minutes
    Cmd, Ci, N1, N2, N3, N4, N5, N6, NF, NR, SE = x

    '''Calcium dynamics'''
    dCmd, dCi = ExocytosisModelCalcium(t, Cmd, Ci)   
    
    '''Granule dynamics'''
    # Potentiation factor
    # GlucFact = 1.2 for 3 G, 0 for 0 G, and Grest=0 for 0 G too
    GlucFact = 1.2
    
    # The inter-compartment kinetic rates
    k1=20; km1=100; r1=0.6; rm1=1; r20=2; rm2=0.001;
    r30=2; rm3=0.0001; u1=2000; u2=3; u3=0.02;
    Kp=2.3; Kp2=2.3
    
    # Vesicles: initial conditions
    #    N1(start=14.71376);
    #    N2(start=0.612519);
    #    N3(start=0.0084499);
    #    N4(start=5.098857e-6);
    #    N5(start=24.539936);
    #    N6(start=218.017777);
    #    NF(start=0.003399);
    #    NR(start=0.50988575);
    #    SE(start=0.0);

    # EQUATIONS
    # Rates for pre-docking granules depend on cytosol Ca++ (Ci)
#     r2 = GlucFact*r20*Ci/(Ci + Kp2)
    r2 = r20*Ci/(Ci + Kp2)
    r3 = GlucFact*r30*Ci/(Ci + Kp)

    # Rates for docked granules depend on microdomain Ca++ (Cmd)  
    dN1 = ts*(-(3*k1*Cmd + rm1)*N1 + km1*N2 + r1*N5) # Primed
    dN2 = ts*(3*k1*Cmd*N1 -(2*k1*Cmd + km1)*N2 + 2*km1*N3) # Bound
    dN3 = ts*(2*k1*Cmd*N2 -(2*km1 + k1*Cmd)*N3 + 3*km1*N4) # Triggered
    dN4 = ts*(k1*Cmd*N3 - (3*km1 + u1)*N4) # Fused
    dN5 = ts*(rm1*N1 - (r1 + rm2)*N5 + r2*N6) # Primed
    dN6 = ts*(r3 + rm2*N5 - (rm3 + r2)*N6) # Docked
    dNF = ts*(u1*N4 - u2*NF) # Fused
    dNR = ts*(u2*NF - u3*NR) # Releasing
    dSE = ts*(u3*NR) # Secretion

    # Compose the state vector
    dx = [dCmd, dCi, dN1, dN2, dN3, dN4, dN5, dN6, dNF, dNR, dSE]

    return np.array(dx, dtype='float64')

# Derivatives of the [Ca++] in microdomain and cytosol. 
# Calls the SquareSource function to get the membrane potential (V) at time t.

def ExocytosisModelCalcium(t,Cmd,Ci):
    
    # Global parameters 
    ts=60 # Conversion to minutes

    # These names mirror the original XPP model 
    fmd=0.01; fi=0.01; B=200; fv=0.00365; 
    alpha=5.18e-15; vmd=4.2e-15; vcell=1.15e-12;
    gL=250; Vm=-20; Vca=25; sm=5

    # Leak fluxes
    Jsercamax=41.0; Kserca=0.27; Jpmcamax=21.0;
    Kpmca=0.5; Jleak=-0.94; Jncx0=18.67;

    # Ca++ concentrations, with initial values
    #    Cmd(start=0.0674);
    #    Ci(start=0.06274);

    # EQUATIONS    
    #Forced membrane potential
    V = ExocytosisModelMembrane(t);
    # V = Vmem(step)

    # The leak currents follow these three kinetic laws
    Jserca = Jsercamax*Ci**2/(Kserca**2 + Ci**2)
    Jpmca = Jpmcamax*Ci/(Kpmca + Ci)
    Jncx = Jncx0*(Ci - 0.25)
    L = Jserca + Jpmca + Jncx + Jleak

    # Rates are in seconds, following Yi-der, multiply ODE RHS's by ts to get in minutes
    # Currents passing through the voltage-switched channels
    IL=gL*minf(V)*(V- Vca)
    IR=0.25*IL
  
    # Molar fluxes:
    JL = alpha*IL/vmd
    JR = alpha*IR/vcell

    # Compartment model
    dCmd = ts*(-fmd*JL - fmd*B*(Cmd - Ci))
    dCi  = ts*(-fi*JR + fv*fi*B*(Cmd - Ci) - fi*L)

    return dCmd, dCi

# The voltage-gated switching function. It is a function of V, which returns the fraction of open channels. Parameters from paper. 
def minf(V):
    '''
        To test-
            v=linspace(-100,20,100)
            plot(v,minf(v))
    '''
    Vm=-20
    Sm=5
    return 1./(1 + np.exp((Vm-V)/Sm))

def ExocytosisModelInit():
    
    # Initial state
    Cmd_0=0.0674
    Ci_0=0.06274
    
    N1_0=14.71376
    N2_0=0.612519
    N3_0=0.0084499
    N4_0=5.098857e-6
    N5_0=24.539936
    N6_0=218.017777
    NF_0=0.003399
    NR_0=0.50988575
    SE_0=0.0

    x_0 = [Cmd_0, Ci_0, N1_0, N2_0, N3_0, N4_0, N5_0, N6_0, NF_0, NR_0, SE_0]
    
    return x_0

# Model of a square-wave source. This is used to represent the (forced) external membrane potential. 
def ExocytosisModelMembrane(t):
    '''
        To test-
            test_t=linspace(0,60,600);
            plot(test_t,SquareSource(test_t))
    ''' 
    
    # Vrest=-70; Vburst=-20
    # tstep=0.0; ton=6.0; toff=6.0
    
    # # Figure 2
    # ton=8/60; toff=8/60
    
    # # # Figure 4 square-wave
    # tcycle = ton + toff
    # V = Vrest + (Vburst - Vrest)*(heav(np.mod(t, tcycle)) - heav(np.mod(t, tcycle) - toff))   
    
    # # # Figure 4&6 step glucose stimulation
    # # tcycle = 1e-20
    # # V = Vrest + (Vburst - Vrest)*(heav(np.remainder(t, tcycle)) - heav(np.remainder(t, tcycle) - toff)) 
    
    Vrest=-73; Vburst=-23
    tstep=0.0; ton=6.0-1e-10; toff=6.0

    ton=48/60; toff=10/60
    tcycle = ton + toff
    V = Vrest + (Vburst - Vrest)*(heav(np.mod(t, tcycle)-1e-10) - heav(np.mod(t, tcycle) - toff))     


    return V

# Heaviside function
def heav(x):
    return .5*(1+np.sign(x))

# %%
from scipy.interpolate import interp1d

def interpolate(x, y, xnew):
    
    f = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    ynew = f(xnew)

    return ynew

# %%
x_0 = ExocytosisModelInit()
steps = 1000000
t = np.linspace(0, 5, steps)

for celli in range(1,2):
    cell_potential = np.genfromtxt('../Input_IHC/input_IHC_no_hub/input_IHC_cell_{}.csv'.format(celli), delimiter=',', skip_header=1)
Vmem = interpolate(x=np.linspace(0,300,len(cell_potential)), 
                   y=cell_potential[:, 0], 
                   xnew=np.linspace(0, 300, steps))

result_G3 = odeint(ExocytosisModel, x_0, t)
result = np.concatenate((ExocytosisModelMembrane(t).reshape(-1,1), result_G3), axis=1)

# fout = open('./input_ISK_con.csv', 'w')
# for i in range(len(result)):
#     print(*list(result[i, :]), file=fout, sep=',')
# fout.close()

# f = interp1d(t, result_G3[:, -1])
# SE_res = f(np.arange(0,50))
# measured_G3 = 4.5*(SE_res[3:len(SE_res)]-SE_res[1:(len(SE_res)-2)])


# %%
for i in range(12):
    if i == 0:
        plt.plot(Vmem)
    plt.plot(result[:, i])
    plt.show()
# %%
