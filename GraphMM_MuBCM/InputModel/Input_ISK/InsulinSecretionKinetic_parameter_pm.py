
'''Calcium dynamics'''

fmd=0.01; 
fi=0.01; 
B=200; 
fv=0.00365; 
alpha=5.18e-15; 
vmd=4.2e-15; 
vcell=1.15e-12;
gL=250; 
Vm=-20; 
Vca=25; 
sm=5; 
Jsercamax=41.0; 
Kserca=0.27; 
Jpmcamax=21.0;
Kpmca=0.5; 
Jleak=-0.94; 
Jncx0=18.67; 


'''Granule dynamics'''

k1=20; 
km1=100; 
r1=0.6; 
rm1=1.0; 
r20=2; 
rm2=0.001;
r30=2; 
rm3=0.0001; 
u1=2000; 
u2=3.0; 
u3=0.02;
Kp=2.3; 
Kp2=2.3; 
Sm=5

# Potentiation factor

GlucFact = 1.2 # for 3 G, 0 for 0 G, and Grest=0 for 0 G too

time_scale=60*10 

'''Initial values'''

def InsulinSecretionKineticInit():
    
    # Initial state
    V_0=-73

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
    SE_0=1e-4

    x_0 = [V_0, Cmd_0, Ci_0, N1_0, N2_0, N3_0, N4_0, N5_0, N6_0, NF_0, NR_0, SE_0]
    
    return x_0
