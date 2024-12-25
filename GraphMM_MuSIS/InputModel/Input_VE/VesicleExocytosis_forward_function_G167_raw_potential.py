'''    
	Author: Chenxi Wang(chenxi.wang@salilab.org) 
	Date: 2021.01.22
	refer to Bertuzzi, A., Salinari, S. & Mingrone, G. Insulin granule trafficking in beta-cells: mathematical model of glucose-induced insulin secretion. Am. J. Physiol. Endocrinol. Metab. 293, E396–409 (2007).
	
'''

import numpy as np
from InputModel.Input_VE.VesicleExocytosis_parameter_G167 import *



''' help functions '''

def _f_H_gamma(G_ex):
	if G_ex <= G_star: H_gamma = 0.0
	elif G_star < G_ex <= G_hat: H_gamma = h_hat*(G_ex - G_star)/(G_hat - G_star)
	else: H_gamma = h_hat
	return H_gamma

def _f_H_rho(r): 
    if r < r_basal: H_rho = 0.0
    else: H_rho = kp*(r-r_basal)
    return H_rho

def _f(G_ex):
	if G_ex < G_star: f = f_basal
	else: f = f_basal + (1-f_basal)*(G_ex - G_star)/(Kf + G_ex - G_star)
	# f = f_basal
	return f


''' I. Inuslin granule dynamics '''

def _f_I(I_0, V_0, ts):
	# bI changes under different G_ex
	I_t = I_0 + ts*(-k*I_0*V_0 - aI*I_0 + bI)
	return I_t

def _f_V(V_0, I_0, F_0, ts):
	# F change to continous here tentatively, tv is simplified here
	# V_t = V_0 + ts*(-k*I_0*V_0 - aV*V_0 + bV + sigma*F[t-tV])
	V_t = V_0 + ts*(-k*I_0*V_0 - aV*V_0 + bV + sigma*F_0)
	return V_t

def _f_F(F_0, p_0, D_IR_0, G_ex, ts, time, sim_state, ton_lst):
	F_t = F_0 + ts*(_f_rho(p_0, G_ex, ts, time, sim_state, ton_lst)*D_IR_0 - sigma*F_0)
	return F_t

def _f_R(R_0, I_0, V_0, r_0, G_ex, ts):
	gamma = _f_gamma(r_0, G_ex, ts)
	R_t = R_0 + ts*(k*I_0*V_0  - gamma*R_0)
	return R_t

def _f_D(D_0, r_0, R_0, D_IR_0, G_ex, ts):
	D_t = D_0 + ts*(_f_gamma(r_0, G_ex, ts)*R_0 - k1_ass*(CT - D_IR_0)*D_0 + k1_dis*D_IR_0)
	return D_t

def _f_D_IR(D_IR_0, D_0, p_0, G_ex, ts, time, sim_state, ton_lst):
	D_IR_t = D_IR_0 + ts*(k1_ass*(CT - D_IR_0)*D_0 - k1_dis*D_IR_0 - _f_rho(p_0, G_ex, ts, time, sim_state, ton_lst)*D_IR_0)
	return D_IR_t


''' II. rate coefficients '''

H_gamma_G_ex = _f_H_gamma(G_ex)

def _f_gamma(r_0, G_ex, ts): 
	#G_ex here is G_ex[t-1] todo: connect two time step together, here G_ex is constant
	r_t = r_0 + ts*(eta*(-r_0 + r_basal + _f_ATP(G_ex) + _f_H_gamma(G_ex)))
	# r_t = r_basal
	return r_t

def _f_rho(p_0, r_0, ts, step, sim_state, ton_lst):
    
    # ###### one-step stimulation
	if sim_state == 'continous_K_stimulation':   
		t1 = 0*ts # one step, start from 6 or 0
		if step*ts >= t1: p_t = p_basal + p_hat*(1-np.exp(-epsilon*(step*ts-t1))) + sp*(step*ts-t1)
		else: p_t = p_basal
    
    # ####### cycle stimulation
	elif sim_state == 'intermitte_K_stimulation':    
		t_cycle = 6 # min
		if step*ts >= 0*t_cycle and step*ts < 1*t_cycle:
			t1 = 0*t_cycle
			p_t = p_basal + p_hat*(1-np.exp(-epsilon*(step*ts-t1))) + sp*(step*ts-t1)
		elif step*ts >= 2*t_cycle and step*ts < 3*t_cycle:
			t1 = 2*t_cycle
			p_t = p_basal + p_hat*(1-np.exp(-epsilon*(step*ts-t1))) + sp*(step*ts-t1)
		elif step*ts >= 4*t_cycle and step*ts < 5*t_cycle:
			t1 = 4*t_cycle
			p_t = p_basal + p_hat*(1-np.exp(-epsilon*(step*ts-t1))) + sp*(step*ts-t1)
		elif step*ts >= 6*t_cycle and step*ts < 7*t_cycle:
			t1 = 6*t_cycle
			p_t = p_basal + p_hat*(1-np.exp(-epsilon*(step*ts-t1))) + sp*(step*ts-t1)
		else:
			p_t = p_basal

	# ####### IHC pm simulation
	elif sim_state == 'IHC_pm_fit_simulation':    
		ton=48/60; toff=10/60
		t_cycle = ton + toff # min
		if step*ts > 0*t_cycle and step*ts < 0*t_cycle+1*toff:
			t1 = 0*t_cycle
			p_t = p_basal + p_hat*(1-np.exp(-epsilon*(step*ts-t1))) + sp*(step*ts-t1)
		elif step*ts >= 1*t_cycle and step*ts < 1*t_cycle+1*toff:
			t1 = 1*t_cycle
			p_t = p_basal + p_hat*(1-np.exp(-epsilon*(step*ts-t1))) + sp*(step*ts-t1)
		elif step*ts >= 2*t_cycle and step*ts < 2*t_cycle+1*toff:
			t1 = 2*t_cycle
			p_t = p_basal + p_hat*(1-np.exp(-epsilon*(step*ts-t1))) + sp*(step*ts-t1)
		elif step*ts >= 3*t_cycle and step*ts < 3*t_cycle+1*toff:
			t1 = 3*t_cycle
			p_t = p_basal + p_hat*(1-np.exp(-epsilon*(step*ts-t1))) + sp*(step*ts-t1)
		elif step*ts >= 4*t_cycle and step*ts < 4*t_cycle+1*toff:
			t1 = 4*t_cycle
			p_t = p_basal + p_hat*(1-np.exp(-epsilon*(step*ts-t1))) + sp*(step*ts-t1)
		elif step*ts >= 5*t_cycle and step*ts < 5*t_cycle+1*toff:
			t1 = 5*t_cycle
			p_t = p_basal + p_hat*(1-np.exp(-epsilon*(step*ts-t1))) + sp*(step*ts-t1)
		else:
			p_t = p_basal

	# ####### IHC pm simulation
	elif sim_state == 'IHC_pm_potential':
		cur_time = step 
		p_t = p_basal
		if len(ton_lst) >= 1:
			for pair in ton_lst:
				on, off = pair
				if on <= cur_time and cur_time <= off:
					p_t = p_basal + p_hat*(1-np.exp(-epsilon*(cur_time-on))) + sp*(cur_time-on)
                    # p_t = p_basal + p_hat*(1-np.exp(-epsilon*(cur_time*ts-on*ts))) + sp*(cur_time*ts-on*ts)
		else:
			t1 = 0*ts # one step, start from 6 or 0
			if step*ts >= t1: 
				p_t = p_basal + p_hat*(1-np.exp(-epsilon*(step*ts-t1))) + sp*(step*ts-t1)
			else: 
				p_t = p_basal
	else:
		print('ERROR: unknow state.')

	return p_t



def _f_ATP(G_ex): #todo: sin function and periodicity
    # ATP_t = 1/2*_f_HG(G_ex)*np.sin(t)
    ATP_t = 0.0
    return ATP_t


''' III. beta-cell population '''

def _f_ISR(F_0, G_ex):
	ISR_t = I0*sigma*F_0*_f(G_ex)*Nc*Ni*am2m*n_insulin*g2pg
	return ISR_t

def _f_G_ex(G_ex):
	return G_ex



''' IV. fit potential period '''

# def Vmem2Ksti_time(k_timelength, Vmem):

#     data = Vmem

#     if (data > -35).any():
#         threshold = 2  
#         first_diff = [data[i] - data[i - 1] for i in range(1, len(data))]
#         key_points = [i for i, diff in enumerate(first_diff) if abs(diff) > threshold]

#         data = key_points
#         threshold = 200 ############# for 50000 length
#         first_diff = [data[i] - data[i - 1] for i in range(1, len(data))]
#         key_key_points = [key_points[0]] + \
#             [[key_points[i], key_points[i+1]] for i, diff in enumerate(first_diff) if abs(diff) > threshold] + \
#                 [key_points[-1]]
        
#         combined_list = []
#         for i in range(len(key_key_points) - 1):
#             start = key_key_points[i]
#             end = key_key_points[i + 1]
#             if isinstance(start, list):
#                 start = start[1]
#             if isinstance(end, list):
#                 end = end[0]
#             combined_list.append([start, end])
#         combined_list[-1][-1] = len(Vmem)
        
#         K_sti_time = np.array(combined_list)
#         ratio = k_timelength / len(Vmem)
#         K_sti_time = np.round(K_sti_time*ratio,0).astype(int)
    
#     else:
#         K_sti_time = []

#     return K_sti_time


def Vmem2Ksti_time(k_timelength, Vmem):

    data = Vmem

    if (data > -35).any():
        threshold = 2  
        first_diff = [data[i] - data[i - 1] for i in range(1, len(data))]
        key_points = [i for i, diff in enumerate(first_diff) if abs(diff) > threshold]

        data = key_points
        threshold = 200 ############# for 50000 length
        first_diff = [data[i] - data[i - 1] for i in range(1, len(data))]
        key_key_points = [key_points[0]] + \
            [[key_points[i], key_points[i+1]] for i, diff in enumerate(first_diff) if abs(diff) > threshold] + \
                [key_points[-1]]
        
        combined_list = []
        for i in range(len(key_key_points) - 1):
            start = key_key_points[i]
            end = key_key_points[i + 1]
            if isinstance(start, list):
                start = start[1]
            if isinstance(end, list):
                end = end[0]
            combined_list.append([start, end])
        # combined_list[-1][-1] = len(Vmem)
        
        K_sti_time = np.array(combined_list)
        ratio = k_timelength / len(Vmem)
        K_sti_time = np.round(K_sti_time*ratio,0).astype(int)
    
    else:
        K_sti_time = []

    return K_sti_time