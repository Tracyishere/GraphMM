import numpy as np
import matplotlib.pyplot as plt
import math

'''
Plots
'''

def plot_groundtruth(GT, dt, sim_time, error, name=None):
    
    fig = plt.figure(figsize=(20, 4))
    n_variable = GT.shape[1]
    time = np.arange(0, sim_time, dt)

    for i in range(n_variable):
        ax = fig.add_subplot(1, 4, i+1)
        
        plt.yticks(fontproperties = 'Arial Narrow', size = 17)
        plt.xticks(fontproperties = 'Arial Narrow', size = 17)
        ax=plt.gca()  
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        plt.xlabel('Time [min]', fontproperties = 'Arial Narrow', size = 20)
        plt.ylabel(name[i], fontproperties = 'Arial Narrow', size = 20)
        
        plt.plot(time, GT[:, i], color='grey', linewidth=2, label='groundtruth') 
        
        if error is None:
            continue
        else:
            plt.fill_between(time, 
                             (GT[:, i] - error[:, i]).reshape(-1,), 
                             (GT[:, i] + error[:, i]).reshape(-1,), 
                             alpha=0.2, color='grey')

        plt.legend(loc='lower center', prop={'size': 14, 'family': 'Arial Narrow'})


def plot_inputmodel(input_result, input_std, dt, sim_time, name=None):
    
    fig = plt.figure(figsize=(20, 4))
    n_variable = input_result.shape[1]
    time = np.arange(0, sim_time, dt)

    for i in range(n_variable):
        ax = fig.add_subplot(1, 4, i+1)
        
        plt.yticks(fontproperties = 'Arial Narrow', size = 17)
        plt.xticks(fontproperties = 'Arial', size = 17)
        ax=plt.gca()  
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        plt.xlabel('Time [min]', fontproperties = 'Arial Narrow', size = 20)
        plt.ylabel(name[i], fontproperties = 'Arial Narrow', size = 20)
        
        plt.plot(time, input_result[:, i], color='black', linewidth=2, label='input model') 
        plt.fill_between(time, input_result[:, i] - input_std[:, i], 
        input_result[:, i] + input_std[:, i], alpha=0.1, color='black')

        plt.legend(loc='lower center', prop={'size': 14, 'family': 'Arial Narrow'})


def plot_surrogatemodel(surrogate, meta=None, inputmodel=None, inputmodelerror=None):
    
    sim_time = np.arange(0, surrogate.total_time, surrogate.dt)
    fig = plt.figure(figsize=(surrogate.n_state*6, 4))
    
    for k in range(surrogate.n_state):
        ax = fig.add_subplot(1, surrogate.n_state, k+1)
        
        plt.yticks(fontproperties = 'Arial Narrow', size = 17)
        plt.xticks(fontproperties = 'Arial Narrow', size = 17)
        plt.ylabel(surrogate.state[k],fontsize = 20, fontname = 'Arial')
        plt.xlabel('Time [{}]'.format(surrogate.unit),fontsize = 20, fontname = 'Arial')
        ax = plt.gca()  
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
                
        plt.plot(sim_time, surrogate.mean[:, k], color='orange', label = 'surrogate model', linewidth=2)
        plt.fill_between(sim_time, 
                         surrogate.mean[:, k]-surrogate.std[:, k], 
                         surrogate.mean[:, k]+surrogate.std[:, k],
                         color='orange', alpha = 0.3, label = 'surrogate model noise')
        
        if inputmodel is not None:
            plt.plot(sim_time, inputmodel[:, k], color='black', label = 'input model', linewidth=1.8)
        
        if inputmodelerror is not None:
            plt.fill_between(sim_time, inputmodel[:, k]-inputmodelerror[:, k], 
                                       inputmodel[:, k]+inputmodelerror[:, k],
                                       color='black', alpha = 0.1)

        if meta is not None:
            plt.plot(sim_time, meta[:, k], color='red', label = 'Metamodel')
            plt.fill_between(sim_time, meta[:, k]-meta.std[:, k], meta[:, k]+meta.std[:, k], 
                             color='red', alpha = 0.3, label = 'Metamodel noise')
        
        plt.legend(loc='best', prop={'size': 14, 'family': 'Arial Narrow'})


def plot_metamodel(meta,model_type,filename=None,plot_coupler=False,plot_error=False,plot_save=False):
    
    if model_type == 'GraphMM':
        fig = plt.figure(figsize=(8, 6))
        plt.yticks(fontproperties = 'Arial Narrow', size = 17)
        plt.xticks(fontproperties = 'Arial Narrow', size = 17)
        # for one coupling var only
        v1_idx, v2_idx = list(meta.coupling_graph.connect_idx.values())[0]
        m1 = meta.coupling_graph.model_idx['a']
        m2 = meta.coupling_graph.model_idx['b']
        v1 = m1.state[v1_idx]
        v2 = m2.state[v2_idx]
        
        plt.title('Metamodel_{}_{}'.format(v1, v2),fontsize = 20, fontname = 'Arial Narrow')
        plt.xlabel('Time [min]',fontsize = 20, fontname = 'Arial Narrow')
        plt.ylabel(v1, fontsize = 20, fontname = 'Arial Narrow')
        
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        
        sim_time = np.arange(0, m1.total_time, m1.dt)

        plt.plot(sim_time, m1.mean[:,v1_idx], label = 'surrogate_M1', color='darkgreen')
        if plot_error:
            plt.fill_between(sim_time, 
                            m1.mean[:,v1_idx]-m1.std[:,v1_idx],
                            m1.mean[:,v1_idx]+m1.std[:,v1_idx],
                            color='darkgreen', alpha = 0.3)
        
        sim_time = np.arange(0, m2.total_time, m2.dt)

        plt.plot(sim_time, meta.meta_mean[:,2].reshape(-1,), label = 'meta_M1', color='C2')
        if plot_error:
            plt.fill_between(sim_time, 
                        meta.meta_mean[:,2].reshape(-1,)-meta.meta_std[:,2].reshape(-1,), 
                        meta.meta_mean[:,2].reshape(-1,)+meta.meta_std[:,2].reshape(-1,), 
                        color='C2', alpha = 0.3)

        if plot_coupler:
            if len(meta.coupling_graph.coupling_variable) >= 1:
                plt.plot(sim_time, meta.coupling_graph.coupling_variable[0][:,0], label = 'coupling var', color='black')
                if plot_error:
                    plt.fill_between(sim_time, 
                                meta.coupling_graph.coupling_variable[0][:,0].reshape(-1,)-\
                                    meta.coupling_graph.coupling_variable[0][:,1].reshape(-1,), 
                                meta.coupling_graph.coupling_variable[0][:,0].reshape(-1,)+\
                                    meta.coupling_graph.coupling_variable[0][:,1].reshape(-1,),  
                                        color='grey', alpha = 0.3)

        plt.plot(sim_time, meta.meta_mean[:,-2].reshape(-1,), label = 'meta_M2', color='red')
        if plot_error:
            plt.fill_between(sim_time,  
                        meta.meta_mean[:,-2].reshape(-1,)-meta.meta_std[:,-2].reshape(-1,), 
                        meta.meta_mean[:,-2].reshape(-1,)+meta.meta_std[:,-2].reshape(-1,), 
                        color='red', alpha = 0.3)

        
        sim_time = np.arange(0, m2.total_time, m2.dt)
        plt.plot(sim_time, m2.mean[:,v2_idx], label = 'surrogate_M2', color='darkred')
        if plot_error:
            plt.fill_between(sim_time, 
                            m2.mean[:,v2_idx]-m2.std[:,v2_idx], 
                            m2.mean[:,v2_idx]+m2.std[:,v2_idx],
                            color='darkred', alpha = 0.3)
        
        
        plt.legend(loc='lower center', prop={'size': 16, 'family': 'Arial Narrow'})
        if plot_save and filename != None:
            plt.savefig(filename, dpi=200)
        plt.show()

    elif model_type == 'analytical':

        fig = plt.figure(figsize=(8, 6))
        plt.yticks(fontproperties = 'Arial', size = 17)
        plt.xticks(fontproperties = 'Arial', size = 17)
        v1_idx, v2_idx = meta.coupling_graph.connect_idx['a_b']
        v1 = meta.m1.state[v1_idx]
        v2 = meta.m2.state[v2_idx]
        plt.title('Metamodel_{}_{}'.format(v1, v2),fontsize = 20, fontname = 'Arial')
        plt.xlabel('Time [min]',fontsize = 20, fontname = 'Arial')
        plt.ylabel(v1,fontsize = 20, fontname = 'Arial')
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        
        sim_time = np.arange(0, meta.m1.total_time, meta.m1.dt)

        plt.plot(sim_time, meta.m1.mean[:,v1_idx], label = 'surrogate_M1', color='darkgreen')
        if plot_error:
            plt.fill_between(sim_time, 
                        meta.m1.mean[:,v1_idx]-meta.m1.std[:,v1_idx], 
                        meta.m1.mean[:,v1_idx]+meta.m1.std[:,v1_idx],
                        color='darkgreen', alpha = 0.3)


        sim_time = np.arange(0, meta.m2.total_time, meta.m2.dt)
        
        plt.plot(sim_time, meta.meta_m1_mean[:,v1_idx].reshape(-1,), label = 'meta_M1', color='C2')
        if plot_error:
            plt.fill_between(sim_time, 
                        meta.meta_m1_mean[:,v1_idx].reshape(-1,)-\
                        meta.meta_m1_std[:,v1_idx].reshape(-1,), 
                        meta.meta_m1_mean[:,v1_idx].reshape(-1,)+\
                        meta.meta_m1_std[:,v1_idx].reshape(-1,), 
                        color='C2', alpha = 0.3)

        if plot_coupler:
            if meta.coupler is not None:
                plt.plot(sim_time, meta.coupler[:,0], label = 'coupler', color='black')
                if plot_error:
                    plt.fill_between(sim_time, 
                                meta.coupler[:,0].reshape(-1,)-meta.coupler[:,1].reshape(-1,), 
                                meta.coupler[:,0].reshape(-1,)+meta.coupler[:,1].reshape(-1,),  
                                        color='grey', alpha = 0.3)

        plt.plot(sim_time, meta.meta_m2_mean[:,v2_idx].reshape(-1,), label = 'meta_M2', color='red')
        if plot_error:
            plt.fill_between(sim_time,  
                        meta.meta_m2_mean[:,v2_idx].reshape(-1,)-\
                        meta.meta_m2_std[:,v2_idx].reshape(-1,), 
                        meta.meta_m2_mean[:,v2_idx].reshape(-1,)+\
                        meta.meta_m2_std[:,v2_idx].reshape(-1,), 
                        color='red', alpha = 0.3)

        plt.plot(sim_time, meta.m2.mean[:,v2_idx], label = 'surrogate_M2', color='darkred')
        if plot_error:
            plt.fill_between(sim_time, 
                        meta.m2.mean[:,v2_idx]-meta.m2.std[:,v2_idx], 
                        meta.m2.mean[:,v2_idx]+meta.m2.std[:,v2_idx],
                        color='darkred', alpha = 0.3)
        
        
        plt.legend(loc='lower center', prop={'size': 16, 'family': 'Arial Narrow'})
        if plot_save and filename != None:
            plt.savefig(filename, dpi=200)
        plt.show()

    else:

        print('ERROR: No such model')


def plot_normpdf(u, sig, color, name=None):
    
    x = np.linspace(u - 3*sig, u + 3*sig, 100000)
    y = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig)

    if name is not None:
        plt.xlabel(name, labelpad=10)
    plt.ylabel('Density', labelpad=4)
    ax=plt.gca()  
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.plot(x, y, linestyle='-', linewidth=2, c=color)   
    plt.fill(x, y, alpha=0.1, c=color)