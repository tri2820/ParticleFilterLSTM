import pickle
import numpy as np
import matplotlib.pyplot as plt 

method = ['SIR','KLD','MRE','MREKLD']

merger = {
    'errors':[],
    'elapsed':[],
    'num_particles':[]
}

LABEL = {
    'SIR': 'SIR PF',
    'KLD': 'KLD PF',
    'MRE': 'SIR PF with MrE (Proposal 2)',
    'MREKLD': 'KLD PF with MrE (Proposal 1)',
}

LS = {
    'SIR': 'dotted',
    'KLD': 'dashdot',
    'MRE': 'solid',
    'MREKLD': '--',
}

N = 800

def average_(a):
    return np.average(np.average(np.array(a),axis=0),axis=0)

plt.ylabel('Average error')
plt.xlabel('Step')
plt.xticks(list(range(20)))

for method in ['SIR','KLD','MREKLD','MRE']:

    merger = {
    'errors':[],
    'elapsed':[],
    'num_particles':[]
    }

    for t in [1,2,3,4]:
        data = pickle.load( open(f'COLLETION_{method}_{N}_{t}.pkl', 'rb') )
        merger['errors'].append(data['errors'])
        merger['elapsed'].append(data['elapsed'])

    print(f'{method} elapsed',np.average(np.array(merger['elapsed'])))
    print(f'{method} error',np.average(np.array(merger['errors'])))
    
    errors = average_(merger['errors'])

    label = LABEL[method]
    linestyle = LS[method]

    plt.plot(errors, label=label, linestyle=linestyle, color='black')
    plt.legend()

    txt="Fig 1 Comparing average error between traditional PFs and proposals"
    plt.figtext(0.5, 0.02, txt, wrap=True, horizontalalignment='center', fontsize=12)



plt.show()
