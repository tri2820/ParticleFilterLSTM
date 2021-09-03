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
    'MREKLD': 'solid',
}

N = 800

def average_(a):
    return np.average(np.average(np.array(a),axis=0),axis=0)

plt.ylabel('Num particle (log scale)')
plt.xlabel('Step')
plt.xticks(list(range(20)))

for method in ['KLD','MREKLD']:
    merger = {
    'errors':[],
    'elapsed':[],
    'num_particles':[]
    }

    for t in [1,2,3,4]:
        data = pickle.load( open(f'COLLETION_{method}_{N}_{t}.pkl', 'rb') )
        merger['num_particles'].append(data['num_particles'])

    
    num_particle = average_(merger['num_particles'])
    drawn = np.log(num_particle)
    print(num_particle)

    label = LABEL[method]
    linestyle = LS[method]

    plt.plot(drawn, label=label, linestyle=linestyle, color='black')
    plt.legend()

    # txt="Fig 2 Comparing number of particles between KLD PF and KLD PF with MrE"
    # plt.figtext(0.5, 0.02, txt, wrap=True, horizontalalignment='center', fontsize=12)

plt.show()
