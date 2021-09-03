import logging
from scipy.stats import norm as norm_distribution
from scipy.stats import multivariate_normal as xd_norm_distribution
from numpy.linalg import norm as norm_distance
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from numpy.random import random as rand
from operator import itemgetter
import scipy
from kld_sampling import KLDResampling
import time

"""
Sometimes landmarks arrangement gets symmetry problem 
"""

measure = lambda landmarks, robot, sensor_std_err: norm_distance(landmarks - robot, axis=1) + (rand(len(landmarks)) * sensor_std_err)
effective = lambda weights: sum(np.square(weights))
uniform = lambda size: np.ones(size) / size
most_likely = lambda cloud: max(cloud, key=itemgetter(0))[1]
mean = lambda weights, particles: np.average(particles, weights=weights, axis=0)
resample_from_index = lambda particles, weights, indexes: (particles[indexes], uniform(len(indexes)))
variance = lambda weights,particles: np.average((particles - mean(weights,particles))**2, weights=weights, axis=0)
track = lambda t: 0.25*t**1.5
ROOM_SIZE = 20
landmarks = np.array([[-1, 2], [5, 10], [12,14], [18,21]])
INIT_NP = 800


def start_simulating():
    start_time = time.time()
    errors = []
    robot_positions = []
    robot = np.array([0.,0.])
    sensor_std_err = 0.1
    particles = (rand([INIT_NP,2])-.2) * ROOM_SIZE
    weights = uniform(INIT_NP)
    
    if __debug__:
        print('robot',robot)
        print('landmarks',landmarks)
        print(f'particlex{len(particles)} {particles[:5]}...')

        sns.scatterplot(x=landmarks[:,0],y=landmarks[:,1])
        plt.show()

        # [scalar], shape=(NL,)
        # Measure from each landmark to robot
        zs = measure(landmarks, robot, sensor_std_err)

        # [scalar], shape =(NP,)
        # Distance from particles to landmark
        distance = np.linalg.norm(particles[:, 0:2] - landmarks[0], axis=1)

        # [1d multimodal], shape=(NL,)
        # Likeihood from a landmark POV 
        # Norm around particle, if z is near particle then pdf(z) high
        likeihood = norm_distribution(distance, sensor_std_err)

    for ITER in range(0,20):
        # Robot move
        robot = [ITER+np.random.normal(scale=0.1), track(ITER)+np.random.normal(scale=0.1)]

        # Transition 
        v = np.random.normal(loc=1., scale=0.5, size=particles.shape)
        particles += v

        # likeihood on new observation
        zs = measure(landmarks, robot, sensor_std_err)
        NP = len(particles)
        likeihood = np.ones(NP) / NP
        for i, landmark in enumerate(landmarks):
            distance = np.linalg.norm(particles - landmark, axis=1)
            likeihood *= norm_distribution(distance, sensor_std_err).pdf(zs[i])
        
        weights*=likeihood
        weights += 1.e-300
        weights /= sum(weights) 

        # Calculate error
        error = norm_distance(mean(weights,particles)-robot)

        # Log
        errors.append(error)
        robot_positions.append(robot)

        if __debug__:
            print('ITER',ITER)
            print('robot',robot)
            print('mean', mean(weights,particles))
            print('delta weights', max(weights)-min(weights))
            print('effective',effective(weights))
            print('variance', variance(weights,particles))
        
        if __debug__:
            sns.scatterplot(x=particles[:,0],y=particles[:,1],hue=weights)
            sns.scatterplot(x=landmarks[:,0],y=landmarks[:,1], color='blue')
            sns.scatterplot(x=[robot[0]],y=[robot[1]], color='green')
            plt.show()

        # Resampling
        NP = len(particles)
        particles_index = np.random.choice(list(range(NP)), NP, p=weights)
        particles = particles[particles_index]
        weights = np.ones(NP) / NP

    if __debug__:
        robot_positions = np.array(robot_positions)
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('iter')

        ax1.set_ylabel('robot positions', color=color)
        ax1.plot(robot_positions[:,0],robot_positions[:,1], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('errors', color=color)
        ax2.plot(errors, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout() 
        plt.show()
    elapse = (time.time() - start_time)
    return errors, elapse

if __name__ == "__main__": 
    import pickle
    for t in [1,2,3,4]:
        COLLECTION = {
            'errors': [],
            'elapsed': []
        }

        
        NUM_RUN = 0
        while NUM_RUN < 500:
            print(f'SAMPLE {NUM_RUN}')
            try:
                errors, elapsed = start_simulating()   
                COLLECTION['errors'].append(errors)
                COLLECTION['elapsed'].append(elapsed)
                NUM_RUN+=1
            except:
                print(f"Error on run {NUM_RUN}, retry")

        print(np.average(COLLECTION['errors']))
        print(np.average(COLLECTION['elapsed']))

        
        with open(f'COLLETION_SIR_{INIT_NP}_{t}.pkl', 'wb') as sample_bak:
            pickle.dump(COLLECTION, sample_bak)
        

    
    

    
        
    

    

    