from UCBAlg import UCBalgorithm
import numpy as np

def binary_search(func,xmin,xmax,tol=1e-5):
    ''' func: function
    [xmin,xmax] is the interval where func is increasing
    returns x in [xmin, xmax] such that func(x) =~ 1 and xmin otherwise'''

    assert isinstance(xmin, float)
    assert isinstance(func(0.5*(xmax+xmin)), float)

    l = xmin
    r = xmax
    while abs(r-l) > tol:
        x = 0.5*(r + l)
        if func(x) > 1.0:
            r = x
        else:
            l = x

    x = 0.5*(r + l)
    return x


class UCBHyperparam:

    def __init__(self, m, burn_in = 1, confidence_radius = 2, 
        min_range = 0, max_range = 1, epsilon = 0):
        #self.hyperparam_list = hyperparam_list
        self.ucb_algorithm = UCBalgorithm(m, burn_in = 1, min_range = 0, max_range = 1, epsilon = epsilon)
        #self.discount_factor = discount_factor
        #self.forced_exploration_factor = forced_exploration_factor
        self.m = m
        self.confidence_radius = confidence_radius
        self.burn_in = burn_in
        self.T = 1

        #self.m = m# len(self.hyperparam_list)
        self.base_probas = np.ones(self.m)/(1.0*self.m)
        #self.importance_weighted_cum_rewards = np.zeros(self.m)
        #self.T = T
        #self.counter = 0
        #self.anytime = False
        #self.forced_exploration_factor = forced_exploration_factor
        #self.discount_factor = discount_factor
        # if self.anytime:
        #     self.T = 1


    def sample_base_index(self):
        index = self.ucb_algorithm.get_ucb_arm(self.confidence_radius)
        if self.T <= self.burn_in:
            self.base_probas = np.ones(self.m)/(1.0*self.m)
        else:
            self.base_probas = np.zeros(self.m)
            self.base_probas[index] = 1
        self.T += 1
        return index


    def get_distribution(self):
        return self.base_probas

        
    def update_distribution(self, arm_idx, reward, more_info = dict([])):        
        self.ucb_algorithm.update_arm_statistics(arm_idx, reward)
