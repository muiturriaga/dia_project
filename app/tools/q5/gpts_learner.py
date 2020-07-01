import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# Gaussian Process learner for Thompson Sampling approach
# arms = budget_allocations
# rewards = value of matching
class GptsLearner:
    def __init__(self, n_arms):
        self.t = 0      # episode number
        self.n_arms = n_arms
        self.arms = np.arange(start=0, stop=n_arms, step=1)  # index of budget alloc vector
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms)
        self.pulled_arms = []       # history of pulled arms
        self.collected_rewards = np.array([])       # history of rewards assigned to pulled arms
        alpha = 1.0       # noise measure for gp regressor
        kernel = C() * RBF()    # kernel = constant kernel + radial basis function
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=10)

    def update_obs(self, arm_idx, reward):
        self.pulled_arms.append(arm_idx)
        self.collected_rewards = np.append(self.collected_rewards, reward)

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)       # fit gp regression
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)   # predict for each arms
        self.sigmas = np.maximum(self.sigmas, 1e-2)     # force sigma to be higher than zero

    # update both observation and model, count episodes run
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_obs(pulled_arm, reward)
        self.update_model()

    # return index of pulled arm, highest value of sampled normal
    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        return np.argmax(sampled_values)
