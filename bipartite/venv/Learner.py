import numpy as npu


class Learner:
    def __init__(self, n_arm):
        self.n_arm = n_arm
        self.t = 0
        self.reward_per_arms = x = [[] for i in range(superarm)]
        self.collected_rewards = np.array([])
        self.beta_parameters = np.ones((n_arms, 2))

    #      should we use a beta parameter?
    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx

    # here we should choose what to pull
    def update_observation(self, pulled_super_arm, reward):
        self.rewards_per_super_arms[pulled_super_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        # the reward must be something between [0,1] so we can update beta parameter
        self.beta_parameters[pulled_arm, 0] += reward
        self.beta_parameters[pulled_arm, 1] += (1.0 - reward)
