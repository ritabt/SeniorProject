import numpy as np
from matplotlib import pyplot as plt

class Plot():
    def __init__(self):
        return

    def smooth(selft, scalars, weight):  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)                        # Save it
            last = smoothed_val                                  # Anchor the last smoothed value
            
        return smoothed

    # compute stats
    def stats(self, r_per_episode, R, cum_R, cum_R_episodes, 
            cum_loss_per_episode, cum_loss, cum_loss_episodes):
        r_per_episode = np.append(r_per_episode, R) # store reward per episode
        cum_R_episodes += R
        cum_R = np.append(cum_R, cum_R_episodes) # store cumulative reward of all episodes

        cum_loss_episodes += cum_loss_per_episode
        cum_loss = np.append(cum_loss, cum_loss_episodes) # store cumulative loss of all episodes

        return (r_per_episode, cum_R_episodes, cum_R, cum_loss_episodes, cum_loss)

    # plot performance
    def plot_charts(self, values, y_label, epsilon=None):
        fig = plt.figure(figsize=(10,5))
        plt.title("DQN performance")
        plt.xlabel("Episode")
        plt.ylabel(y_label)

        np_out_file = "values/" + y_label + "_plot.png"
        np.save(np_out_file, values)

        if epsilon is not None:
            epsilon = epsilon*100
            np_out_file = "values/" + y_label + "_epsilon.png"
            np.save(np_out_file, values)
            plt.plot(epsilon, 'g', label="epsilon percentage")
            plt.plot(values, 'b', label="reward per episode")
            values = self.smooth(values, 0.99)
            plt.plot(values, 'r', label="smoothed reward per episode")
            # ax.plot(x_labels, smooth(train_data, .9), x_labels, train_data)
            plt.legend(loc='upper right')
        else:
            plt.plot(values)
        out_file = "plots/" + y_label + "_plot.png"
        plt.savefig(out_file) 
        plt.close()

    def display(self, r_per_episode, cum_R, cum_loss, max_episodes, epsilon, lr):
        self.plot_charts(r_per_episode, "Reward_lr_"+str(lr), epsilon)
        self.plot_charts(cum_R, "cumulative_reward_lr_"+str(lr))
        self.plot_charts(cum_loss, "cumulative_loss_lr_"+str(lr))

        avg_r = np.sum(r_per_episode) / max_episodes
        print("avg_r", avg_r)  

        avg_loss = np.sum(cum_loss) / max_episodes
        print("avg_loss", avg_loss)  