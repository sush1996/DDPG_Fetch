import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

#FetchReach Plots

env = 'FetchSlide-v1'

'''
ddpg = np.load('{}_DDPG.npy'.format(env))
ddpg_per = np.load('{}_DDPG_PER.npy'.format(env))
ddpg_her = np.load('{}_DDPG_HER.npy'.format(env))
'''


ddpg_her_per_0 = np.load('{}_DDPG_HER.npy'.format(env))
ddpg_her_per_5 = np.load('{}_DDPG_HER_PER.npy'.format(env))
ddpg_her_per_10 = np.load('{}_DDPG_HER_PER_10.npy'.format(env))
#ddpg_her_per = np.load('{}_DDPG_HER_PER.npy'.format(env))

'''
ep_n_0 = np.where(ddpg_her_per_0 == 1.0)[0][0]#[i for i in ddpg_her_per_0 if i == 1.0][0]
ep_n_2 = np.where(ddpg_her_per_2 == 1.0)[0][0]#[i for i in ddpg_her_per_2 if i == 1.0][0]
ep_n_4 = np.where(ddpg_her_per_4 == 1.0)[0][0]#[i for i in ddpg_her_per_4 if i == 1.0][0]
ep_n_6 = np.where(ddpg_her_per_6 == 1.0)[0][0]#[i for i in ddpg_her_per_6 if i == 1.0][0]
ep_n_8 = np.where(ddpg_her_per_8 == 1.0)[0][0]#[i for i in ddpg_her_per_8 if i == 1.0][0]
ep_n_10 = np.where(ddpg_her_per_10 == 1.0)[0][0]#[i for i in ddpg_her_per_10 if i == 1.0][0]


plt.bar(np.arange(len([0.0,0.2,0.4,0.6,0.8,1.0])), [ep_n_0, ep_n_2, ep_n_4, ep_n_6, ep_n_8, ep_n_10])
plt.show()
'''
n = range(50)#range(len(ddpg))
plt.plot(n, ddpg_her_per_0, label='alpha = 0.0')
#plt.plot(n, ddpg_her_per_2, label='alpha = 0.2')
plt.plot(n, ddpg_her_per_5, label='alpha = 0.5')
'''
plt.plot(n, ddpg_her_per_6, label='alpha = 0.6')
plt.plot(n, ddpg_her_per_8, label='alpha = 0.8')
'''
plt.plot(n, ddpg_her_per_10, label='alpha = 1.0')
plt.xlabel('Epoch')
plt.ylabel('Mean Success Rate')
plt.title('Mean Success Rate variation with alpha for DDPG+HER+PER on FetchPush-v1')
plt.legend()
plt.savefig('alpha_plots_fr')
plt.show()


'''
#Plot 1 : All Plots
plt.plot(n, ddpg, label='DDPG')
plt.plot(n, ddpg_per, label='DDPG + PER')
plt.plot(n, ddpg_her, label='DDPG + HER')
plt.plot(n, ddpg_her_per, label='DDPG + HER + PER')
plt.xlabel('Epoch')
plt.ylabel('Mean Success Rate')
plt.title('Mean Success Rate Plots for {}'.format(env))
plt.legend()
plt.savefig("{}_plots/all_plots".format(env))
plt.show()

#Plot 2 : DDPG and PER variants
plt.plot(n, ddpg, label='DDPG')
plt.plot(n, ddpg_per, label='DDPG + PER')
plt.xlabel('Epoch')
plt.ylabel('Mean Success Rate')
plt.title('Mean Success Rate Plots for {}'.format(env))
plt.legend()
plt.savefig("{}_plots/ddpg_per_plots".format(env))
plt.show()

#Plot 3: HER variants
plt.plot(n, ddpg_her, label='DDPG + HER')
plt.plot(n, ddpg_her_per, label='DDPG + HER + PER')
plt.xlabel('Epoch')
plt.ylabel('Mean Success Rate')
plt.title('Mean Success Rate Plots for {}'.format(env))
plt.legend()
plt.savefig("{}_plots/ddpg_her_per_plots".format(env))
plt.show()
'''