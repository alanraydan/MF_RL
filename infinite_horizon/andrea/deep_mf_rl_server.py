#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:15:44 2020

@author: angiuli
"""

# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import time
import itertools
from tqdm import trange

import os

# import gym  #requires OpenAI gym installed
# env = gym.envs.make("MountainCarContinuous-v0") 

# tf.reset_default_graph()

# %% Initialization
num_episodes = 15000
modstep = 50
nr_ckp = int(num_episodes // modstep)

global vn_hidden1, pn_hidden1
vn_hidden1 = 64
pn_hidden1 = 64

# lr_actor = 0.000005  #set learning rates
# lr_critic = 0.00003

beta = 1

ratio = 100  # time points in [0,1]

delta_t = 1 / ratio

gamma = np.exp(-beta * delta_t)

T = 20

l = 1.5
d = 1 / 4
kk = 0.6
xi = 1 / 2
c5 = 1

sigma = 0.3

num_A = -beta + np.sqrt(beta**2 + 8 * (d + xi))
den_A = 4
A = num_A / den_A


def true_action_stat_mfg(state, mean):
    G1_stat = -2 * (d * l * mean + xi * kk) / (beta + 2 * A)
    a_true = -(2 * A * state + G1_stat)
    return a_true


def true_action_stat_mfc(state, mean):
    # B_stat=-2*(-1+2*l-l**2)*mean/(beta+2*A)
    B_stat = -2 * (mean * (l * d * (2 - l) - c5) + xi * kk) / (beta + 2 * A)
    # B_stat=-2 * ( mean * ( l * d * (2 - l) - c_5 ) + xi * kk)

    a_true_stat = -(2 * A * state + B_stat)
    return a_true_stat


def m_stat(beta, l, d, kk, xi):
    g2 = (-beta + np.sqrt(beta**2 + 8 * (xi + d))) / 4
    m = (kk * xi) / (g2 * (beta + 2 * g2) - d * l)
    return m


m_mfg = m_stat(beta, l, d, kk, xi)


def m_stat_c(beta, l, d, kk, xi, c_5):
    g2 = (-beta + np.sqrt(beta**2 + 8 * (xi + d))) / 4
    m = (kk * xi) / (g2 * (beta + 2 * g2) + c_5 - l * d * (2 - l))
    return m


m_mfc = m_stat_c(beta, l, d, kk, xi, c5)

border = 1
mean_initial = m_mfg
x_left = -2 * border + mean_initial
x_right = 2 * border + mean_initial
delta_x = np.sqrt(delta_t)
dim_x = int((x_right - x_left) / delta_x + 1)
x_space = np.linspace(x_left, x_right, dim_x)  # delta_x=0.01

act_comparison_mfg = true_action_stat_mfg(x_space, 0.8)
act_comparison_mfc = true_action_stat_mfc(x_space, m_mfc)

# sigma_stat=sigma**2/(4*A)

sigma_stat = 0.234


def new_state(state, action, noise):
    env_sigma = 0.3

    # ratio=100 # time points in [0,1]

    delta_t = 1 / ratio

    new_state = state + action * delta_t + env_sigma * np.sqrt(delta_t) * noise

    return new_state


def reward_step(s, act, m):
    r_val = 0.5 * act**2 + d * (s - l * m)**2 + xi * (s - kk)**2 + c5 * m**2

    rew = r_val * delta_t

    return rew


# %% Comments
# tf.reset_default_graph()
# input_dims = 1
# state_placeholder = tf.placeholder(tf.float32, [None, input_dims]) 

# def value_function(state):
#     # vn_hidden1 = 128  
#     n_hidden2 = 128
#     n_outputs = 1

#     with tf.variable_scope("value_network"):
#         # init_xavier = tf.contrib.layers.xavier_initializer()
#         # init_xavier = tf.initializers.glorot_uniform()

#         hidden1 = tf.layers.dense(state, vn_hidden1, tf.nn.elu)#, init_xavier)
#         V = tf.layers.dense(hidden1, n_outputs, None)#, init_xavier)

#         # hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu)#, init_xavier) 
#         # V = tf.layers.dense(hidden2, n_outputs, None)#, init_xavier)
#     return V


# def policy_network(state):
#     # pn_hidden1 = 64
#     n_hidden2 = 32
#     n_outputs = 1

#     with tf.variable_scope("policy_network"):
#         # init_xavier = tf.contrib.layers.xavier_initializer()
#         # init_xavier = tf.initializers.glorot_uniform()

#         hidden1 = tf.layers.dense(state, pn_hidden1, tf.nn.elu)#, init_xavier)
#         mu = tf.layers.dense(hidden1, n_outputs, None)#, init_xavier)
#         sigma = tf.layers.dense(hidden1, n_outputs, None)#, init_xavier)

#         # hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu)#, init_xavier)
#         # mu = tf.layers.dense(hidden2, n_outputs, None)#, init_xavier)
#         # sigma = tf.layers.dense(hidden2, n_outputs, None)#, init_xavier)

#         sigma = tf.nn.softplus(sigma) + 1e-5
#         # norm_dist = tf.contrib.distributions.Normal(mu, sigma)
#         norm_dist = tf.distributions.Normal(mu, sigma)
#         action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
#         # action_tf_var = tf.clip_by_value(
#         #     action_tf_var, env.action_space.low[0], 
#         #     env.action_space.high[0])
#     return action_tf_var, norm_dist

################################################################
# sample from state space for state normalization
# import sklearn
# import sklearn.preprocessing

# state_space_samples = np.array(
#     [env.observation_space.sample() for x in range(10000)])
# scaler = sklearn.preprocessing.StandardScaler()
# scaler.fit(state_space_samples)

# #function to normalize states
# def scale_state(state):                 #requires input shape=(2,)
#     scaled = scaler.transform([state])
#     return scaled                       #returns shape =(1,2)   
###################################################################

# lr_actor = 0.00002  #set learning rates
# lr_critic = 0.001


# lr_actor = 0.00001  #set learning rates
# lr_critic = 0.00056

# #promising
# lr_actor = 0.00001  #set learning rates
# lr_critic = 0.00003


# lr_actor = 0.000005  #set learning rates
# lr_critic = 0.00003

# lr_actor = 0.0001  #set learning rates
# lr_critic = 0.001

############## decay rate
# initial_learning_rate = 1e-30

# decay_rate = 1.30

# decay_steps = 100

# def decayed_learning_rate(step):
#   return initial_learning_rate * decay_rate ** (step / decay_steps)

# # global_step_placeholder = tf.placeholder(tf.float32) #tf.train.get_global_step()
# # # # step_placeholder = tf.placeholder(tf.float32)
# # lr_actor = decayed_learning_rate(global_step_placeholder)

##############
# %% Architecture
def training(lr_actor, lr_critic, omega_mu, run):
    # start = time.time()

    omega_mu = np.round(omega_mu, 2)

    tf.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    tf.reset_default_graph()
    input_dims = 1
    state_placeholder = tf.placeholder(tf.float32, [None, input_dims])

    def value_function(state):
        # vn_hidden1 = 128  
        # n_hidden2 = 128
        n_outputs = 1

        with tf.variable_scope("value_network"):
            # init_xavier = tf.contrib.layers.xavier_initializer()
            # init_xavier = tf.initializers.glorot_uniform()

            hidden1 = tf.layers.dense(state, vn_hidden1, tf.nn.elu)  # , init_xavier)
            V = tf.layers.dense(hidden1, n_outputs, None)  # , init_xavier)

            # hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu)#, init_xavier) 
            # V = tf.layers.dense(hidden2, n_outputs, None)#, init_xavier)
        return V

    def policy_network(state):
        # pn_hidden1 = 64
        # n_hidden2 = 32
        n_outputs = 1

        with tf.variable_scope("policy_network"):
            # init_xavier = tf.contrib.layers.xavier_initializer()
            # init_xavier = tf.initializers.glorot_uniform()

            hidden1 = tf.layers.dense(state, pn_hidden1, tf.nn.elu)  # , init_xavier)
            mu = tf.layers.dense(hidden1, n_outputs, None)  # , init_xavier)
            sigma = tf.layers.dense(hidden1, n_outputs, None)  # , init_xavier)

            # hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu)#, init_xavier)
            # mu = tf.layers.dense(hidden2, n_outputs, None)#, init_xavier)
            # sigma = tf.layers.dense(hidden2, n_outputs, None)#, init_xavier)

            sigma = tf.nn.softplus(sigma) + 1e-5
            # norm_dist = tf.contrib.distributions.Normal(mu, sigma)
            norm_dist = tf.distributions.Normal(mu, sigma)
            action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
            # action_tf_var = tf.clip_by_value(
            #     action_tf_var, env.action_space.low[0], 
            #     env.action_space.high[0])
        return action_tf_var, norm_dist, mu, sigma

    p_dir = './files'

    p_dir = './files/arch'

    name_test = p_dir + '/omega_' + str(omega_mu) + '_lr_a_' + str(lr_actor) + '_lr_c_' + str(
        lr_critic) + '_run_' + str(run)

    # define required placeholders
    action_placeholder = tf.placeholder(tf.float32)
    delta_placeholder = tf.placeholder(tf.float32)
    target_placeholder = tf.placeholder(tf.float32)

    action_tf_var, norm_dist, mu_var, sigma_var = policy_network(state_placeholder)
    V = value_function(state_placeholder)

    # define actor (policy) loss function
    loss_actor = tf.log(norm_dist.prob(action_placeholder) + 1e-5) * delta_placeholder

    training_op_actor = tf.train.AdamOptimizer(
        lr_actor, name='actor_optimizer').minimize(loss_actor)

    # tf.summary.scalar("learning_rate", lr_actor)
    # tf.summary.scalar("current_step", global_step)
    # tf.summary.scalar("loss", loss_actor)
    # define critic (state-value) loss function
    loss_critic = tf.reduce_mean(tf.squared_difference(
        tf.squeeze(V), target_placeholder))
    training_op_critic = tf.train.AdamOptimizer(
        lr_critic, name='critic_optimizer').minimize(loss_critic)
    ################################################################
    # Training loop
    # gamma = 0.99        #discount factor

    saver = tf.train.Saver(filename=name_test)
    print('true: ', act_comparison_mfg[15:26])
    start = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.import_meta_graph('my_test_model.meta')
        # saver.restore(sess,tf.train.latest_checkpoint('./'))

        # episode_history = []
        mean_e_mfg = []
        mean_e_mfc = []

        nstep = int(T / delta_t)

        mf_mu = np.zeros((nr_ckp, nstep))
        mf_sigma = np.zeros((nr_ckp, nstep))

        std_e = []
        # loss_a = []
        # loss_c = []
        episode = 0
        ckp = 0

        sample_mean = np.zeros((nstep, 2))

        sample_std = np.ones(nstep)

        sample_M = np.zeros(nstep)

        for episode in trange(num_episodes + 1):
            # receive initial state from E

            s_sigma = sample_std[-1]

            x0 = np.random.normal(sample_mean[-1, 1], s_sigma)

            bound = 3 * s_sigma + sample_mean[-1, 1]

            if x0 > bound:

                x0 = bound

            elif x0 < -bound:

                x0 = -bound

            state = np.reshape(x0, (-1, 1))  # env.reset()   # state.shape -> (2,)

            # reward_total = 0 
            steps = 0

            z = np.random.normal(0, 1, size=nstep)

            while (steps < nstep):

                # Sample action according to current policy
                # action.shape = (1,1)

                l_mu = 1 / (1 + episode)**omega_mu

                sample_mean[steps, 1] = sample_mean[steps, 1] + l_mu * (state - sample_mean[steps, 1])

                sample_M[steps] = sample_M[steps] + (state - sample_mean[steps, 0]) * (state - sample_mean[steps, 1])

                if episode > 0:
                    # episode = n - 1 where n is the number of sample points
                    sample_std[steps] = np.sqrt(sample_M[steps] / (episode))

                action = sess.run(action_tf_var, feed_dict={
                    state_placeholder: state})
                # Execute action and observe reward & next state from E
                # next_state shape=(2,)    
                # env.step() requires input shape = (1,)
                next_state = new_state(state, action, z[steps])

                reward = reward_step(state, action, sample_mean[steps, 1])

                # reward_total += reward
                # V_of_next_state.shape=(1,1)
                V_of_next_state = sess.run(V, feed_dict=
                {state_placeholder: next_state})
                # Set TD Target
                # target = r + gamma * V(next_state)
                target = reward + gamma * np.squeeze(V_of_next_state)

                # td_error = target - V(s)
                # needed to feed delta_placeholder in actor training
                td_error = target - np.squeeze(sess.run(V, feed_dict=
                {state_placeholder: state}))

                # Update actor by minimizing loss (Actor training)
                _, loss_actor_val = sess.run(
                    [training_op_actor, loss_actor],
                    feed_dict={action_placeholder: np.squeeze(action),
                               state_placeholder:  state,
                               delta_placeholder:  td_error})  # ,global_step_placeholder: episode*nstep+steps })
                # Update critic by minimizinf loss  (Critic training)
                _, loss_critic_val = sess.run(
                    [training_op_critic, loss_critic],
                    feed_dict={state_placeholder:  state,
                               target_placeholder: target})

                state = next_state

                sample_mean[steps, 0] = sample_mean[steps, 1]
                # end while
                # episode_history.append(reward_total)
                steps += 1

            if episode % modstep == 0 and episode > 0:
                print("Episode: {}".format(episode))

                # print('reward_total',reward_total)
                # add = np.reshape(sample_mean[:,1],(np.shape(sample_mean[:,1])[0],))
                mf_mu[ckp, :] = sample_mean[:, 1]
                # print(sample_mean[:3,1])
                # print(mf_mu)
                mf_sigma[ckp, :] = sample_std

                ckp += 1

                act = []
                act_std = []

                for x in x_space[15:26]:
                    mu_x, sigma_x = sess.run([mu_var, sigma_var], feed_dict={
                        state_placeholder: [[x]]})

                    act.append(mu_x)

                    act_std.append(sigma_x)

                # print('mean error actions in x[15:26]')
                act = np.asarray(act)
                act = np.reshape(act, (np.shape(act)[0],))

                mean_error_mfg = np.mean(np.abs(act_comparison_mfg[15:26] - act))
                mean_e_mfg.append(mean_error_mfg)

                # print(mean_error)
                mean_error_mfc = np.mean(np.abs(act_comparison_mfc[15:26] - act))
                mean_e_mfc.append(mean_error_mfc)

                # print('average/min/max std actions in x[15:26] ')
                std_error = np.mean(act_std)
                std_e.append(std_error)
                # print(std_error, min(act_std),max(act_std))

            # print("Episode: {}, Number of Steps : {}, Cumulative reward: {:0.2f}".format(
            # #     episode, steps, reward_total[0][0]))
            # if episode%modstep==0 and episode >0:
            #     print("Episode: {}".format(episode))
            #     act=[]
            #     mean =[]
            #     var =[]
            #     for x in x_space[15:26]:
            #         act_x=[]
            #         for j in range(500):
            #             action  = sess.run(action_tf_var, feed_dict={
            #                               state_placeholder: [[x]]})
            #             act_x.append(action)
            #         act.append(np.reshape(act_x,(500)))

            #         mean.append(np.mean(act_x))

            #         var.append(np.var(act_x))

            #     print(mean)
            if episode % 1000 == 0 and episode > 0:
                saver.save(sess, name_test)
                p_dir = './files/npy'

                np.save(p_dir + '/mf_mu_om_' + str(omega_mu) + '_run_' + str(run) + '.npy', mf_mu)
                np.save(p_dir + '/mf_sigma_om_' + str(omega_mu) + '_run_' + str(run) + '.npy', mf_sigma)
                np.save(p_dir + '/std_e_om_' + str(omega_mu) + '_run_' + str(run) + '.npy', std_e)
                np.save(p_dir + '/mean_e_mfg_om_' + str(omega_mu) + '_run_' + str(run) + '.npy', mean_e_mfg)
                np.save(p_dir + '/mean_e_mfc_om_' + str(omega_mu) + '_run_' + str(run) + '.npy', mean_e_mfc)

        print('Execution time:', - start + time.time())
        saver.save(sess, name_test)
        p_dir = './files/npy'

        np.save(p_dir + '/mf_mu_om_' + str(omega_mu) + '_run_' + str(run) + '.npy', mf_mu)
        np.save(p_dir + '/mf_sigma_om_' + str(omega_mu) + '_run_' + str(run) + '.npy', mf_sigma)
        np.save(p_dir + '/std_e_om_' + str(omega_mu) + '_run_' + str(run) + '.npy', std_e)
        np.save(p_dir + '/mean_e_mfg_om_' + str(omega_mu) + '_run_' + str(run) + '.npy', mean_e_mfg)
        np.save(p_dir + '/mean_e_mfc_om_' + str(omega_mu) + '_run_' + str(run) + '.npy', mean_e_mfc)

        # np.save(p_dir+'/loss_a_run_'+str(run)+'.npy',loss_a)
        # np.save(p_dir+'/loss_c_run_'+str(run)+'.npy',loss_c)
        # np.save(p_dir+'/tot_epis_run_'+str(run)+'.npy',episode)
        # end = time.time()
        # print(end - start)
        # if np.mean(episode_history[-100:]) > 90 and len(episode_history) >= 101:
        #     print("****************Solved***************")
        #     print("Mean cumulative reward over 100 episodes:{:0.2f}" .format(
        #         np.mean(episode_history[-100:])))Actor


# %% Parallel inizialization


# lr_v = [ 10**-6]
# k = 1
# j=0
# for x in range(7):
#     tot = j +k
#     lr_v.append(10**-6 * 5**k * 2** j)
#     if tot%2==0:
#         k+=1
#     else:
#         j+=1

# lr_values = [[x,y] for x in lr_v for y in lr_v if x<y ]
lr_actor = 5e-06

lr_critic = 1e-05

# omega_mu = 0.85

n_run = 4
# omega=np.linspace(0.5,0.6,2)
# omega=np.linspace(0.7,0.8,2)
# omega=np.linspace(0.9,1,2)

omega = [0.8]
runs = [x for x in range(n_run)]

par_set = list(itertools.product(omega, runs))

# %% Parallel execution
p_dir = './files'
if not os.path.exists(p_dir):
    os.mkdir(p_dir)
p_dir = './files/arch'
if not os.path.exists(p_dir):
    os.mkdir(p_dir)

p_dir = './files/npy'
if not os.path.exists(p_dir):
    os.mkdir(p_dir)

start = time.time()

if __name__ == "__main__":
    Parallel(n_jobs=n_run)(delayed(training)(lr_actor, lr_critic, par[0], par[1]) for par in par_set)

end = time.time()
print(end - start)

# %% Plots

plot_dir = './plots'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

plot_dir = './plots/controls'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

plot_dir = './plots/std_errors'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

plot_dir = './plots/mean_errors_mfg'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

plot_dir = './plots/mean_errors_mfc'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

plot_dir = './plots/mf_mean'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

plot_dir = './plots/mf_std'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

tf.reset_default_graph()
input_dims = 1
state_placeholder = tf.placeholder(tf.float32, [None, input_dims])


def value_function(state):
    # vn_hidden1 = 128  
    # n_hidden2 = 128
    n_outputs = 1

    with tf.variable_scope("value_network"):
        # init_xavier = tf.contrib.layers.xavier_initializer()
        # init_xavier = tf.initializers.glorot_uniform()

        hidden1 = tf.layers.dense(state, vn_hidden1, tf.nn.elu)  # , init_xavier)
        V = tf.layers.dense(hidden1, n_outputs, None)  # , init_xavier)

        # hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu)#, init_xavier) 
        # V = tf.layers.dense(hidden2, n_outputs, None)#, init_xavier)
    return V


def policy_network(state):
    # pn_hidden1 = 64
    # n_hidden2 = 32
    n_outputs = 1

    with tf.variable_scope("policy_network"):
        # init_xavier = tf.contrib.layers.xavier_initializer()
        # init_xavier = tf.initializers.glorot_uniform()

        hidden1 = tf.layers.dense(state, pn_hidden1, tf.nn.elu)  # , init_xavier)
        mu = tf.layers.dense(hidden1, n_outputs, None)  # , init_xavier)
        sigma = tf.layers.dense(hidden1, n_outputs, None)  # , init_xavier)

        # hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu)#, init_xavier)
        # mu = tf.layers.dense(hidden2, n_outputs, None)#, init_xavier)
        # sigma = tf.layers.dense(hidden2, n_outputs, None)#, init_xavier)

        sigma = tf.nn.softplus(sigma) + 1e-5
        # norm_dist = tf.contrib.distributions.Normal(mu, sigma)
        norm_dist = tf.distributions.Normal(mu, sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
        # action_tf_var = tf.clip_by_value(
        #     action_tf_var, env.action_space.low[0], 
        #     env.action_space.high[0])
    return action_tf_var, norm_dist, mu, sigma


action_placeholder = tf.placeholder(tf.float32)
delta_placeholder = tf.placeholder(tf.float32)
target_placeholder = tf.placeholder(tf.float32)

action_tf_var, norm_dist, mu_var, sigma_var = policy_network(state_placeholder)
V = value_function(state_placeholder)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

print('Actor hidden nodes: %s ' % (pn_hidden1))
print('Critic hidden nodes: %s ' % (vn_hidden1))
# print ('Actor learning rate: %s ' %(lr_actor))
# print ('Critic learning rate: %s ' %(lr_critic))
print('Training episodes: %s' % (num_episodes))

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

border = 1
mean_initial = 0.8
x_left = -2 * border + mean_initial
x_right = 2 * border + mean_initial
delta_x = np.sqrt(delta_t)
dim_x = int((x_right - x_left) / delta_x + 1)
x_space = np.linspace(x_left, x_right, dim_x)  # delta_x=0.01

act_comparison_mfg = true_action_stat_mfg(x_space, 0.8)

lr_a = lr_actor

lr_c = lr_critic

for par in par_set:

    omega_mu = np.round(par[0], 2)

    run = par[1]
    # index=1
    # lr = lr_values[index]

    print(str() + '_lr_c_' + str(lr_c))

    # Opening a file 
    file1 = open('./files/arch/checkpoint', 'w')
    L = [
        "model_checkpoint_path: \"omega_" + str(omega_mu) + "_lr_a_" + str(lr_a) + "_lr_c_" + str(lr_c) + "_run_" + str(
            run) + "\"\n",
        "all_model_checkpoint_paths: \"omega_" + str(omega_mu) + "_lr_a_" + str(lr_a) + "_lr_c_" + str(
            lr_c) + "_run_" + str(run) + "\" "]
    # L = ["This is Delhi \n", "This is Paris \n", "This is London \n"] 
    # s = "Hello\n"

    # Writing a string to file 
    # file1.write(s) 

    # Writing multiple strings 
    # at a time 
    file1.writelines(L)

    # Closing file 
    file1.close()

    # Checking if the data is 
    # # written to file or not 
    # file1 = open('myfile.txt', 'r') 
    # print(file1.read()) 
    # file1.close()

    sample_size = 500
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        # test_lr_a_1e-06_lr_c_4.9999999999999996e-06
        # test_lr_a_4.9999999999999996e-06_lr_c_9.999999999999999e-06
        saver = tf.train.import_meta_graph(
            './files/arch/omega_' + str(omega_mu) + '_lr_a_' + str(lr_a) + '_lr_c_' + str(lr_c) + '_run_' + str(
                run) + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./files/arch/'))

        ###### New plotting

        act_mu = []
        act_std = []

        for x in x_space:
            mu_x, sigma_x = sess.run([mu_var, sigma_var], feed_dict={
                state_placeholder: [[x]]})
            act_mu.append(np.reshape(mu_x, 1))

            act_std.append(np.reshape(sigma_x, 1))

        mean = np.asarray(act_mu)
        mean = np.reshape(mean, (np.shape(mean)[0],))

        act_std = np.asarray(act_std)
        act_std = np.reshape(act_std, (np.shape(act_std)[0],))

    print(mean[15:26])
    print('run', run, 'std err', np.mean(act_std))
    plt.figure()
    plt.plot(x_space, mean, color='orange', label='learned')
    plt.fill_between(x_space, mean - act_std, mean + act_std)

    plt.plot(x_space, act_comparison_mfg, color='blue', label='MFG')
    plt.plot(x_space, act_comparison_mfc, color='red', label='MFC')

    p_dir = './files/npy'
    plot_dir = './plots'

    plt.xlabel('state x')
    plt.ylabel('$\\alpha(x)$')
    plt.title('Control function')
    plt.legend()
    plt.savefig(
        plot_dir + '/controls/fill_controls_lr_a_' + str(lr_a) + '_lr_c_' + str(lr_c) + '_run_' + str(run) + '.png')

    plt.figure()
    plt.plot(x_space, mean, color='orange', label='learned')
    # plt.fill_between(x_space,mean-act_std,mean+act_std)
    plt.plot(x_space, act_comparison_mfg, color='blue', label='MFG')
    plt.plot(x_space, act_comparison_mfc, color='red', label='MFC')

    p_dir = './files/npy'
    plot_dir = './plots'

    plt.xlabel('state x')
    plt.ylabel('$\\alpha(x)$')
    plt.title('Control function')
    plt.legend()
    plt.savefig(plot_dir + '/controls/controls_lr_a_' + str(lr_a) + '_lr_c_' + str(lr_c) + '_run_' + str(run) + '.png')

    mf_mu = np.load(p_dir + '/mf_mu_om_' + str(omega_mu) + '_run_' + str(run) + '.npy')
    mf_sigma = np.load(p_dir + '/mf_sigma_om_' + str(omega_mu) + '_run_' + str(run) + '.npy')

    std_e = np.load(p_dir + '/std_e_om_' + str(omega_mu) + '_run_' + str(run) + '.npy')
    mean_e_mfg = np.load(p_dir + '/mean_e_mfg_om_' + str(omega_mu) + '_run_' + str(run) + '.npy')
    mean_e_mfc = np.load(p_dir + '/mean_e_mfc_om_' + str(omega_mu) + '_run_' + str(run) + '.npy')

    plt.figure()
    plt.plot(std_e)
    plt.xlabel('checkpoints')
    plt.ylabel('std error')
    plt.title('Std err in %s epis, lr_a = %s, lr_c = %s, run = %s' % (num_episodes, lr_a, lr_c, run))
    # plt.legend()
    plt.savefig(
        plot_dir + '/std_errors/std_om_' + str(omega_mu) + '_lr_a_' + str(lr_a) + '_lr_c_' + str(lr_c) + '_run_' + str(
            run) + '.png')

    plt.figure()
    plt.plot(mean_e_mfg)
    plt.xlabel('checkpoints ')
    plt.ylabel('mean error')
    plt.title('MFG-Err om = %s, l_a = %s, l_c = %s, run = %s' % (omega_mu, lr_a, lr_c, run))
    plt.savefig(plot_dir + '/mean_errors_mfg/mfg_error_om_' + str(omega_mu) + '_l_a_' + str(lr_a) + '_l_c_' + str(
        lr_c) + '_run_' + str(run) + '.png')

    plt.figure()
    plt.plot(mean_e_mfc)
    plt.xlabel('checkpoints ')
    plt.ylabel('mean error')
    plt.title('MFC-Err om = %s, l_a = %s, l_c = %s, run = %s' % (omega_mu, lr_a, lr_c, run))
    plt.savefig(plot_dir + '/mean_errors_mfc/mfc_error_om_' + str(omega_mu) + '_l_a_' + str(lr_a) + '_l_c_' + str(
        lr_c) + '_run_' + str(run) + '.png')

    plt.figure()
    plt.plot(mf_mu[:, -1], label='learned')
    plt.plot(m_mfg * np.ones(nr_ckp), label='MFG')
    plt.plot(m_mfc * np.ones(nr_ckp), label='MFC')
    plt.title('Evolution population mean')
    plt.legend()
    plt.savefig('./plots/mf_mean/mf_mean_om_' + str(omega_mu) + '_run_' + str(run) + '.png')

    plt.figure()
    plt.plot(mf_sigma[:, -1], label='learned')
    plt.plot(0.234 * np.ones(nr_ckp), label='MFG/C')
    plt.title('Evolution population std')
    plt.legend()
    plt.savefig('./plots/mf_std/mf_std_om_' + str(omega_mu) + '_run_' + str(run) + '.png')
