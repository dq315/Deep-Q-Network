
# coding: utf-8

# In[1]:


import sys
import gym
import numpy as np
import random
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


class DeepQNetwork:
    def __init__(self, sess, n_actions, n_features, learning_rate, gamma, replace_target_iter):
        self.sess = sess
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = gamma
        self.learn_step_counter = 0
        self.replace_target_iter = replace_target_iter
        
        # inputs 
        self.S = tf.placeholder(tf.float32, [None, self.n_features], name='s')  
        self.S_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    
        self.A = tf.placeholder(tf.int32, [None, ], name='a')
        self.R = tf.placeholder(tf.float32, [None, ], name='r')
        self.D = tf.placeholder(tf.bool, [None, ], name='done')
        
        # variables
        with tf.variable_scope('net'):
            self.q = self._build_net(self.S, scope='eval', trainable=True)
            self.q_ = self._build_net(self.S_, scope='target', trainable=False)
        
        # parameters for target_net and evaluate_net
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net/eval')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net/target')
        
        # target_net hard replacement
        self.target_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        
        # step 1: compute q'(s',a') >> s' from batch, a' from target_net (amax)
        self.q_target = self.R if self.D is True else self.R + self.gamma * tf.reduce_max(self.q_, axis=1) 
        # step 2: compute q(s,a) >> s from batch, a from batch
        a_indices = tf.stack([tf.range(tf.shape(self.A)[0], dtype=tf.int32), self.A], axis=1)
        self.q_eval_wrt_a = tf.gather_nd(params=self.q, indices=a_indices)
        # step 3: compute td_error >> mse(q_eval, q_target)
        self.td_error = tf.losses.mean_squared_error(labels=(self.q_target), predictions=self.q_eval_wrt_a)
        # step 4: train >> Adam
        self.train = tf.train.AdamOptimizer(self.lr).minimize(self.td_error, var_list=self.e_params) 
        
    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 24, tf.nn.relu, 
                                  kernel_initializer=init_w, bias_initializer=init_b, 
                                  name='l1', trainable=trainable)
            net = tf.layers.dense(net, 24, tf.nn.relu, 
                                  kernel_initializer=init_w, bias_initializer=init_b, 
                                  name='l2', trainable=trainable)
            q = tf.layers.dense(net, self.n_actions, 
                                kernel_initializer=init_w, bias_initializer=init_b, 
                                name='q', trainable=trainable)
            return q

    def choose_action(self, s):
        if np.random.uniform() < epsilon:
            action = random.randrange(self.n_actions)
        else:
            actions_value = self.sess.run(self.q, feed_dict={self.S: s[np.newaxis, :]})
            action = np.argmax(actions_value)
        return action

    def learn(self, bs, ba, br, bs_, bd):
        # train evaluate_net and get loss
        self.sess.run(self.train, feed_dict={self.S: bs, self.A: ba, self.R: br, self.S_: bs_, self.D: bd})        
        l = self.sess.run(self.td_error, feed_dict={self.S: bs, self.A: ba, self.R: br, self.S_: bs_, self.D: bd})
        self.learn_step_counter += 1
        return l
    
    def update_target(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace)


# In[3]:


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.memory = np.zeros((capacity, dims))
        self.counter = 0

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, [a], [r], s_, [done]))
        index = self.counter % self.capacity  
        self.memory[index, :] = transition
        self.counter += 1

    def sample(self, n):
        indices = np.random.choice(np.minimum(self.capacity, self.counter), size=n)
        return self.memory[indices, :]


# In[ ]:


# In case of CartPole-v1, maximum length of episode is 500
env = gym.make('CartPole-v1')
#env = gym.make('MountainCar-v0')

# network parameters
n_actions = env.action_space.n
n_features = env.observation_space.shape[0]
print('n_features:', n_features)
print('n_actions:', n_actions)
learning_rate = 0.001
gamma = 0.99
replace_target_iter = 1

# exploration parameters
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999
        
# reply buffer parameters
capacity = 2000
train_start = 1000
batch_size = 64
dims = n_features * 2 + 1 + 1 + 1
print('batch dims:', dims)

EPISODES = 200

# dqn agent & reply buffer
sess = tf.Session()
dqn = DeepQNetwork(sess, n_actions, n_features, learning_rate, gamma, replace_target_iter)
M = Memory(capacity, dims)
sess.run(tf.global_variables_initializer())


# In[ ]:


episode_r = []
cost_his = []

for episode in range(EPISODES):
    done = False
    step_count = 0
    step_r = 0
    step_loss = 0

    # initial observation
    state = env.reset()

    while not done:
        
        step_count += 1

        # fresh env
        if True:
            env.render()

        # RL choose action based on state
        action = dqn.choose_action(state)

        # RL take action and get next state and reward
        next_state, reward, done, info = env.step(action)
        # if an action make the episode end, then gives penalty of -100
        reward = reward if not done or step_r == 499 else -100
        
        # store <s, a, r, s_>
        M.store_transition(state, action, reward, next_state, done)

        # train eval_net, epsilon decay, update target_net
        if M.counter >= train_start:

            # sample batch
            bt = M.sample(batch_size)
            bs = bt[:, :n_features]
            ba = bt[:, n_features: n_features + 1]
            br = bt[:, -n_features - 2: -n_features -1]
            bs_ = bt[:, -n_features -1: -1]
            bd = bt[:, -1:]           
            ba = np.hstack(ba) #(batch_size,)
            br = np.hstack(br) #(batch_size,)
            bd = np.hstack(bd) #(batch_size,)
            # train
            tra_results = dqn.learn(bs, ba, br, bs_, bd)
            step_loss += tra_results
            cost_his.append(step_loss)
            
            # epsilon decay per train
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            
            # update target network every C steps
            #dqn.update_target()

        # swap observation
        state = next_state

        # record step reward
        step_r += reward

        # break while loop when end of this episode
        if done:
            
            # update target network at the end of the epsiode game has a better performance
            dqn.update_target()
            
            step_r = step_r if step_r == 500 else step_r + 100
            episode_r.append(step_r)

            print('Episode:', episode+1, 
                  'Reward:', step_r,
                  'Loss:', '%.0f'%(step_loss/step_count),
                  'Epsilon:', '%.4f'%epsilon,)        


# In[ ]:


plt.figure(figsize=(25, 8))
plt.plot(np.arange(len(episode_r)), episode_r)
plt.ylabel('Reward',fontsize=20)
plt.xlabel('training episodes',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()


# In[ ]:


plt.figure(figsize=(25, 8))
plt.plot(np.arange(len(cost_his)), cost_his)
plt.ylabel('Loss',fontsize=20)
plt.xlabel('training steps',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

