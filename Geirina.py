"""
__author__ = 'jiajun,yanzan'
# Copyright (C) 2018 - 2020 GEIRI North America
# Authors: jiajaun <jiajun.duan@geirina.net>
"""

from grid2op.Agent import DoNothingAgent


import time
import datetime
import warnings
import copy
import itertools

import sys
import csv

import os
import json
import math
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from grid2op.Parameters import Parameters
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from l2rpn_baselines.Geirina.DDQN_NN import DeepQNetworkDueling
from l2rpn_baselines.Geirina.prioritized_memory import Memory


class Geirina(DoNothingAgent):
    """
    Do nothing agent of grid2op, as a lowerbond baseline for l2rpn competition.

    Note that a Baseline should always somehow inherit from :class:`grid2op.Agent.BaseAgent`.

    It serves as a template agent to explain how a baseline can be built.

    As opposed to bare grid2op Agent, baselines have 3 more methods:
    - :func:`Template.load`: to load the agent, if applicable
    - :func:`Template.save`: to save the agent, if applicable
    - :func:`Template.train`: to train the agent, if applicable

    The method :func:`Template.reset` is already present in grid2op but is emphasized here. It is called
    by a runner at the beginning of each episode with the first observation.

    The method :func:`Template.act` is also present in grid2op, of course. It the main method of the baseline,
    that receives an observation (and a reward and flag that says if an episode is over or not) an return a valid
    action.

    **NB** the "real" instance of environment on which the baseline will be evaluated will be built AFTER the creation
    of the baseline. The parameters of the real environment on which the baseline will be assessed will belong to the
    same class than the argument used by the baseline. This means that if a baseline is built with a grid2op
    environment "env", this environment will not be modified in any manner, all it's internal variable will not
    change etc. This is done to prevent cheating.

    """
    def __init__(self,
                 action_space,
                 observation_space,
                 name,
                 save_path, 
                 learning_rate=1e-4,
                 gamma=0.99,
                 replace_target_iter=300,
                 replay_memory_size=300,
                 PER_alpha=0.6,
                 PER_beta=0.4,
                 batch_size = 32,
                 epsilon_start=0.7,
                 epsilon_end=0.001,
                 verbose_per_episode=1,
                 seed=22,
                 verbose=False,
                 data_dir=".",
                 model_name="geirina",
                 restore=False
                 ):
        DoNothingAgent.__init__(self, action_space)
        self.do_nothing = self.action_space()
        self.name = name
        self.agent_action_space = action_space
        self.save_path = os.path.abspath(save_path)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.model_name = model_name
        self.restore = restore

        self.id = set()  
        self.line_count = [0 for i in range(20)]      

        # training params
        self.n_features = np.sum(observation_space.shape)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter
        self.replay_memory_size = replay_memory_size
        self.PER_alpha = PER_alpha
        self.PER_beta = PER_beta
        self.batch_size = batch_size
        self.verbose_per_episode = verbose_per_episode
        self.verbose = verbose
        # state control
        self.seed = seed
       # np.random.seed(seed)
        tf.set_random_seed(seed)

        # control epsilon
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        

        # Directory
        # if you provide that, make sure to clearly indicate it
        # and to throw an error when it's not compatible with the environment used !
        self.data_dir = os.path.abspath(data_dir)
        action_to_index_dir = os.path.join(self.data_dir, 'action_to_index_jd1.npy')
        action_dir = os.path.join(self.data_dir, 'all_actions_jd1.npy')       

        # load actions
        self.action_space = np.load(action_dir, allow_pickle=True)
        #print("self action space", self.action_space)
        #print("self action space shape", self.action_space.shape)
        self.n_actions = self.action_space.shape[0]
        #print("self n_actions", self.n_actions)
        self.action_to_index = np.load(action_to_index_dir, allow_pickle=True)
        #print("self action to index", self.action_to_index)

        # build graph
        self.graph = tf.Graph()

        with self.graph.as_default():
          self.dqn_main = DeepQNetworkDueling(learning_rate=learning_rate,
                                              n_actions=self.n_actions,
                                              n_state=self.n_features,
                                                scope='dqn_main')
          self.dqn_target = DeepQNetworkDueling(learning_rate=learning_rate,
                                                n_actions=self.n_actions,
                                              n_state=self.n_features,
                                                scope='dqn_target')
          # Saver
          self.saver = tf.train.Saver()
          self.sess = tf.Session(graph=self.graph)
          self.sess.run(tf.global_variables_initializer())

        # The replay memory
        self.replay_memory = Memory(self.replay_memory_size, PER_alpha, PER_beta)

        # hard copy
        self.params_copy_hard = [target.assign(main) for main, target in zip(
            self.dqn_main.params_train, self.dqn_target.params_train)]

        # summary
        self.timestamp = datetime.datetime.now().strftime('%m-%d-%H-%M')
        self.episode_score_history = [0] 


    def action_transformation(self, action_array, name):
        # Transfer the l2rpn action array to a dict of grid2op
        action_dict = {}
        obj_id_dict = {}

        if name == 'l2rpn_case14_sandbox':
          self.line_map = [0,1,2,3,4,5,6,15,16,17,9,8,7,18,19,11,10,12,13,14]
          load_map = [0,1,2,3,4,5,6,7,8,9,10]
        elif name == "l2rpn_2019":
          self.line_map = [i for i in range(20)]
          load_map = [0,1,2,3,4,5,6,7,8,9,10]          
        else:
          self.line_map = [0,1,7,8,9,10,11,15,16,17,14,13,12,18,19,3,2,4,5,6]
          load_map = [0,1,3,4,5,6,7,8,9,10,2]


        # generator
        offset = 0
        switches_gen_array = action_array[:5]
        switch_gen_id_list = list(np.where(switches_gen_array == 1)[0])
        if switch_gen_id_list:
          obj_id_dict["generators_id"] = switch_gen_id_list

        # load
        offset += 5
        switches_load_array = action_array[offset:offset+11]
        switch_load_id_list = list(np.where(switches_load_array == 1)[0])
        if switch_load_id_list:
          obj_id_dict["loads_id"] = [load_map[int(i)] for i in switch_load_id_list]
          #obj_id_dict["loads_id"] = [int(i) for i in switch_load_id_list]

        # line ox
        offset += 11
        switches_lines_or_array = action_array[offset:offset+20]
        switch_or_id_list = list(np.where(switches_lines_or_array == 1)[0])

          #print(obj_id_dict["lines_or_id"])
        # line ex
        offset += 20
        switches_lines_ex_array = action_array[offset:offset+20]
        switch_ex_id_list = list(np.where(switches_lines_ex_array == 1)[0])
        
        if name == 'l2rpn_case14_sandbox':
          if 14 in switch_or_id_list:
            switch_or_id_list.remove(14)
            switch_ex_id_list.append(14) 
          elif 14 in switch_ex_id_list:
            switch_ex_id_list.remove(14)
            switch_or_id_list.append(14) 

        if switch_or_id_list:
          obj_id_dict["lines_or_id"] = [self.line_map[int(i)] for i in switch_or_id_list]
          #obj_id_dict["lines_or_id"] = [int(i) for i in switch_or_id_list]

        if switch_ex_id_list:
          obj_id_dict["lines_ex_id"] = [self.line_map[int(i)] for i in switch_ex_id_list]
        #  obj_id_dict["lines_ex_id"] = [int(i) for i in switch_ex_id_list]

        if len(obj_id_dict) != 0:
          action_dict["change_bus"] = obj_id_dict

        # line status
        switches_lines_status_array = action_array[-20:]
        switch_lines_id_list = list(np.where(switches_lines_status_array == 1)[0])
        if len(switch_lines_id_list) != 0:
          action_dict["change_line_status"] = switch_lines_id_list
        topo_action = self.agent_action_space({})
        if len(action_dict) != 0:
          try:
            topo_action = self.agent_action_space(action_dict)
          except AmbiguousAction:
            print("This Is AmbiguousAction: You can only change line status with int or boolean numpy array vector.")
        return  topo_action

    def normalize(self, state):
    	#normalize the state
        for i, x in enumerate(state[0]):
            if x >= 1e5: state[0][i] /= 1e4
            elif x >= 1e4 and x < 1e5 : state[0][i] /= 1e3
            elif x >= 1e3 and x < 1e4 : state[0][i] /= 1e2

        return(state)

    def train(self, env,
              iterations,
              logdir = "logs-train", 
              isco=''):

        self.env = env
#        self.agent_action_space = self.env.action_space
        time_start = time.time()
        self.epsilon_decay_steps = 3 * iterations // 150
        epsilons = np.append(np.linspace(self.epsilon_start, 0.2, int(self.epsilon_decay_steps/5)), np.linspace(0.19, self.epsilon_end, int(self.epsilon_decay_steps*4/5)))
        print(len(epsilons))
        count_action = {}
        

        # session config
        config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True)
        # store result
        result_columns = ["Episode", "Totle Timestep", "Total Score"] 
        with open("result_summary.csv",'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result_columns)

        # initialize sees
        with tf.Session(config=config, graph=self.graph) as sess:
          sess.run(tf.global_variables_initializer())
          if self.restore:
            self.saver.restore(sess, (self.save_path + '/{}.ckpt'.format(self.model_name)))

          time_step = 0
          loss = 0
          episode = 0

          print("Start training...")
          sys.stdout.flush()

          done = False
          flag_next_chronic = False
          

          # start training for data in chronics
          while time_step < iterations:
            self.id = set()
            self.id.clear()

            state = self.env.reset().to_vect()
            #initialize observation
            self.observation = self.env.helper_observation(self.env)
            obs = None
            state = [0 if i != i else i for i in state]
            state = np.array(state).reshape((-1, self.n_features))
            line_count = [0 for i in range(20)]
            lines_status_list_bool = []

            for step in itertools.count():
              reward_tot = 0
              print("Epsilons is ", epsilons[min(time_step, self.epsilon_decay_steps - 1)], epsilons[self.epsilon_decay_steps - 1] )
              
              # update the target estimator
              if time_step % self.replace_target_iter == 0:
                sess.run([self.params_copy_hard])

              self.action_next = np.zeros(76)
              if step >=1:             
                  #count the cooldown time for broken line
                  for i, v in enumerate(lines_status_list_bool):
                    if v <= 0:
                      self.id.add(i)
                      line_count[i] +=1
                      if line_count[i] >10: line_count[i] = 10
                    if v > 0:
                        try: 
                            line_count[i] = 0
                            self.id.remove(i)
                        except: pass
                      
                  idlist = [i for i in self.id]
                  action_transf_dict = {}

                  #if cooldown time is reached, close the line and check validation
                  if len(idlist) > 0:   
                    print("Broken_Line", line_count)               
                    for i, x in enumerate(line_count):
                        if x == 10 and (False in lines_status_list_bool): 
                            set_status = self.agent_action_space.get_set_line_status_vect()
                            for o, e in [(1,1),(1,2),(2,1),(2,2)]:
                                action_transf_dict = self.agent_action_space.reconnect_powerline(line_id=i,bus_or=o,bus_ex=e)  
                                obs_simulate, reward_simulate, done_simulate, _ = self.observation.simulate(action_transf_dict)
                                print(action_transf_dict, done_simulate)
                                if not done_simulate: break                 

                            self.action_next[56+self.line_map.index(idlist[0])] = 1.0
                            action = self.action_to_index.item().get(tuple(self.action_next)) 
                            self.id.clear()
                            line_count[i] = 0
                            break

              #if not close line or game over in simulation, call agent
              if 1.0 not in self.action_next or done_simulate:
                  # choose action
                  action, q_predictions = self.dqn_main.act(
                      sess, self.normalize(state), epsilons[min(time_step, self.epsilon_decay_steps - 1)])
                  # guide to do nothning (action = 155)
                  if np.random.uniform() > epsilons[min(time_step, self.epsilon_decay_steps - 1)]: action = 155
                  print("training action index", action)
                  # check loadflow
                  thermal_limits = np.abs(self.env.backend.get_thermal_limit())

                  # check overflow
                  action_is_legal = True
                  has_danger = False
                  has_overflow = False              
                  action_array = self.action_space[action]
                  action_transf_dict = self.action_transformation(action_array, self.env.name)

                  obs_simulate, reward_simulate, done_simulate, infor = self.observation.simulate(action_transf_dict)
                  #print("first simulate score", reward_simulate)

                  if reward_simulate is None:
                    has_danger = True
                  elif obs is not None: 
                    has_overflow = any(obs.rho>1.02)

                  # if game over, run more simulation
                  if done_simulate or reward_simulate is None or has_danger or has_overflow:
                    # if has overflow. try all actions
                    if has_overflow:
                      print('has overflow !!!!!!!!!!!!!!!!!!!!!!!!!')
                    if has_danger:
                      print('has danger !!!!!!!!!!!!!!!!!!!!!!!!!!!')

                    top_actions = np.argsort(q_predictions)[-1: -10: -1].tolist()
                    chosen_action = 155
                    max_score = float('-inf')
                    for action in tuple(top_actions):           
                      action_class_helper = self.env.helper_action_env
                      action_array = self.action_space[action]
                      action_transf_dict = self.action_transformation(action_array, self.env.name)
                      action_is_legal = action_class_helper.legal_action(action_transf_dict, self.env)
                      if not action_is_legal:
                        #print("illegal!!!")
                        continue
                      else:
                        obs_simulate, reward_simulate, done_simulate, _= self.observation.simulate(action_transf_dict)
                        #print("simulate score", reward_simulate, action)
                      if obs_simulate is None:
                        continue
                      else:
                        obs_dict = obs_simulate.to_dict()
                        lineflow_simulate_ratio = obs_dict["rho"]

                        lineflow_simulate_ratio = [round(x, 4) for x in lineflow_simulate_ratio]

                        # seperate big line and small line
                        has_danger = False
                        for ratio, limit in zip(lineflow_simulate_ratio, thermal_limits):
                          if ratio > 1.05:
                            has_danger = True

                        if not done_simulate and reward_simulate > max_score and not has_danger:
                          max_score = reward_simulate
                          chosen_action = action
                          print('current best action: {}, score: {:.4f}'.format(chosen_action, reward_simulate))

                    # chosen action
                    action = chosen_action
              
                  # count action
                  count_action[action] = count_action.get(action, 0) + 1

                  # take a step
                  action_array = self.action_space[action]
                  action_transf_dict = self.action_transformation(action_array, self.env.name)
              #print("applied action is ", action_transf_dict)              
              obs, reward, done, infos = self.env.step(action_transf_dict)
              self.observation = obs
              #self.observation.update(self.env)

             # record data
              next_state_dict = obs.to_dict()
              lines_status_list_bool = next_state_dict["rho"]

              state_list = [0 if i != i else i for i in self.env.get_obs().to_vect()] 
              if done:
                next_state = state
                reward = -15
              else:
                next_state = np.array(state_list).reshape((-1, self.n_features))

              if reward >= 100: reward_tot = reward / 750
              else: reward_tot = reward / 15

              # record
              self.episode_score_history[episode] += reward

              # Save transition to replay memory
              if done:
                # if done: store more
                for i in range(4):
                  self.replay_memory.store(
                      [self.normalize(state), action, reward_tot, self.normalize(next_state), done])
              else:
                self.replay_memory.store(
                      [self.normalize(state), action, reward_tot, self.normalize(next_state), done])
                

              # learn
              if time_step > self.replay_memory_size and time_step % 5 == 0:
                
              # Sample a minibatch from the replay memory
                tree_idx, batch_samples, IS_weights = self.replay_memory.sample(self.batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(
                    np.array, zip(*batch_samples))
                states_batch = states_batch.reshape((-1, self.n_features))
                next_states_batch = next_states_batch.reshape(
                    (-1, self.n_features))

                # Calculate q targets
                q_values_next = self.dqn_target.predict(sess, next_states_batch)
                q_values_next = np.array(q_values_next[0])
                targets_batch = reward_batch + \
                    np.invert(done_batch).astype(np.float32) * \
                    self.gamma * np.amax(q_values_next, axis=1)

                # Perform gradient descent update
                loss, abs_TD_errors = self.dqn_main.update(sess, states_batch, targets_batch,
                                                           action_batch.reshape((-1, 1)), IS_weights)
                # Update priority
                self.replay_memory.batch_update(tree_idx, abs_TD_errors)

                # verbose step summary
              if episode % self.verbose_per_episode == 0 and (time_step + 1) % 1 == 0:
                print("episode: {}, step: {},  action: {}, loss: {:4f},  reward: {:4f}, score: {:.4f}, tot: {:.4f}\n".
                      format(episode + 1, step + 1, action, loss, reward, self.episode_score_history[episode], reward_tot))
                sys.stdout.flush()

              # update state
              state = next_state
              time_step += 1

              #if done or time_step > total_train_step or flag_next_chronic:
              if done: 
                break

            episode += 1
            self.episode_score_history.append(0)

            # save model per episode
            if episode >= 50:
              self.saver.save(sess, os.path.join(self.save_path, '{}.ckpt'.format(self.model_name)))
              print('Model Saved!')

            # verbose episode summary
            print("\nepisode: {}, mean_score: {:4f}, sum_score: {:4f}\n".
                      format(episode + 1, self.episode_score_history[episode] / (step + 1), self.episode_score_history[episode]))
            print("action count: {}\n".format(sorted(count_action.items())))
            with open("result_summary.csv", 'a',newline='') as f:
                result = [episode + 1]
                result.append(step+1)
                result.append(self.episode_score_history[episode])
                writer = csv.writer(f)
                writer.writerow(result)
                
          time_end = time.time()
          print("\nFinished, Total time used: {}s".format(time_end - time_start))

    def load(self, model_name):
      self.saver.restore(self.sess, os.path.join(self.save_path, '{}.ckpt'.format(model_name)))
      print('Model {} Loaded!'.format(model_name))

    def act(self, observation, reward, done=False):

      #check line:
      state_dict = observation.to_dict()
      lines_status_list_bool = state_dict["line_status"]  

      self.action_next = np.zeros(76)
      for i, v in enumerate(state_dict['rho']):
        if v <= 0:
          self.id.add(i)
          self.line_count[i] +=1
          if self.line_count[i] >10: self.line_count[i] = 10
        if v > 0:
            try: 
                self.line_count[i] = 0
                self.id.remove(i)
            except: pass
          
      idlist = [i for i in self.id]
      action_transf_dict = {}
      done_simulate = False

      #if cooldown time is reached, close the line and check validation
      if len(idlist) > 0:   
        print("Broken_Line", self.line_count)               
        for i, x in enumerate(self.line_count):
            if x == 10 and (False in lines_status_list_bool): 
                set_status = self.agent_action_space.get_set_line_status_vect()
                for o, e in [(1,1),(1,2),(2,1),(2,2)]:
                    action_transf_dict = self.agent_action_space.reconnect_powerline(line_id=i,bus_or=o,bus_ex=e)  
                    obs_simulate, reward_simulate, done_simulate, _ = observation.simulate(action_transf_dict)
                    print(action_transf_dict, done_simulate)
                    if not done_simulate: break                 

                self.action_next[56+idlist[0]] = 1.0
                action = self.action_to_index.item().get(tuple(self.action_next)) 
                self.id.clear()
                self.line_count[i] = 0
                break
      #Else agent pick action              
      if 1.0 not in self.action_next or done_simulate:

        action_array = self.action_space[155]
        action_transf_dict = self.action_transformation(action_array, self.env.name)

        obs_simulate, reward_simulate, done_simulate, infor = observation.simulate(action_transf_dict)
        #print("first simulate score", reward_simulate)
        has_danger = False
        has_overflow = False 
        if reward_simulate is None or reward_simulate <= 0:
          has_danger = True
        elif obs_simulate is not None: 
          has_overflow = any(obs_simulate.rho>1.05)

        # if game over, run more simulation
        if done_simulate or reward_simulate is None or has_danger or has_overflow:

          state = [0 if i != i else i for i in observation.to_vect()] 
          action, q_predictions = self.dqn_main.act(self.sess, self.normalize([state]), 0.0)
          top_actions = np.argsort(q_predictions)[-1: -11: -1].tolist()

          chosen_action = 155
          max_score = float('-inf')
          for action in tuple(top_actions+[155]):           

            action_array = self.action_space[action]
            action_transf_dict = self.action_transformation(action_array, self.env.name)
            obs_simulate, reward_simulate, done_simulate, _= observation.simulate(action_transf_dict)
            #print("simulate score", reward_simulate, action)
            if obs_simulate is None:
              continue
            else:
              obs_dict = obs_simulate.to_dict()
              lineflow_simulate_ratio = obs_dict["rho"]
              lineflow_simulate_ratio = [round(x, 4) for x in lineflow_simulate_ratio]

              # seperate big line and small line
              has_danger = False
              for ratio in lineflow_simulate_ratio:
                if  ratio > 1.02:
                  has_danger = True

              if not done_simulate and reward_simulate > max_score and not has_danger:
                max_score = reward_simulate
                chosen_action = action
                print('current best action: {}, score: {:.4f}'.format(chosen_action, reward_simulate))

          # chosen action
          action_array = self.action_space[chosen_action]
          action_transf_dict = self.action_transformation(action_array, self.env.name)
      #print(reward,done,action_transf_dict)
      return action_transf_dict

