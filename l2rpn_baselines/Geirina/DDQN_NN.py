# Copyright (C) 2018 - 2020 GEIRI North America, JEPC
# Authors: jiajaun <jiajun.duan@geirina.net>

import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


class DeepQNetworkDueling(object):
  def __init__(self,
               n_state=420,
               n_actions=175,
               learning_rate=1e-5,
               scope='dqn',
               summaries_dir=None):
    """
    Deep Q Network with experience replay and fixed Q-target

    Args:
    
      learning_rate: model learning rate
      scope: graph scope name
      summaries_dir: log directory of Tensorboard Filewriter

    Returns:
      pass
    """


    self.learning_rate = learning_rate
    self.n_state = n_state
    self.n_actions = n_actions
    self.scope = scope
    self.summary_writer = None

    # Build model and create Tensorboard summary
    with tf.variable_scope(scope):
      self._build_model()
      if summaries_dir:
        summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
        if not os.path.exists(summary_dir):
          os.makedirs(summary_dir)
        self.summary_writer = tf.summary.FileWriter(summary_dir)

  def _build_model(self):
    """
    Build Tensorflow graph
    """
    with tf.variable_scope('inputs'):
      # State as Vector
      self.s = tf.placeholder(
          shape=[None, self.n_state], dtype=tf.float32, name='s')
      # Placeholder for TD target
      self.q_target = tf.placeholder(
          shape=[None], dtype=tf.float32, name='q_target')
      # Placeholder for actions
      self.action_ph = tf.placeholder(
          shape=[None, 1], dtype=tf.int32, name='action')
      # Placeholder for Importance-Sampling(IS) weights
      self.IS_weights = tf.placeholder(
          shape=[None], dtype=tf.float32, name='IS_weights')

    # forward pass
    def forward_pass(s):
      
      s = tf.layers.batch_normalization(s, name = 'bn')
      fc1 = tf.layers.Dense(units=256, activation=tf.nn.relu, name='fc1')(s)

      # fc_V
      fc_V = tf.layers.Dense(64, activation=tf.nn.relu, name='fc_V_1')(fc1)
      self.fc_V = tf.layers.Dense(1, activation=None, name='fc_V_2')(fc_V)

      # fc_A
      fc_A = tf.layers.Dense(32, activation=tf.nn.relu, name='fc_A_1')(fc1)
      self.fc_A = tf.layers.Dense(self.n_actions, activation=None, name='fc_A_2')(fc_A)

      with tf.variable_scope('q_predict'):
        mean_A = tf.reduce_mean(self.fc_A, axis=1, keep_dims=True)
        q_predict = tf.add(self.fc_V, tf.subtract(self.fc_A, mean_A))

      return q_predict

    # forward pass
    self.q_predict = forward_pass(self.s)

    # q_predict for chosen action
    indexes = tf.reshape(tf.range(tf.shape(self.q_predict)[0]), shape=[-1, 1])
    action_indexes = tf.concat([indexes, self.action_ph], axis=1)
    self.q_predict_action = tf.gather_nd(self.q_predict, action_indexes)

    # calculate the mean batch loss
    self.abs_TD_errors = tf.abs(self.q_target - self.q_predict_action)
    self.loss_batch = self.IS_weights * \
        tf.squared_difference(self.q_target, self.q_predict_action)

    with tf.variable_scope('loss'):
      self.loss = tf.reduce_mean(self.loss_batch)

    # optimizer
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    # train_op
   # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.variable_scope('train_op'):
     # with tf.control_dependencies(update_ops):
      self.train_op = self.optimizer.minimize(
          self.loss, tf.train.get_global_step())

    # trainable parameters
    self.params_train = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    # Summaries for Tensorboard
    # self.summaries = tf.summary.merge([
    #     tf.summary.histogram("loss_hist", self.loss_batch),
    #     tf.summary.histogram("q_values_hist", self.q_predict),
    # ])

  def epsilon_greedy(self, q_predict, epsilon):
    roll = np.random.uniform()
    if roll < epsilon:
      return np.random.randint(self.n_actions)
    else:
      return np.argmax(q_predict[0])

  def act(self, sess, s, epsilon):
    q_predict = sess.run(self.q_predict, feed_dict={self.s: s})
    return self.epsilon_greedy(q_predict, epsilon), q_predict[0]

  def predict(self, sess, s):
    return sess.run([self.q_predict], feed_dict={self.s: s})

  def update(self, sess, s, q_target, action, IS_weights):
    feed_dict = {self.s: s, self.q_target: q_target,
                 self.action_ph: action, self.IS_weights: IS_weights}
    _, loss, abs_TD_errors = sess.run(
        [self.train_op, self.loss, self.abs_TD_errors], feed_dict)
   # if self.summary_writer:
   #   self.summary_writer.add_summary(summaries)
    return loss, abs_TD_errors
