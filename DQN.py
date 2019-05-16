#import copy
import numpy as np
import tensorflow as tf
from envir import Environment      # Environment class
from collections import deque
import random
GAMMA=0.9
EPISODE = 10000
STEP = 23
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
REPLAY_SIZE = 2000
BATCH_SIZE = 32
TEST = 5
global loss_track
    


class Agent():
    def __init__(self, env):
    
        #init experience replay
        self.replay_buffer = deque()
        
        #init some parameter
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = 3      ##输入是离散的数据，包括负荷需求、光伏发电量、SOC
        self.action_dim = 100    ##内燃机发电和电池充放电分别离散成10个动作

        self.create_Q_network()
        self.create_training_method()

        #init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        #loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("could not find previous network weights")

        global summary_writer
        summary_writer = tf.train.SummaryWriter('~/logs', graph = self.session.graph)
   
    def create_Q_network(self):

        #network weights
        W1 = self.weights_variable([self.state_dim, 30])
        b1 = self.bias_variable([30])
        W2 = self.weights_variable([30, self.action_dim])
        b2 = self.bias_variable([self.action_dim])

        #input layer
        self.state_input = tf.placeholder("float", [None, self.state_dim])

        #hidden layer
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)

        #Q value layer 
        self.Q_value = tf.matmul(h_layer, W2) + b2

    def create_training_method(self):
        #one-hot representation
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.mul(self.Q_value, self.action_input), reduction_indices = 1)
        #compute loss
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        tf.scalar_summary("loss", self.cost)
        global merged_summary_op
        merged_summary_op = tf.merge_all_summaries()
        #optimize training process automatically
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) >  BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 10
        loss_track = []
        #Step 1. obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        #Step 2. calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict = {self.state_input:next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                # Q(s, a) = Reward + max(Q(next_state, a))
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
                # 由此慢慢探索，get到label，继而能够去更新cost，从而优化Q_network
            
            
        
        self.optimizer.run(feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
            })

        summary_str = self.session.run(merged_summary_op, feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
            })
        _, loss = self.session.run([merged_summary_op, self.cost], feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
            })
        summary_writer.add_summary(summary_str, self.time_step)
        if self.time_step % 100 == 0:
            loss_track.append(loss)
            print("Step %d: loss = %.2f" % (self.time_step, loss))
        if self.time_step % 1000 == 0:
            self.saver.save(self.session, "saved_networks/" + "network" + "-dqn", global_step = self.time_step)

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict = {
            self.state_input:[state]
            })[0]
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)

        else:
            return np.argmax(Q_value)

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 100000

    def action(self, state):
        
        return np.argmax(self.Q_value.eval(feed_dict = {
            self.state_input:[state]
            })[0])

    def weights_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)
        
def main():
    env = Environment()
    agent = Agent(env)
    reward_track = []
    huafei_track = []
    pg_output = []
    for episode in range(EPISODE):
        #initialize task
        state = env.reset()
        for step in range(STEP):
            ##action = agent.egreedy_action(env.soc_drop, state)
            action = agent.egreedy_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_track.append(reward)
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
    #Test every 100 episode
        if episode % 100 == 0:
            total_reward = 0
            huafei = 0.0
            for i in range(5):
                state = env.reset()
                for j in range(STEP):
                    action = agent.action(state)
                    
                    huafei = huafei + env.pdpower(action) * 10 + pg_output * 16 
                    next_state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
         
            ave_reward = total_reward / 5
            print ('episode: ', episode, 'Evaluation on Average Reward:' , ave_reward)
            ave_huafei = huafei / 5
            huafei_track.append(ave_huafei)
            print ('episode: ', episode, 'Huafei:' , ave_huafei)
        
    f = open('DQN_reward_reply.txt', 'w')
    print >> f, reward_track
    f.close()
    
    f = open('DQN_loss_100.txt', 'w')
    print >> f, reward_track
    f.close()
    
    f = open('DQN_reply_huafei.txt', 'w')
    print >> f, huafei_track
    f.close()
    print("finished")

if __name__ == '__main__':
    main()
