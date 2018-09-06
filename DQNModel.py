import numpy as np
import tensorflow as tf


class atariAgent():
    """ Atari Agent contains the model and functions to predict and train the agent"""
    def __init__(self,totalActions,scope = "agent"):
        self.scope = scope
        self.totalActions = totalActions
        with tf.variable_scope(self.scope):
            self.QModel()
    
    def QModel(self):
        """Contains the model"""
        self.Xin = tf.placeholder(shape=[None,84,84,4],dtype=tf.uint8,name='Xin')
        self.y = tf.placeholder(shape=[None],dtype=tf.float32,name='yin')
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32,name='actions')
        
        X = tf.to_float(self.Xin) / 255.0
        
        #model starts
        conv1 = tf.contrib.layers.conv2d(X,16,8,4,activation_fn=tf.nn.relu)
        
        conv2 = tf.contrib.layers.conv2d(conv1,32,4,2,activation_fn=tf.nn.relu)
        
        convOut = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(convOut,256,activation_fn=tf.nn.relu)
        self.QValues = tf.contrib.layers.fully_connected(fc1,self.totalActions,activation_fn=None)
        
        batchSize = tf.shape(self.Xin)[0]
        yIndices = tf.range(batchSize) * self.totalActions + self.actions
        self.predictedActions = tf.gather(tf.reshape(self.QValues,[-1]),yIndices)
        
        #calculates loss function
        self.losses = tf.squared_difference(self.y, self.predictedActions)
        self.loss = tf.reduce_mean(self.losses)
        
        #training step
        self.optimizer = tf.train.RMSPropOptimizer(0.00025,0.99)
        self.trainStep = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        
        
    def play(self,sess,states):
        """runs the model for the given state and predicts the Q values"""
        return sess.run(self.QValues,{self.Xin : states})
        
    def train(self,sess,states,y,actions):
        """Trains the Agent on the given input and target values and returns the loss
        """
        feed_dict = { self.Xin: states, self.y: y, self.actions: actions }
        loss, _ = sess.run([self.loss, self.trainStep],feed_dict)
        
        return loss
    
    