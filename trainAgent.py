import numpy as np
import tensorflow as tf
import gym
import cv2 as cv
import matplotlib.pyplot as plt
import os
from gym.wrappers import Monitor
from DQNModel import *
from ops import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#some values
numEpisodes = 1000
initReplaySize = 10000
replaySize = 25000
batchSize = 32
startE = 1.0
endE = 0.1
annealingSteps = 20000
discountFactor = 0.99
videoFrequency = 20000
copyFrequency = 20000
checkpointDir = "checkpoint"
monitorDir = "monitor"


def trainAgent():
    #start environment
    env = gym.make('BreakoutDeterministic-v4')
    
    totalActions = env.action_space.n
    
    tf.reset_default_graph()
    
    targetModel = atariAgent(totalActions,scope="targetModel")
    QModel = atariAgent(totalActions,scope="QModel")
    
    if not os.path.exists(checkpointDir):
        os.makedirs(checkpointDir)
    if not os.path.exists(monitorDir):
        os.makedirs(monitorDir)
        
    checkpoint = os.path.join(checkpointDir,"model")
    monitor = os.path.join(monitorDir,"game")
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        
        env = Monitor(env, directory=monitor, video_callable=lambda totalStep: totalStep % \
                      videoFrequency == 0, resume=True)
        
        state = env.reset()
        
        ckpt = tf.train.latest_checkpoint(checkpointDir)
        if ckpt:
            saver.restore(sess,ckpt)
            totalStep = 0
            print("Existing checkpoint {} restored...".format(ckpt))
        else:
            totalStep = 0
            
        cell = namedtuple("cell","state reward action nextState isDone")
        
        replayMemory = initExperienceReplay(env,initReplaySize,cell)
        
        epsilonValues = np.linspace(startE,endE,num=annealingSteps)
        
        
        
        episodeLengths = []
        episodeRewards = []
        
        for episode in range(numEpisodes):
            state = env.reset()
            state = preprocess(state)
            state = np.stack([state] * 4, axis=2)
            totalReward = 0.0
            loss = None
            episodeLength = 0
            
            while(True):
                if(totalStep%copyFrequency == 0):
                    copyParameters(sess,targetModel,QModel)
                    print("Target Model updated...")

                
                epsilon = epsilonValues[min(totalStep, annealingSteps-1)]
                QValues = QModel.play(sess,np.expand_dims(state,0))
                bestAction = EGreedyPolicy(epsilon,QValues)

                nextState,reward,isDone,_ = env.step(bestAction)
                totalReward += reward
                nextState = preprocess(nextState)
                #nextState = np.stack([nextState] * 4,axis=2)
                nextState = np.append(state[:,:,1:],nextState[:,:,np.newaxis],axis=2)
                
                if(len(replayMemory) == replaySize):
                    replayMemory.popleft()
                
                replayMemory.append(cell(state,reward,bestAction,nextState,isDone))

                indices = np.random.choice(len(replayMemory)-1,batchSize,replace=False)
                batch = [replayMemory[i] for i in indices]
                #batch = random.sample(replayMemory, batchSize)
                states,rewards,actions,nextStates,isDones = map(np.array,zip(*batch))
                
                #targetmodel prediction
                tQValues = targetModel.play(sess,nextStates)
                targetY = rewards + (1 - isDones) * discountFactor * np.amax(tQValues,axis=1)

                #gradient descent step
                loss = QModel.train(sess,states,targetY,actions)
                episodeLength += 1
                totalStep += 1
                
                if(isDone):
                    episodeRewards.append(reward)
                    episodeLengths.append(episodeLength)
                    print("Episode {} Stats:\n\tGlobal step: {} Final reward: {} episode length: {} loss: {}\n"\
                          .format(episode,totalStep,totalReward,episodeLength,loss))
                        
                    saver.save(tf.get_default_session(), checkpoint)
                    break
                    
                else:
                    state = nextState
                    totalStep += 1

                    
if(__name__ == "__main__"):
    trainAgent()