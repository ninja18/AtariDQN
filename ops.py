import numpy as np
import tensorflow as tf
import gym
import cv2 as cv
from collections import namedtuple,deque


def preprocess(img):
    grayImg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    resized = cv.resize(grayImg,(84,110))
    preProcessedImg = resized[:84,:84]
    
    return preProcessedImg


def initExperienceReplay(env,initReplaySize,cell):
    replayBuffer = deque()
    state = env.reset()
    state = preprocess(state)
    state = np.stack([state]*4,axis=2)
    print("Filling Experience memory of the agent")
    for i in range(initReplaySize):
        action = env.action_space.sample()
        nextState, reward, isDone, _ = env.step(action)
        nextState = preprocess(nextState)
        nextState = np.append(state[:,:,1:],nextState[:,:,np.newaxis],axis=2)
        replayBuffer.append(cell(state,reward,action,nextState,isDone))
        if(isDone):
            state = env.reset()
            state = preprocess(state)
            state = np.stack([state]*4,axis=2)
        else:
            state = nextState
    
    env.close()
    print("Filled memory of size {}".format(len(replayBuffer)))       
    return replayBuffer


def EGreedyPolicy(epsilon,QValues):
    numActions = QValues.shape[1]
    probs = np.ones(numActions, dtype=float) * epsilon / numActions
    best_action = np.argmax(QValues)
    
    probs[best_action] += (1.0 - epsilon)
    #print(probs)
    optimizedAction = np.random.choice(numActions,p=probs)
    return optimizedAction


def copyParameters(sess,targetModel,QModel):
    params1 = [var for var in tf.trainable_variables() if var.name.startswith(targetModel.scope)]
    params1 = sorted(params1,key=lambda var: var.name)
    params2 = [var for var in tf.trainable_variables() if var.name.startswith(QModel.scope)]
    params2 = sorted(params2,key=lambda var: var.name)
    copies = []
    for p1,p2 in zip(params1,params2):
        copy = p1.assign(p2)
        copies.append(copy)
    sess.run(copies)
