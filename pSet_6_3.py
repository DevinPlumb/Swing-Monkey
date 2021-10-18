import math
import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt
import sys

from SwingyMonkey import SwingyMonkey

class Learner:

  def __init__(self):
    self.last_state  = None
    self.last_action = None
    self.last_reward = None
    self.Q = np.zeros((100, 2))

  def reset(self):
    self.last_state  = None
    self.last_action = None
    self.last_reward = None

  def action_callback(self, state):
    '''Implement this function to learn things and take actions.
    Return 0 if you don't want to jump and 1 if you do.'''
    
    if self.last_state == None:
        self.last_action = 0
        self.last_state  = state
        return self.last_action
    
    else:
        
        def stateConvert(s):
            # Parse the dictionary
            score = s['score']
    
            tree = s['tree']
            dist = tree['dist']
            treeTop = tree['top']
            treeBot = tree['bot']
    
            monkey = s['monkey']
            vel = monkey['vel']
            monkeyTop = monkey['top']
            monkeyBot = monkey['bot']

            # Define state with binning
            s_new = (dist+115)/125
            if monkeyBot > treeBot:
                s_new = s_new+5
                if monkeyBot+monkeyTop > treeBot+treeTop:
                    s_new = s_new+5
                    if monkeyTop > treeTop:
                        s_new = s_new+5
            if vel > -30:
                s_new = s_new+20
                if vel > -20:
                    s_new = s_new+20
                    if vel > -10:
                        s_new = s_new+20
                        if vel > 0:
                            s_new = s_new+20
            return s_new
    
        alpha = 0.1
        gamma = 0.99
    
        s = stateConvert(self.last_state)
        s_prime = stateConvert(state)
        a_prime = 0
        if self.Q[s_prime, 1] > self.Q[s_prime, a_prime]:
            a_prime = 1
    
        self.Q[s, self.last_action] = self.Q[s, self.last_action] + alpha*((self.last_reward + gamma*self.Q[s_prime, a_prime])-self.Q[s, self.last_action])
    
        self.last_action = a_prime
        self.last_state = state

        return self.last_action

  def reward_callback(self, reward):
    '''This gets called so you can see what reward you get.'''

    self.last_reward = reward

iters = 500
learner = Learner()

x = np.arange(iters)
scores = []
cumulativeScore = 0
cumulativeScores = []
maxScore = 0
maxX = []
maxScores = []
mean = []

for ii in range(iters):

  # Make a new monkey object.
  swing = SwingyMonkey(sound=False,            # Don't play sounds.
                       text="Epoch %d" % (ii), # Display the epoch on screen.
                       tick_length=1,          # Make game ticks super fast.
                       action_callback=learner.action_callback,
                       reward_callback=learner.reward_callback)

  # Loop until you hit something.
  while swing.game_loop():
    pass

  scores.append(swing.score)
  cumulativeScore = cumulativeScore + swing.score
  cumulativeScores.append(cumulativeScore)
  mean.append(float(cumulativeScore)/float(ii+1))
  if swing.score > maxScore:
    maxScore = swing.score
    maxX.append(ii)
    maxScores.append(maxScore)
  

  # Reset the state of the learner.
  learner.reset()
    
plt.scatter(x, np.asarray(scores))
plt.xlabel('Generation')
plt.ylabel('Score')
plt.title('Score Improvement')
plt.savefig("pSet_6_3_scoreImprovement.png")
plt.show()

plt.plot(x, np.asarray(cumulativeScores))
plt.xlabel('Generation')
plt.ylabel('Cumulative Score')
plt.title('Cumulative Score Growth')
plt.savefig("pSet_6_3_cumulativeScoreGrowth.png")
plt.show()

plt.plot(np.asarray(maxX), np.asarray(maxScores))
plt.xlabel('Generation')
plt.ylabel('Max Score')
plt.title('Max Score Growth')
plt.savefig("pSet_6_3_maxScoreGrowth.png")
plt.show()

plt.plot(x, np.asarray(mean))
plt.xlabel('Generation')
plt.ylabel('Mean Score')
plt.title('Mean Score Growth')
plt.savefig("pSet_6_3_meanScoreGrowth.png")
plt.show()