# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import gridworld

import random,util,math
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qvalue = util.Counter()

    def getQValue(self, state, action):
        return self.qvalue[(state, action)]

    def computeValueFromQValues(self, state):
        legalActions = self.getLegalActions(state)

        if len(legalActions) == 0:
            return 0.0

        return max([self.getQValue(state, a) for a in legalActions])

    def computeActionFromQValues(self, state):
        legalActions = self.getLegalActions(state)

        if len(legalActions) == 0:
            return None

        best_value = float('-inf')
        best_actions = []

        for action in legalActions:
            q = self.getQValue(state, action)

            if q > best_value:
                best_value = q
                best_actions = [action]
            elif q == best_value:
                best_actions.append(action)

        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        
        legalActions = self.getLegalActions(state)
            # if no actions (terminal state)
        if len(legalActions) == 0:
            return None

        # epsilon-greedy: explore or exploit
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)   # explore
        else:
            return self.computeActionFromQValues(state)  # exploit
        

    def update(self, state, action, nextState, reward: float):
       current_q = self.getQValue(state, action)
       next_value = self.computeValueFromQValues(nextState)

       sample = reward + self.discount * next_value

       self.qvalue[(state, action)] = (1 - self.alpha) * current_q + self.alpha * sample


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

