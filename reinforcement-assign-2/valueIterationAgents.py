# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Loop for the number of iterations specified
        for i in range(self.iterations):
            # Create a new counter to store updated values 
            newValues = util.Counter()

            # Go through every state in the MDP
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                else:
                    actions = self.mdp.getPossibleActions(state)
                    # Compute Q-values for all possible actions
                    q_values = []

                    for action in actions:
                        q = self.computeQValueFromValues(state, action)
                        q_values.append(q)

                    # Update state value with the maximum Q-value
                    if len(q_values) > 0:
                        newValues[state] = max(q_values)

            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        # Initialize Q-value
        q = 0
        
        # Loop over possible next states and their probabilities
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            
            # Bellman equation:
            q += prob * (reward + self.discount * self.values[nextState])
        return q

    def computeActionFromValues(self, state):
        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state)

        best_action = None
        best_value = float('-inf')

        # Find the action with the highest Q-value
        for action in actions:
            q = self.computeQValueFromValues(state, action)
            if q > best_value:
                best_value = q
                best_action = action

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state) # Return the optimal policy at a state.

    def getAction(self, state):
        return self.computeActionFromValues(state) # Return the action chosen by the policy (no exploration)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action) # Return Q(s, a) based on current value estimates 
