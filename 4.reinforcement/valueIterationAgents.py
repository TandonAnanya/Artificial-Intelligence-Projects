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
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            temp = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                action_values = util.Counter()
                for action in self.mdp.getPossibleActions(state):
                    s=0
                    for item in self.mdp.getTransitionStatesAndProbs(state, action):
                        s+=item[1] * (self.mdp.getReward(state, action, item[0]) + self.discount * self.values[item[0]])
                    action_values[action] = s
                temp[state] = max(list(action_values.values()))
            self.values = temp


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        s=0
        for item in self.mdp.getTransitionStatesAndProbs(state, action):
            s+= item[1] * (self.mdp.getReward(state, action, item[0]) + self.discount * self.values[item[0]])
        return s
       #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            s=0
            for item in self.mdp.getTransitionStatesAndProbs(state, action):
                reward = self.mdp.getReward(state, action, item[0])
                s+=item[1] * (reward + self.discount * self.values[item[0]])
            actions[action] = s
        return actions.argMax()
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"        
        i = 0
        while True:
          if i<self.iterations:           
            for state in self.mdp.getStates():
              Qstate = util.Counter()
              for action in self.mdp.getPossibleActions(state):
                Qstate[action] = self.computeQValueFromValues(state, action)
              
              self.values[state] = Qstate[Qstate.argMax()]
              
              i += 1
              if i >= self.iterations:
                return
          else:
            break

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        values, dic, pq = self.values, {}, util.PriorityQueue()        
        for state in self.mdp.getStates():
          dic[state] = set()

        for state in self.mdp.getStates():          
          Qstate = util.Counter()                                
          for action in self.mdp.getPossibleActions(state):
            for item in self.mdp.getTransitionStatesAndProbs(state, action):
              if item[1] != 0:
                dic[item[0]].add(state)
            Qstate[action] = self.computeQValueFromValues(state, action)

          if self.mdp.isTerminal(state)==False:            
            pq.update(state, -1*abs(values[state] - Qstate[Qstate.argMax()]))

        i=0
        while i<self.iterations:        
          if pq.isEmpty():
            return
          state = pq.pop()
          if self.mdp.isTerminal(state)==False:
            Qstate = util.Counter()
            for action in self.mdp.getPossibleActions(state):
              Qstate[action] = self.computeQValueFromValues(state, action)
            values[state] = Qstate[Qstate.argMax()]

          for item in dic[state]:
            X = util.Counter()
            for action in self.mdp.getPossibleActions(item):              
              X[action] = self.computeQValueFromValues(item, action)                          
            if self.theta<abs(values[item] - X[X.argMax()]):
              pq.update(item, -1*abs(values[item] - X[X.argMax()]))
          i+=1