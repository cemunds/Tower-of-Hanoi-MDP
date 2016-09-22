# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 12:16:12 2016

@author: CE
"""

import numpy as np
from copy import deepcopy
from random import randint

class Solver(object):
    
    def __init__(self):
        self.gamma = 0.9
        self.states =   [
                            TowerOfHanoi([["A", "B"], [], []]),
                            TowerOfHanoi([["A"], ["B"], []]),
                            TowerOfHanoi([["A"], [], ["B"]]),
                            TowerOfHanoi([[], ["A", "B"], []]),
                            TowerOfHanoi([[], ["A"], ["B"]]),
                            TowerOfHanoi([["B"], ["A"], []]),
                            TowerOfHanoi([[], [], ["A", "B"]]),
                            TowerOfHanoi([["B"], [], ["A"]]),
                            TowerOfHanoi([[], ["B"], ["A"]]),
                            TowerOfHanoi([["B", "A"], [], []]),
                            TowerOfHanoi([[], ["B", "A"], []]),
                            TowerOfHanoi([[], [], ["B", "A"]])
                        ]
        self.actions = [(0,1),(0,2),(1,2),(1,0),(2,0),(2,1)]
        self.utility = np.zeros(len(self.states))
        self.policy = [None for s in range(len(self.states))]
        
    def get_applicable_actions(self, state):
        if state.is_goal_state(): return []        
        actions = []
        for index, pin in enumerate(state.pins):
            for action in filter(lambda x: x[0] == index and len(pin) > 0, self.actions):
                actions.append(action)
        return actions
        
    def get_all_states(self):
        pass
    
    def get_transition_probability(self, state, action):
        srcPin, dstPin = action
        wrongPin = None
        if srcPin == 0:
            if dstPin == 1:
                wrongPin = 2
            else:
                wrongPin = 1   
        elif srcPin == 1:
            if dstPin == 0:
                wrongPin = 2
            else:
                wrongPin = 0
        elif srcPin == 2:
            if dstPin == 0:
                wrongPin = 1
            else:
                wrongPin = 0
                
        return [(0.9, state.move_disk(action)),
                ((0.1), state.move_disk((srcPin, wrongPin)))]
        
    def get_reward(self, state, action):
        if state.is_goal_state():
            return 100
        elif state.is_invalid_state():
            return -10
        else:
            return -1
    
    def get_utility(self, state):
        for index, s in enumerate(self.states):
            if s == state:
                return self.utility[index]
    
    def update_utility(self, state, value):
        for index, s in enumerate(self.states):
            if s == state:
                self.utility[index] = value
                return
        
    def init_policy(self):
        for state in self.states:
            actions = self.get_applicable_actions(state)
            if len(actions) == 0: continue
            rnd = randint(0, len(actions)-1)
            self.update_policy(state, actions[rnd])
    
    def update_policy(self, state, action):
        for index, s in enumerate(self.states):
            if s == state:
                self.policy[index] = action
                return
    
    def value_iteration(self, eps):
        while True:
            delta = 0
            for state in self.states:
                actions = self.get_applicable_actions(state)
                if len(actions) == 0: continue
                state_utilities = []
                state_actions = []
                
                for action in actions:
                    transitions = self.get_transition_probability(state, action)
                    reward = sum(p * self.get_reward(s1, action) for (p, s1) in transitions)
                    utility = reward + 0.9 * sum(p * self.get_utility(s1) for (p, s1) in transitions)
                    state_utilities.append(utility)
                    state_actions.append(action)
                
                index_max = state_utilities.index(max(state_utilities))
                max_utility = state_utilities[index_max]
                best_action = state_actions[index_max]
                delta = max(delta, abs(max_utility - self.get_utility(state)))
                self.update_utility(state, max_utility)
                self.update_policy(state, best_action)
            if delta < eps:
                return
                
    def policy_iteration(self):
        self.init_policy()
        while True:
            unchanged = True
            for state in self.states:
                actions = self.get_applicable_actions(state)
                if len(actions) == 0: continue
                    
                for action in actions: 
                    max_a = None #argmax(self.get_applicable_actions(state), lambda a: )
                    if(self.get_policy_action(state) != max_a):
                        self.update_policy(max_a)
                        unchanged = False
            if unchanged:
                return
    
class TowerOfHanoi(object):    
    
    def __init__(self, state):
        self.number_of_disks = sum(1 for pin in state for disk in pin)
        self.pins = state
        
    def __eq__(self, other):
        for i, pin in enumerate(self.pins):
            if pin != other.pins[i]:
                return False
        return True
        
    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __str__(self):
        a = "-".join(str(x) for x in self.pins[0])
        b = "-".join(str(x) for x in self.pins[1])
        c = "-".join(str(x) for x in self.pins[2])
        return "Pin1: {0} Pin2: {1} Pin3: {2}".format(a, b, c)
        
    def move_disk(self, action):
        srcPin, dstPin = action
        pins = deepcopy(self.pins)
        disk = pins[srcPin].pop()
        pins[dstPin].append(disk)
        return TowerOfHanoi(pins)
        
    def is_goal_state(self):
        if len(self.pins[2]) != self.number_of_disks: return False
        last = None
        for disk in self.pins[2]:
            if not last == None and not ord(disk) > ord(last):
                return False
            last = disk
        return True
        
    def is_invalid_state(self):
        last = None
        for pin in self.pins:
            for disk in pin:
                if not last == None and ord(disk) < ord(last):
                    return True
                last = disk
            last = None
        return False

solver = Solver()
solver.value_iteration(0.000001);