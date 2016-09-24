# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 12:16:12 2016

@author: CE
"""

import random
import numpy as np
from copy import deepcopy

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
    
    def update_policy(self, state, action):
        for index, s in enumerate(self.states):
            if s == state:
                self.policy[index] = action
                return
                
    def get_policy_action(self, state):
        for index, s in enumerate(self.states):
            if s == state:
                return self.policy[index]
    
    def value_iteration(self, eps):
        counter = 0
        while True:
            delta = 0
            counter += 1
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
                print(counter)
                return
                
    
    def policy_evaluation(self):
        rewards = []
        u_matrix = np.zeros((len(self.states), len(self.states)))
        for state in solver.states:
            action = solver.get_policy_action(state)
            if action == None: 
                rewards.append(0)        
                continue
            transitions = solver.get_transition_probability(state, action)
            rewards.append(sum(p * solver.get_reward(s1, action) for (p, s1) in transitions))        
        
        for i, state1 in enumerate(solver.states):
            action = solver.get_policy_action(state1)
            if action == None: 
                u_matrix[i][i] += 1
                continue
            transitions = solver.get_transition_probability(state1, action)
            for j, state2 in enumerate(solver.states):
                if i == j: u_matrix[i][j] += 1
                if state2 == transitions[0][1]:
                    p = transitions[0][0]
                elif state2 == transitions[1][1]:
                    p = transitions[1][0]
                else:
                    p = 0
                u_matrix[i][j] -= solver.gamma * p
        rewards = np.array(rewards)
        solver.utility = np.linalg.solve(u_matrix, rewards)
    
    
    def init_policy(self):
        for index, state in enumerate(self.states):
            actions = self.get_applicable_actions(state)
            if len(actions) == 0: continue
            self.policy[index] = random.choice(actions)
    
    def policy_iteration(self):
        self.init_policy()
        while True:
            self.policy_evaluation()
            unchanged = True
            for state in self.states:
                actions = self.get_applicable_actions(state)
                if len(actions) == 0: continue
                state_utilities = []
                state_actions = []
                    
                for action in actions: 
                    transitions = self.get_transition_probability(state, action)
                    reward = sum(p * self.get_reward(s1, action) for (p, s1) in transitions)
                    utility = reward + self.gamma * sum(p * self.get_utility(s1) for (p, s1) in transitions)
                    state_utilities.append(utility)
                    state_actions.append(action)
                
                index_max = state_utilities.index(max(state_utilities))
                best_action = state_actions[index_max]
                if(self.get_policy_action(state) != best_action):
                    self.update_policy(state, best_action)
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
solver.value_iteration(0.000001)
#solver.policy_iteration()