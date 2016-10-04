# -*- coding: utf-8 -*-
import random
import numpy as np
from copy import deepcopy

class Learner(object):
    
    def __init__(self, initial_state):
        self.gamma = 0.9
        self.current_state = initial_state
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
        self.q_matrix = np.zeros((len(self.states), len(self.actions)))
        self.learning_rates = np.ones((len(self.states), len(self.actions)))
        
    def get_applicable_actions(self, state):
        if state.is_goal_state(): return []        
        actions = []
        for index, pin in enumerate(state.pins):
            for action in filter(lambda x: x[0] == index and len(pin) > 0, self.actions):
                actions.append(action)
        return actions
        
    def get_reward(self, state):
        if state.is_goal_state():
            return 100
        elif state.is_invalid_state():
            return -10
        else:
            return -1
            
    def execute_action(self, action):
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
                
        return np.random.choice([self.current_state.move_disk(action), self.current_state.move_disk((srcPin, wrongPin))], p=[0.9, 0.1])
    
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
    
    def update_q_value(self, state, action, value):
        action_index = self.actions.index(action)
        for i, s in enumerate(self.states):
            if s == state:
                state_index = i
                break
        self.q_matrix[state_index][action_index] = value
    
    def get_q_value(self, state, action):
        action_index = self.actions.index(action)
        for i, s in enumerate(self.states):
            if s == state:
                state_index = i
                break
        return self.q_matrix[state_index][action_index]
        
    def reset_state(self):
        while self.current_state.is_goal_state():
            self.current_state = np.random.choice(self.states)
        
    def get_learning_rate(self, state, action):
        action_index = self.actions.index(action)
        for i, s in enumerate(self.states):
            if s == state:
                state_index = i
                break
        n = self.learning_rates[state_index][action_index]
        return 1.0/n
        
    def update_learning_rate(self, state, action):
        action_index = self.actions.index(action)
        for i, s in enumerate(self.states):
            if s == state:
                state_index = i
                break
        self.learning_rates[state_index][action_index] += 1
        
    def q_learning(self):
        i = 0
        while i < 100000:
            if(self.current_state.is_goal_state()):
                self.reset_state()
            actions = self.get_applicable_actions(self.current_state)
            action = random.choice(actions)
            new_state = self.execute_action(action)
            reward = self.get_reward(new_state)
            q_value = self.get_q_value(self.current_state, action)
            learning_rate = self.get_learning_rate(self.current_state, action)
            max_q = max([self.get_q_value(new_state, a) for a in self.get_applicable_actions(new_state)]+[0])
            new_q = q_value + learning_rate * (reward + self.gamma * max_q - q_value)
            self.update_q_value(self.current_state, action, new_q)            
            self.update_learning_rate(self.current_state, action)
            self.current_state = new_state
            i += 1
        
        for i, state in enumerate(self.states):
            if state.is_goal_state(): continue
            action_index = np.argmax(self.q_matrix[i])
            self.update_utility(state, max(self.q_matrix[i]))
            self.update_policy(state, self.actions[action_index])
    
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

tower = TowerOfHanoi([["A", "B"], [], []])
learner = Learner(tower)
learner.q_learning()