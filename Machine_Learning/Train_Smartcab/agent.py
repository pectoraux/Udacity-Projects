import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple
from random import uniform

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.previous_reward = 0
        self.previous_action = None
        self.success_so_far = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.previous_reward = 0
        self.previous_action = None
        self.state = None
    
    def make_state(self, inputs):
    	next_waypoint = self.planner.next_waypoint()
    	behavior = 'Normal'
    	deadline = self.env.get_deadline(self)
    	if deadline > 38:
    		behavior = 'Patient'
    	elif deadline > 14:
    		behavior = 'Normal'
    	else:
    		behavior = 'Urgent' 		
    	State = namedtuple('State', ['light', 'next_waypoint', 'behavior'])
    	return State(inputs['light'], next_waypoint, behavior)			

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.make_state(inputs)

        # TODO: Select action according to your policy
        action = random.choice([None,'forward','left','right'])

        # Execute action and get reward
        reward = self.env.act(self, action)
       	
       	# Keeping track of number of times agent reaches destination
        if self.env.done == True:
        	self.success_so_far += 1
       
        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): state = {}, deadline = {}, inputs = {}, action = {}, reward = {}".format(self.state, deadline, inputs, action, reward)  # [debug]


class QLearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(QLearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.remaining_time = []
        self.negative_rewards = 0
        self.positive_rewards = 0
        self.previous_reward = None
        self.previous_action = None
        self.success_so_far = 0
        self.valid_actions = [None, 'forward', 'right', 'left']
        self.previous_state = None
        self.state = None
        self.alpha = .5
        self.epsilon = .3
        self.gamma = 0.2
        self.q_dict = dict()

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.previous_action = None
        self.previous_state = None
        self.state = None
        self.epsilon = 0.01  # Setting epsilon any higher seems to be decreasing the performance 

    def flip_coin(self, p):
    	return (random.random() < p)

    def make_state(self, inputs):
    	next_waypoint = self.planner.next_waypoint()
    	behavior = 'Normal'
    	deadline = self.env.get_deadline(self)
    	if deadline > 38:
    		behavior = 'Patient'
    	elif deadline > 14:
    		behavior = 'Normal'
    	else:
    		behavior = 'Urgent' 		
    	State = namedtuple('State', ['light', 'next_waypoint', 'behavior'])
    	return State(inputs['light'], next_waypoint, behavior)
    
    def get_Q(self, state, action):
    	#default q-value of 9, this value is kept low to help with convergence
    	return self.q_dict.get((state, action), 9)

    def get_max_Q(self, state):
    	q = [self.get_Q(state, action) for action in self.valid_actions]
    	return max(q)
    	
    def choose_action(self, state):  # TODO: Learn policy based on state, action, reward
    	maxQ = self.get_max_Q(state)
    	q = [self.get_Q(state, action) for action in self.valid_actions]
    	if self.flip_coin(self.epsilon):
    		best = range(len(self.valid_actions))
    		i = random.choice(best)
    	else:
    		# Chooses randomly between all maxQ's in the case of a tie -- helps with convergence
    		best = [i for i in range(len(self.valid_actions)) if q[i] == maxQ]
    		i = random.choice(best)
    	action = self.valid_actions[i]
    	return action 	 

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.make_state(inputs)

        # TODO: Select action according to your policy
        action = self.choose_action(self.state)      

        # Execute action and get reward
        reward = self.env.act(self, action)
       	
       	# Keeping track of negative and positive rewards
       	if reward < 0:
       		self.negative_rewards += reward
       	else:
       		self.positive_rewards += reward

       	# Keeping track of number of times agent reaches destination as well as the remaining time at that moment.
        if self.env.done == True:
        	self.success_so_far += 1
        	self.remaining_time.append(self.env.get_deadline(self))

        # Update q-table only for non-initial configurations
        if self.previous_reward != None:
        	self.update_Q_table(self.previous_state, self.previous_action, self.state, self.previous_reward)

        # Storing previous action and state
        self.previous_action = action
        self.previous_state = self.state
        self.previous_reward = reward

        print "QLearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def update_Q_table(self, state, action, next_state, reward):
    	if (state, action) not in self.q_dict:
    		self.q_dict[(state, action)] = 9   
    	else:
    	# Q(s, a) +=  alpha * [reward + gamma * max[Q(s', a)] - Q(s, a)]	
    		self.q_dict[(state, action)] += (self.alpha*(reward + self.gamma*self.get_max_Q(next_state)) -
    										self.q_dict[(state, action)])    									   


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    #a = e.create_agent(LearningAgent)  # create Learning agent
    a = e.create_agent(QLearningAgent)  # create Q-Learning agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0000005, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    n_trials = 100
    sim.run(n_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print "###################### Stats ################################"
    print "Performance of {} is : {}%".format(a.__class__.__name__, a.success_so_far/float(n_trials)*100)
    print "Total number of negative rewards is {}".format(a.negative_rewards)
    print "Total number of positive rewards is {}".format(a.positive_rewards)
    print "Net reward is thus {}".format(a.positive_rewards+a.negative_rewards)
    print "Average of all remaining times is {}".format(sum(a.remaining_time)/float(len(a.remaining_time)))

if __name__ == '__main__':
    run()
