'''
Several doom simulators running otgether
'''
from __future__ import print_function
from .doom_simulator import DoomSimulator

class MultiDoomSimulator:
	
	def __init__(self, all_args):
		
		self.num_simulators = len(all_args)
		self.simulators = []
		for args in all_args:
			self.simulators.append(DoomSimulator(args))
			
		self.resolution = self.simulators[0].resolution
		self.num_channels = self.simulators[0].num_channels
		self.num_meas = self.simulators[0].num_meas
		self.action_len = self.simulators[0].num_buttons
		self.config = self.simulators[0].config
		self.maps = self.simulators[0].maps
		self.continuous_controls = self.simulators[0].continuous_controls
		self.discrete_controls = self.simulators[0].discrete_controls
			
	def step(self, actions):
		"""
		Action can be either the number of action or the actual list defining the action
		
		Args:
			action - action encoded either as an int (index of the action) or as a bool vector
		Returns:
			img  - image after the step
			meas - numpy array of returned additional measurements (e.g. health, ammo) after the step
			rwrd - reward after the step
			term - if the state after the step is terminal
		"""
		assert(len(actions) == len(self.simulators))
		
		imgs = []
		meass = []
		rwrds = []
		terms = []
		
		for (sim,act) in zip(self.simulators, actions):
			img, meas, rwrd, term = sim.step(act)
			imgs.append(img)
			meass.append(meas)
			rwrds.append(rwrd)
			terms.append(term)
			
		return imgs, meass, rwrds, terms
	
	def num_actions(self, nsim):
		return self.simulators[nsim].num_actions
	
	def get_random_actions(self):
		acts = []
		for sim in self.simulators:
			acts.append(sim.get_random_action())
		return acts
