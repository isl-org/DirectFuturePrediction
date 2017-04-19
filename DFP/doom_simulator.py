'''
ViZDoom wrapper
'''
from __future__ import print_function
import sys
import os

vizdoom_path = '../../../../toolboxes/ViZDoom_2017_03_31'
sys.path = [os.path.join(vizdoom_path,'bin/python3')] + sys.path

import vizdoom 
print(vizdoom.__file__)
import random
import time
import numpy as np
import re
import cv2

class DoomSimulator:
    
    def __init__(self, args):        
        self.config = args['config']
        self.resolution = args['resolution']
        self.frame_skip = args['frame_skip']
        self.color_mode = args['color_mode']
        self.switch_maps = args['switch_maps']
        self.maps = args['maps']
        self.game_args = args['game_args']
        
        self._game = vizdoom.DoomGame()
        self._game.set_vizdoom_path(os.path.join(vizdoom_path,'bin/vizdoom'))
        self._game.set_doom_game_path(os.path.join(vizdoom_path,'bin/freedoom2.wad'))
        self._game.load_config(self.config)
        self._game.add_game_args(self.game_args)
        self.curr_map = 0
        self._game.set_doom_map(self.maps[self.curr_map])
        
        # set resolution
        try:
            self._game.set_screen_resolution(getattr(vizdoom.ScreenResolution, 'RES_%dX%d' % self.resolution))
            self.resize = False
        except:
            print("Requested resolution not supported:", sys.exc_info()[0], ". Setting to 160x120 and resizing")
            self._game.set_screen_resolution(getattr(vizdoom.ScreenResolution, 'RES_160X120'))
            self.resize = True

        # set color mode
        if self.color_mode == 'RGB':
            self._game.set_screen_format(vizdoom.ScreenFormat.CRCGCB)
            self.num_channels = 3
        elif self.color_mode == 'GRAY':
            self._game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
            self.num_channels = 1
        else:
            print("Unknown color mode")
            raise

        self.available_controls, self.continuous_controls, self.discrete_controls = self.analyze_controls(self.config)
        self.num_buttons = self._game.get_available_buttons_size()
        assert(self.num_buttons == len(self.discrete_controls) + len(self.continuous_controls))
        assert(len(self.continuous_controls) == 0) # only discrete for now
        self.num_meas = self._game.get_available_game_variables_size()
            
        self.meas_tags = []
        for nm in range(self.num_meas):
            self.meas_tags.append('meas' + str(nm))
            
        self.episode_count = 0
        self.game_initialized = False
        
    def analyze_controls(self, config_file):
        with open(config_file, 'r') as myfile:
            config = myfile.read()
        m = re.search('available_buttons[\s]*\=[\s]*\{([^\}]*)\}', config)
        avail_controls = m.group(1).split()
        cont_controls = np.array([bool(re.match('.*_DELTA', c)) for c in avail_controls])
        discr_controls = np.invert(cont_controls)
        return avail_controls, np.squeeze(np.nonzero(cont_controls)), np.squeeze(np.nonzero(discr_controls))
        
    def init_game(self):
        if not self.game_initialized:
            self._game.init()
            self.game_initialized = True
            
    def close_game(self):
        if self.game_initialized:
            self._game.close()
            self.game_initialized = False
            
    def step(self, action=0):
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
        self.init_game()
        
        rwrd = self._game.make_action(action, self.frame_skip)        
        state = self._game.get_state()
        
        if state is None:
            img = None
            meas = None
        else:        
            # ViZDoom 1.0
            #raw_img = state.image_buffer
                
            ## ViZDoom 1.1 
            if self.color_mode == 'RGB':
                raw_img = state.screen_buffer
            elif self.color_mode == 'GRAY':
                raw_img = np.expand_dims(state.screen_buffer,0)
                
            if self.resize:
                if self.num_channels == 1:
                    if raw_img is None or (isinstance(raw_img, list) and raw_img[0] is None):
                        img = None
                    else:
                        img = cv2.resize(raw_img[0], (self.resolution[0], self.resolution[1]))[None,:,:]
                else:
                    raise NotImplementedError('not implemented for non-Grayscale images')
            else:
                img = raw_img
                
            meas = state.game_variables # this is a numpy array of game variables specified by the scenario
            
        term = self._game.is_episode_finished() or self._game.is_player_dead()
        
        if term:
            self.new_episode() # in multiplayer multi_simulator takes care of this            
            img = np.zeros((self.num_channels, self.resolution[1], self.resolution[0]), dtype=np.uint8) # should ideally put nan here, but since it's an int...
            meas = np.zeros(self.num_meas, dtype=np.uint32) # should ideally put nan here, but since it's an int...
            
        return img, meas, rwrd, term
    
    def get_random_action(self):
        return [(random.random() >= .5) for i in range(self.num_buttons)]
        
    def is_new_episode(self):
        return self._game.is_new_episode()
    
    def next_map(self):     
        if self.switch_maps:
            self.curr_map = (self.curr_map+1) % len(self.maps)
            self._game.set_doom_map(self.maps[self.curr_map])
    
    def new_episode(self):
        self.next_map()
        self.episode_count += 1
        self._game.new_episode()
