import gym
from gym import spaces
import pygame
import numpy as np
from enum import Enum 
import vision_pyGame as vga
from create_nlp_instructions import *
import math
'''
ACTIONS
'''
__STATE_DENSE__ = True # Dense reward
class ACTION(Enum):
    PICK_BIG=0
    PICK_MED=1
    PICK_SMALL=2
    PUT_BIG=3
    PUT_MED=4
    PUT_SMALL=5

'''
BOX TYPE
'''
class BOXTYPE(Enum):
    NONE=0
    BIG=1
    MEDIUM=2
    SMALL=3

class RlNlpWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1000000}
############################################
    def __init__(self,render_mode=None, instr_type = 'policy'):
        self.mode=0
        if render_mode=='rgb_array':
            self.mode=1
        self._visual=None
        self._text='#'
        self._question='?'
        # Just to remove assertion error. #
        self.action_space = spaces.Discrete(6) 
        self.observation_space=spaces.Dict({
            "text": spaces.Text(min_length=1,max_length=100),"question":spaces.Text(min_length=1,max_length=100),
            "visual": spaces.Box(low=0, high=255, shape=(vga.WIDTH,vga.HEIGHT,3), dtype=np.uint8)
        })
        self.instr_type = instr_type
############################################
    def _get_obs(self):
        return {"text": self._text,"question":self._question,"visual": self._visual}

    def _get_info(self):
        return {
            "progress": "currently feature not required."
        }
############################################
    def reset(self, set_no=-1, seed=None, options=None):
        super().reset(seed=seed)
        self.no=np.random.randint(1,1000) if set_no == -1 else set_no
        self.prev_no = 0
        self.dismantle_target = [self.no%10, (self.no//10)%10, (self.no//100)%100]
        self.nlp_obj = CreateInstructions(self.no, type = self.instr_type)
        self.mx_timeSteps,self.curr_time=math.ceil(sum(self.nlp_obj.split_no(self.no))*2*2.5),0 # 2.5 times is the buffer given to solve the problem
        ## Gen initial info ##
        self.carry=False
        self.blocksLeft=[ (self.no//10**i)%10 for i in range(3) ]
        self.blocksLeft.reverse()
        self.boxType=BOXTYPE.NONE
        self._visual=vga.draw_main(self.metadata['render_modes'][self.mode],self.metadata['render_fps'],self.no)
        self._text = self.nlp_obj.get_next_instructions(state_def = \
                                                        (self.blocksLeft, 
                                                         self.carry, 
                                                         self.boxType.value))
        self.progess = 0
        return self._get_obs()
############################################
    def step(self, action):
        if self.instr_type == 'policy': 
            return self.step_policy(action)
        elif self.instr_type == 'state': 
            return self.step_state(action)
        assert False, 'BAD INSTRUCTION TYPE'

############################################
    def step_state(self, action):
        terminated=False
        if action<0 or action>=6:
            reward = -1000
            terminated = True 
            return self._get_obs(), reward, terminated, self._get_info()
            
        def pick(boxArr,b_type):
            if self.carry==False:
                for box in boxArr:
                    if not box.isEmpty:
                        self.carry=True 
                        self.boxType=b_type
                        box.isEmpty=True
                        return
            
        def put(b_type):
            if self.carry==False or self.boxType!=b_type: return
            self.boxType=BOXTYPE.NONE 
            self.carry=False 
            # constructArrElement is where the blocks are put. 
            self.blocksLeft[b_type.value-1]-=1
            for box in vga.constructArrElement[b_type.value-1]:
                if box.isEmpty:
                    box.isEmpty=False 
                    return
                
        def whichNo():
            result,power=0,100
            for c in vga.constructArrElement:
                cnt_box=0
                for box in c:
                    if not box.isEmpty:
                        cnt_box+=1
                result+=cnt_box*power 
                power/=10
            return result
        
        reward=0
        if action==ACTION.PICK_BIG.value:
            pick(vga.big_block,BOXTYPE.BIG)
        elif action==ACTION.PICK_MED.value:
            pick(vga.medium_block,BOXTYPE.MEDIUM)
        elif action==ACTION.PICK_SMALL.value:
            pick(vga.small_block,BOXTYPE.SMALL)
        elif action==ACTION.PUT_BIG.value:
            put(BOXTYPE.BIG)
        elif action==ACTION.PUT_MED.value:
            put(BOXTYPE.MEDIUM)
        elif action==ACTION.PUT_SMALL.value:
            put(BOXTYPE.SMALL)

        '''Extra reward for following instructions'''
        vga.carry_indicator=self.carry
        self._visual=vga.drawAgain()
        curr_no = whichNo()
        solution= True if curr_no == self.no else False
        self.curr_time += 1 
        if not __STATE_DENSE__:
            dismantle_curr_no = [curr_no%10, (curr_no//10)%10, (curr_no//100)%100]
            dismantle_prev_no = [self.prev_no%10, (self.prev_no//10)%10, (self.prev_no//100)%100]
            if (dismantle_curr_no[0] > dismantle_prev_no[0]) or \
            (dismantle_curr_no[1] > dismantle_prev_no[1]) or \
            (dismantle_curr_no[2] > dismantle_prev_no[2]):
                self.prev_no = curr_no
                reward = 1 
            if (dismantle_curr_no[0] > self.dismantle_target[0]) or \
            (dismantle_curr_no[1] > self.dismantle_target[1]) or \
            (dismantle_curr_no[2] > self.dismantle_target[2]):
                terminated = True        
            self.nlp_obj.incr()
            self._text = self.nlp_obj.get_next_instructions(state_def = \
                                                            (self.blocksLeft, 
                                                            self.carry, 
                                                            self.boxType.value))
        else:
            if action == self.nlp_obj.get_next_actions():
                self.nlp_obj.incr()
                reward = 1
            else: 
                terminated = True 
            self._text = self.nlp_obj.get_next_instructions(state_def = \
                                                            (self.blocksLeft, 
                                                            self.carry, 
                                                            self.boxType.value))
        if self.curr_time>self.mx_timeSteps or solution==True:
            terminated = True 
        if terminated:
            reward = 10 if solution else -10
            self.close()
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward/10, terminated, info
    
############################################
    def step_policy(self, action):
        terminated=False
        if action<0 or action>=6:
            reward = -1000
            terminated = True 
            return self._get_obs(), reward, terminated, self._get_info()
            
        def pick(boxArr,b_type):
            if self.carry==False:
                for box in boxArr:
                    if not box.isEmpty:
                        self.carry=True 
                        self.boxType=b_type
                        box.isEmpty=True
                        return
            
        def put(b_type):
            if self.carry==False or self.boxType!=b_type: return
            self.boxType=BOXTYPE.NONE 
            self.carry=False 
            # constructArrElement is where the blocks are put. 
            self.blocksLeft[b_type.value-1]-=1
            for box in vga.constructArrElement[b_type.value-1]:
                if box.isEmpty:
                    box.isEmpty=False 
                    return
                
        def checkSolution():
            result,power=0,100
            for c in vga.constructArrElement:
                cnt_box=0
                for box in c:
                    if not box.isEmpty:
                        cnt_box+=1
                result+=cnt_box*power 
                power/=10
            return True if self.no==result else False 
        
        reward=0
        if action==ACTION.PICK_BIG.value:
            pick(vga.big_block,BOXTYPE.BIG)
        elif action==ACTION.PICK_MED.value:
            pick(vga.medium_block,BOXTYPE.MEDIUM)
        elif action==ACTION.PICK_SMALL.value:
            pick(vga.small_block,BOXTYPE.SMALL)
        elif action==ACTION.PUT_BIG.value:
            put(BOXTYPE.BIG)
        elif action==ACTION.PUT_MED.value:
            put(BOXTYPE.MEDIUM)
        elif action==ACTION.PUT_SMALL.value:
            put(BOXTYPE.SMALL)

        '''Extra reward for following instructions'''
        vga.carry_indicator=self.carry
        self._visual=vga.drawAgain()
        solution=checkSolution() # return True is solution is correct 
        self.curr_time += 1
        if action == self.nlp_obj.get_next_actions():
            self.nlp_obj.incr()
            reward = 1
        else: 
            # If one instruction is not followed the task fails.
            terminated = True 
        self._text = self.nlp_obj.get_next_instructions(state_def = \
                                                        (self.blocksLeft, 
                                                         self.carry, 
                                                         self.boxType.value))
        if self.curr_time>self.mx_timeSteps or solution==True:
            terminated = True 
        if terminated:
            reward = 10 if solution else -10
            self.close()
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward/10, terminated, info
    
############################################
    @property
    def threshold_reward(self):
        return 10
############################################
    def render(self):
        pass 
############################################
    def _render_frame(self):
        pass
############################################
    def close(self):
        vga.close_pyame()
############################################

