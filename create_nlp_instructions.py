spell={1:'first',2:'second',3:'third',4:'fourth',5:'fifth',
       6:'sixth',7:'seventh',8:'eighth',9:'ninth'}        
box_name = {0: 'None', 1: 'hundred', 2: 'ten',
            3: 'unit'}


class CreateInstructions:
####################################################
    def __init__(self, no, type) -> None:
        self.no = no
        self.type = type
        if self.type == 'policy':
            self.instructions, self.exp_actions = self.create_nlp_policy()
            self.instr_cnt, self.action_cnt = 0, 0
        if self.type == 'state':
            self.exp_actions = self.create_nlp_policy()[1]
            self.create_nlp_state()
            self.instr_cnt, self.action_cnt = 0, 0
####################################################  
    def incr(self):
        self.action_cnt += 1
        self.instr_cnt += 1
####################################################        
    def get_next_actions(self):
        if self.action_cnt >= len(self.exp_actions): return None
        return self.exp_actions[self.action_cnt]
####################################################      
    def split_no(self, no):
        hun,ten,uni = no//100, (no - (no//100)*100)//10, no%10
        return hun,ten,uni
####################################################     
    def get_word_name(self, no):
        digit10 = ["", "Twenty",  "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety" ]
        n20 = [  "",       "One",       "Two",      "Three",
                "Four",    "Five",      "Six",      "Seven",
                "Eight",   "Nine",      "Ten",      "Eleven",
                "Twelve",  "Thirteen",  "Fourteen", "Fifteen",
                "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
        hun, ten, uni = self.split_no(no)
        _name = ''
        if hun: 
            _name += n20[hun] + " hundred " 
        if ten:
            if ten*10<20:
                _name += n20[ten*10+uni]
            else:
                _name += digit10[ten-1] + " " + n20[uni]
        else:
            _name += n20[uni]
        return _name.lower(),hun,ten,uni
######################################################
    '''
    Policy based instructions
    ''' 
    def create_nlp_policy(self):
        global spell
        instructions = []
        exp_actions = []
        word_name,hun,ten,uni = self.get_word_name(self.no)
        if hun:
            exp_actions = exp_actions + [0,3]*hun
            instructions = instructions  + [f'Next , pick up the {spell[_//2+1]} hundred block .' if _%2 ==0  else 'Put the hundred block in the hundred\'s place .' for _ in range(2*hun)]
        if ten:
            exp_actions = exp_actions + [1,4]*ten
            instructions = instructions  + [f'Next , pick up the {spell[_//2+1]} ten block .' if _%2 ==0  else 'Put the ten block in the ten\'s place .' for _ in range(2*ten)]
        if uni:
            exp_actions = exp_actions + [2,5]*uni    
            instructions = instructions  +  [f'Next , pick up the {spell[_//2+1]} unit block .' if _%2 ==0  else 'Put the unit block in the unit\'s place .' for _ in range(2*uni)]       
        instructions[0] = f"This is {word_name}. Let's use our blocks to build the number. To build {word_name}"+instructions[0][len("Next"):]
        return instructions, exp_actions
#################################################### 
    '''
    State based instructions.
    '''
    def create_nlp_state(self):
        global spell 
        word_name,self.hun_state,self.ten_state,self.uni_state = self.get_word_name(self.no)
        self.common_prefix = f"This is {word_name}. " 
####################################################    
    def get_next_instructions(self, state_def):
        if self.type == 'policy':
            if self.instr_cnt >= len(self.instructions): return ''
            return self.instructions[self.instr_cnt]
        if self.type == 'state':
            if self.instr_cnt == 0: return self.common_prefix + "Let's use our blocks to build the number."
            x, y, z = self.hun_state - state_def[0][0], \
                      self.ten_state - state_def[0][1], \
                      self.uni_state - state_def[0][2]
            res_str = self.common_prefix + \
                      f"There are {x} blocks in hundred\'s place, {y} blocks" + \
                      f" in ten\'s place and {z} blocks in unit\'s place."
            if state_def[1] == True:
                res_str += f" You are holding a {box_name[state_def[2]]} block."
        return res_str 