import numpy as np
import matplotlib.pyplot as plt 
import pygame 

'''
Global Variables.
'''
MX_NO_OF_DIGITS=3
FPS=1000000
rm=0
frac=0.25
# WIDTH, HEIGHT = int(900*frac), int(600*frac)
WIDTH, HEIGHT = 224, 224
WIN = None
constructArrElement=[[] for _ in range(MX_NO_OF_DIGITS)]
'''
COLORS.
'''
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 225)
GREEN = (0,255,0)
YELLOW = (200, 200, 0)
CYAN = (0,200,200)
MAGENTA = (200,0,200)
NEUTRAL = (128,128,128)
'''
BORDERS
'''
width_do,height_do=int(450*frac),int(150*frac)
DIGIT_OUTER=None
width_di,height_di=int(100*frac),int(100*frac)
'''
FONT
'''
FONT_SIZE=int(50*frac)
FONT = None
'''
CONSTRUCTION_BOX
'''
w_b,h_b=int(32*frac),int(32*frac)
w_m,h_m=int(32*frac),int(32*frac)
w_s,h_s=int(32*frac),int(32*frac)
'''
CONSTRUCT BOX
'''
w_cb,h_cb=int(100*frac),int(200*frac)
big_block=[];medium_block=[];small_block=[]

class RectNode:
    def __init__(self,x,y,width,height,isEmpty) -> None:
        self.rect=pygame.Rect(x,y,width,height)
        self.isEmpty=isEmpty
        
offset=int(10*frac)
w_carry,h_carry=int(128*frac),int(128*frac)
carry_indicator, carry_rect = False, pygame.Rect(90*offset,45*offset,w_carry,h_carry)    
def drawRectangleBorder(rect_,color,width):
    a=(rect_.left,rect_.top)
    b=(rect_.left+rect_.width-width,rect_.top)
    c=(rect_.left,rect_.top+rect_.height-width)
    pygame.draw.rect(WIN,color,pygame.Rect(a[0],a[1],rect_.width,width))
    pygame.draw.rect(WIN,color,pygame.Rect(a[0],a[1],width,rect_.height))
    pygame.draw.rect(WIN,color,pygame.Rect(b[0],b[1],width,rect_.height))
    pygame.draw.rect(WIN,color,pygame.Rect(c[0],c[1],rect_.width,width))

def drawWindowOneTime(no_list):
    global constructArrElement,big_block,medium_block,small_block
    WIN.fill(WHITE)
    drawRectangleBorder(DIGIT_OUTER,BLACK,2)
    gap_x=(width_do/3-width_di)/2
    gap_y=(height_do-height_di)/2
    rectArr=[]
    rx,ry=WIDTH/2-width_do/2,HEIGHT/2-height_do/2
    for _ in range(MX_NO_OF_DIGITS):
        digit_inner=pygame.Rect(rx+gap_x,ry+gap_y,width_di,height_di)
        rectArr.append(digit_inner)
        drawRectangleBorder(digit_inner,BLUE,2)
        rx+=2*gap_x+width_di
    f_indx=0
    for indx in range(len(rectArr)-1,-1,-1):
        x_co=rectArr[indx][0]+(rectArr[indx].width-FONT_SIZE/2)/2
        y_co=rectArr[indx][1]+(rectArr[indx].height-FONT_SIZE)/2
        WIN.blit(FONT.render(str(no_list[f_indx]),True,BLACK),(x_co,y_co))
        f_indx+=1
        if f_indx>=len(no_list):
            break
# BIG BOX
    init_x=offset
    y_co=offset
    for i in range(MX_NO_OF_DIGITS):
        x_co=init_x
        for j in range(MX_NO_OF_DIGITS):
            big_block.append(RectNode(x_co,y_co,w_b,h_b,False))
            x_co+=w_b+offset
        y_co+=h_b+offset
# MEDIUM BOX
    init_x+=MX_NO_OF_DIGITS*offset+MX_NO_OF_DIGITS*(offset+w_b)
    y_co=offset
    for i in range(MX_NO_OF_DIGITS):
        x_co=init_x
        for j in range(MX_NO_OF_DIGITS):
            medium_block.append(RectNode(x_co,y_co,w_m,h_m,False))
            x_co+=w_m+offset
        y_co+=h_m+offset
# SMALL BOX
    init_x+=MX_NO_OF_DIGITS*offset+MX_NO_OF_DIGITS*(offset+w_m)
    y_co=offset
    for i in range(MX_NO_OF_DIGITS):
        x_co=init_x
        for j in range(MX_NO_OF_DIGITS):
            small_block.append(RectNode(x_co,y_co,w_s,h_s,False))
            x_co+=w_s+offset
        y_co+=h_s+offset
# Construct Box
    gap_x=(width_do/3-w_cb)/2
    gap_y=(height_do-h_cb)/2
    rx,ry=WIDTH/2-width_do/2,HEIGHT/1.15-h_cb/2
    c_gap=2
    for _ in range(MX_NO_OF_DIGITS):
        digit_inner=pygame.Rect(rx+gap_x,ry+gap_y,w_cb,h_cb)
        drawRectangleBorder(digit_inner,BLUE,2)
        y_co=digit_inner[1]+c_gap
        for i in range(5):
            x_co=digit_inner[0]+c_gap
            for j in range(2):
                if _%3==0:
                    obj=RectNode(x_co,y_co,w_b,h_b,True)
                    x_co+=w_b+c_gap
                    y_temp=h_b
                elif _%3==1:
                    obj=RectNode(x_co,y_co,w_m,h_m,True)
                    x_co+=w_m+c_gap
                    y_temp=h_b
                else:
                    obj=RectNode(x_co,y_co,w_s,h_s,True)   
                    x_co+=w_s+c_gap     
                    y_temp=h_b            
                constructArrElement[_].append(obj)
            y_co+=c_gap+y_temp
        rx+=2*gap_x+width_di 
    rgb_array = pygame.surfarray.array3d(WIN)
    if rm=='human':
        pygame.display.update()
    return rgb_array

def drawAgain():
    for __ in big_block:
        if __.isEmpty==False:
            pygame.draw.rect(WIN,CYAN,__.rect)
        else:
            pygame.draw.rect(WIN,WHITE,__.rect)
    for __ in medium_block:
        if __.isEmpty==False:
            pygame.draw.rect(WIN,MAGENTA,__.rect) 
        else:
            pygame.draw.rect(WIN,WHITE,__.rect)
    for __ in small_block:
        if __.isEmpty==False:
            pygame.draw.rect(WIN,YELLOW,__.rect)
        else:
            pygame.draw.rect(WIN,WHITE,__.rect)
    color=[CYAN,MAGENTA,YELLOW]
    for id,c in enumerate(constructArrElement):
        for __ in c:
            if __.isEmpty==False:
                pygame.draw.rect(WIN,color[id],__.rect)
    if carry_indicator:
        pygame.draw.rect(WIN,BLACK,carry_rect)
    else:
        pygame.draw.rect(WIN,WHITE,carry_rect)
    rgb_array = pygame.surfarray.array3d(WIN)
    if rm=='human':
        pygame.display.update()
    return rgb_array

def draw_main(render_mode,fps,no):
    global FPS,rm,big_block,medium_block,small_block,constructArrElement,WIN,\
           DIGIT_OUTER,FONT, carry_indicator
    big_block=[];medium_block=[];small_block=[]
    constructArrElement=[[] for _ in range(MX_NO_OF_DIGITS)]
    carry_indicator = False
    FPS,rm=fps,render_mode
    pygame.init()
    if rm=='human':
        WIN = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("NLP_RL_GAME")
    else:
        WIN = pygame.Surface((WIDTH, HEIGHT))
    DIGIT_OUTER=pygame.Rect(WIDTH/2-width_do/2,HEIGHT/2-height_do/2,width_do,height_do)
    FONT = pygame.font.SysFont('Arial',FONT_SIZE)
    no_list=[]
    while no!=0:
        no_list.append(no%10)
        no=no//10
    drawWindowOneTime(no_list)
    if True:
        return drawAgain()

def close_pyame():
    pygame.quit()

# if __name__=='__main__':
#     carry_indicator=True
#     plt.imsave('how_image_looks.png',draw_main("rgb_array",1000000,969))
    