import pygame,sys
import numpy as np
from pygame.locals import *
from keras.models import load_model
import cv2
from pygame import image

windowsizex= 640
windowsizey=480
bound_inc=5
img_count=1
PREDICT= True

WHITE=(255,255,255)
BLACK=(0,0,0)
RED=(255,0,0)

IMGSAVE=False

Model=load_model("amodel.h5")

Labels={0:"Zero", 1:"One",
        2:"Two", 3:"Three",
        4:"Four",5:"Five",
        6:"Six",7:"Seven",
        8:"Eight",9:"Nine"}


#Initializing pygame module
pygame.init()
font=pygame.font.Font("Knowledge Power.ttf",48)
DISPLAYSURF = pygame.display.set_mode((windowsizex,windowsizey))
#backg=DISPLAYSURF.mp_rgb(WHITE)
pygame.display.set_caption("Digit-Board")

iswriting = False

num_xcord=[]
num_ycord=[]

while True:
    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and iswriting :
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF,WHITE,(xcord,ycord),4,0)

            num_xcord.append(xcord)
            num_ycord.append(ycord)
        if event.type == MOUSEBUTTONDOWN:
            iswriting=True
        if event.type == MOUSEBUTTONUP:
            iswriting=False
            num_xcord=sorted(num_xcord)
            num_ycord=sorted(num_ycord)

            Rect_minx,Rect_maxx=max(num_xcord[0]-bound_inc,0),min(windowsizex,num_xcord[-1]+bound_inc)
            Rect_miny,Rect_maxy=max(num_ycord[0]-bound_inc,0),min(num_ycord[-1]+bound_inc,windowsizex)

            num_xcord=[]
            num_ycord=[]

            img_arr=np.array(pygame.PixelArray(DISPLAYSURF))[Rect_minx:Rect_maxx,Rect_miny:Rect_maxy].T.astype(np.float32)

            if IMGSAVE:
                cv2.imwrite("image.png")
                img_count +=1

            if PREDICT:
                image =cv2.resize(img_arr,(28,28))
                image =np.pad(image,(10,10),'constant',constant_values=0)
                image =cv2.resize(image,(28,28))/255

                label=str(Labels[np.argmax(Model.predict(image.reshape(1,28,28,1)))])

                textsurface= font.render(label,True,BLACK,WHITE)
                textrecobj = textsurface.get_rect()
                textrecobj.left, textrecobj.bottom= Rect_minx,Rect_maxy

                DISPLAYSURF.blit(textsurface,textrecobj)
            if event.type== KEYDOWN:
                if event.unicode=="n":
                    DISPLAYSURF.fill(BLACK)

        pygame.display.update()



        


