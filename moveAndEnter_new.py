'''
this file is for following tasks:
1using the Euler angle for moving operation.
2using the Euler angle for mouse-moving operation.
3others...

        liangzid
'''

import pynput
import time 

class MoveAndEnter():


    def __init__(self,alpha,beta,baseline):
        '''
        :param alpha:最小阈值调节
        :param beta:最大阈值调节
        '''

        self.MinNum=alpha
        self.MaxNum=beta

        '''
        ：防止一次移动多次，设置标志位
        ：右转标志位 yaw < 0
        ：左转标志位 yaw > 0
        ：低头标志位 pitch 
        ：抬头标志位 pitch
        ：姿态复位标志位设为False
        '''
        self.RightFlag = False
        self.LeftFlag = False
        self.DownFlag = False
        self.UpFlag = False

        '''
        :设置菜单级数标志位，只有标志位为1 时可以下移， 只有标志位为2时可以上移
        :初始化为1级,一共两级菜单
        '''
        self.menuFlag = 1

        if self.MinNum>self.MaxNum:
            print('错误!最大阈值应高于最小阈值')

        # 基准,正常姿态下的欧拉角
        self.angleBaseline=baseline

        # 此处还需要有判别二者是否在0和1之间的程序

        # 欧拉角的范围 yaw , pitch ,roll 偏行角。俯仰角。滚动角。
        self.Maxangles=[75,60,50]
        self.Minangles=[-75,-60,-50]

        self.Maxangles = [self.Maxangles[0] -self.angleBaseline[0], self.Maxangles[1] - self.angleBaseline[1] , self.Maxangles[2] -self.angleBaseline[2]]
        self.Minangles = [self.Minangles[0] -self.angleBaseline[0], self.Minangles[1] - self.angleBaseline[1] , self.Minangles[2] -self.angleBaseline[2]]



        #键盘操作所需要的
        self.keyboard=pynput.keyboard.Controller()
        self.moveRight=pynput.keyboard.Key.right
        self.moveLeft=pynput.keyboard.Key.left
        self.Enter=[pynput.keyboard.Key.tab,pynput.keyboard.Key.page_up]
        self.exit=[pynput.keyboard.Key.tab,pynput.keyboard.Key.page_down]
        self.tab = pynput.keyboard.Key.tab 
        self.Confirm = pynput.keyboard.Key.space

        #鼠标操作所需要的
        self.mouse=pynput.mouse.Controller()

        self.k=0.1   #这个需要根据实际情况进行设计而非简单地采用一个数值


    def isInTernal(self,angles,):
        '''
        判断一个角度是否在区间范围内的函数
        :param angle:
        :return:
        '''

    def movee(self,angles):
        '''
        angles is :pitch,roll,yaw
        :param angles:
        :return:
        '''
        # 判断角度是否有问题
        self.isInTernal(angles)
        # 判断是否超过了所需要的阈值
        #print(angles)
        diffAngles=[angles[0]-self.angleBaseline[0],angles[1]-self.angleBaseline[1],angles[2]-self.angleBaseline[2]]
        # 判断是否复位
        if self.abs(diffAngles[2]) < self.MaxNum * self.Maxangles[2]:
            self.LeftFlag = False
            self.RightFlag = False

        if self.abs(diffAngles[1]) < self.MaxNum * 0.7 *self.Maxangles[1]:
            self.DownFlag = False
            self.UpFlag = False
        # 滚动角向左，控制键盘点击left
        if diffAngles[2] > self.MaxNum *self.Maxangles[2] and self.LeftFlag == False:
            time.sleep(0.01)
            if diffAngles[2] > self.MaxNum *self.Maxangles[2]:
                self.keyboard.press(self.moveLeft)
                #sleep(xs)
                self.keyboard.release(self.moveLeft)
                print("move Left")
                self.LeftFlag = True
                return 0
        #滚动角向右，控制键盘点击right
        if diffAngles[2] < self.MaxNum * self.Minangles[2] and self.RightFlag == False:
            time.sleep(0.01)
            if diffAngles[2] < self.MaxNum * self.Minangles[2]:
                self.keyboard.press(self.moveRight)
                #sleep(xs)
                self.keyboard.release(self.moveRight)
                print("move Right")
                self.RightFlag = True
                return 0
        #俯仰角上，控制菜单级别向上
        if diffAngles[1] > self.MaxNum *self.Maxangles[1] and self.UpFlag == False and self.menuFlag == 2:
            time.sleep(0.01)
            if diffAngles[1] > self.MaxNum *self.Maxangles[1]:

                self.keyboard.press(self.tab)
                #sleep(xs)
                self.keyboard.release(self.tab)

                self.keyboard.press(self.tab)
                #sleep(xs)
                self.keyboard.release(self.tab)
                print("move up")
                self.UpFlag = True
                self.menuFlag = 1
                return 0
        #俯仰角向下，控制菜单级别向下
        if diffAngles[1] < self.MaxNum * self.Minangles[1] and self.DownFlag == False and self.menuFlag == 1:
            time.sleep(0.01)
            if diffAngles[1] < self.MaxNum * self.Minangles[1]:
                self.keyboard.press(self.tab)
                #sleep(xs)
                self.keyboard.release(self.tab)
                print("move Down")
                self.DownFlag = True
                self.menuFlag = 2
                return 0
        # 俯仰角向下确认，确认按钮
        if diffAngles[1] < self.MaxNum * self.Minangles[1] and self.DownFlag == False and self.menuFlag == 2:        
            time.sleep(0.01)
            if diffAngles[1] < self.MaxNum * self.Minangles[1]:
                self.keyboard.press(self.Confirm)
                #sleep(xs)
                self.keyboard.release(self.Confirm)
                print("Choose")
                self.DownFlag = True
                self.menuFlag = 2
                return 0
        return 0

    def abs(self,Num):
        if Num <= 0:
            return -Num
        else:
            return Num

    def mouse_moving(self,angles_past,angles):
        zelta_angle=angles-angles_past

        if angles[2]-self.angleBaseline[2]<self.MinNum*self.Minangles[2]:
            # 左键按下
            self.mouse.press(pynput.mouse.Button.left)

        elif angles[2] - self.angleBaseline[2] > self.MinNum * self.Maxangles[2]:
            # 右键按下
            self.mouse.press(pynput.mouse.Button.right)

        elif angles[2] - self.angleBaseline[2] > self.MinNum * self.Minangles[2] & angles[2] - self.angleBaseline[2] <0:
            # 左键松开
            self.mouse.release(pynput.mouse.Button.left)

        elif angles[2] - self.angleBaseline[2] > self.MinNum * self.Minangles[2] & angles[2] - self.angleBaseline[2] < 0:
            # 右键松开
            self.mouse.release(pynput.mouse.Button.right)



        # 鼠标移动操作
        zelta=[zelta_angle[0],zelta_angle[1]]


        # 根据变化的向量移动鼠标
        self.mouse.move(zelta[0]*self.k,zelta[1]*self.k)



