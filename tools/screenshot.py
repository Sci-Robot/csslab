'''
目的：
    截图工具
    设定截图的初始位置，截图大小，延时截图，截图提示音

备注：
    如果在主程序中加入循环，就可以实现连续的截图工作了！
    这完全是现有软件没找到这个功能，为了偷懒才写的！！
'''
import os
import sys
import time
import winsound
from PIL import Image, ImageGrab

def screenshot(location,image_size,save_path,delay_time=4):
    '''
    :param location: 图片右上角位置，对应显示器中的位置（这个好像一般不容易知道），可能需要借助工具
    :param image_size: 图片大小
    :param save_path: 图片保存地址
    :param delay_time: 延迟时间
    :return: None
    '''
    SOUND = os.path.abspath(os.path.join('.','fb_notification.wav'))
    time.sleep(delay_time)
    box = (location[0], location[1], location[0] + image_size[0], location[1] + image_size[1])
    im = ImageGrab.grab(box)
    im.save(save_path)
    winsound.PlaySound(SOUND, winsound.SND_FILENAME)
    print('成功截图:',save_path)
    return im


def example():
    Location = (655, 123)
    image_size = ((755, 755))
    while True:
        name_num = input('输入保存文件号码(输入 shut 结束程序)：\n')
        if name_num == 'shut':
            break
        file_name = 'screenshot' + str(name_num) + '.png'
        FILE_DIR = r'G:\data'
        save_path = os.path.join(FILE_DIR, file_name)
        screenshot(location=Location,image_size=image_size,save_path=save_path)

if __name__ == '__main__':
    example()