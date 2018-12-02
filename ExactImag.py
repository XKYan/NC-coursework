import os
import shutil


path_img='./test'
keyWord = '0_level'
path_destination = './level0_test/'

ls = os.listdir(path_img)
print(len(ls))
for i in ls:
    if i.find(keyWord)!=-1:
        shutil.move(path_img+'/'+i,path_destination+i)
    