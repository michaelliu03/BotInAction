# -*- coding:utf-8 -*-


import os
import re

pathlist = []
def eachFile(filepath):
    fileNames = os.listdir(filepath)  # 获取当前路径下的文件名，返回List
    for file in fileNames:
        newDir = os.path.join(filepath, file) # 将文件命加入到当前文件路径后面
        if os.path.isfile(newDir):  # 如果是文件
            pathlist.append(newDir)
        else:
            eachFile(newDir)
    return pathlist

def processFile(fr,fw1,label):
    global fwl
    with open(fr,"r",encoding='UTF-8') as fr1:
        content=fr1.read()
        pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        fw1.write(label+"\t"+pattern.sub("",str(content).replace("\n"," ")))
        fw1.write("\n")

if __name__ == '__main__':
    filePath = "D:\\liuyu\\桌面\\git\\BotInAction\\data\\chapter4\\example2\\"
    eachFile(filePath)
    print(pathlist)
    for path in pathlist:
        label=path.split("\\")[-2]
        label='__label__'+label
        # try:
        fwl = open("", "a", encoding='UTF-8')
        processFile(path,fwl,label)
        # except:
        #     print(path)
