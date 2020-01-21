import os,shutil
import random

def glob_format(path,base_name = False):
    print('--------pid:%d start--------------' % (os.getpid()))
    fmt_list = ('.jpg', '.jpeg', '.png',".xml")
    fs = []
    if not os.path.exists(path):return fs
    for root, dirs, files in os.walk(path):
        for file in files:
            item = os.path.join(root, file)
            # item = unicode(item, encoding='utf8')
            fmt = os.path.splitext(item)[-1]
            if fmt.lower() not in fmt_list:
                # os.remove(item)
                continue
            if base_name:fs.append(file)  # fs.append(os.path.splitext(file)[0])
            else:fs.append(item)
    print('--------pid:%d end--------------' % (os.getpid()))
    return fs

def splitData(dataPath:str,savePath:str):
    if not os.path.exists(savePath):os.makedirs(savePath)
    paths = glob_format(dataPath)
    random.seed(100)
    random.shuffle(paths)

    # 总数据只有`170`张，随机取出`150`张做训练，剩下的`20`张做验证
    for path in paths[:20]:
        imgName = os.path.basename(path)
        dirNmae = os.path.basename(os.path.dirname(path))

        # 剪切到新的目录
        newPath = os.path.join(savePath,dirNmae)
        if not os.path.exists(newPath): os.makedirs(newPath)
        shutil.move(path,os.path.join(newPath,imgName))

        # 对应的mask
        path = path.replace(dirNmae,"PedMasks").replace(".png","_mask.png")
        dirNmae = "PedMasks"
        imgName = os.path.basename(path)
        newPath = os.path.join(savePath, dirNmae)
        if not os.path.exists(newPath): os.makedirs(newPath)
        shutil.move(path, os.path.join(newPath, imgName))

if __name__=="__main__":
    splitData("../PennFudanPed/PNGImages","../valid")