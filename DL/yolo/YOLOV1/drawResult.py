import os
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np

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

def draw(imgs,nrows=4,ncols=5):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    axes = ax.flatten()
    for img,axe in zip(imgs,axes):
        axe.imshow(img)
        axe.axis("off")

    # plt.show()

if __name__=="__main__":
    paths = glob_format(r"C:\Users\MI\Documents\GitHub\result")
    imgs = [np.asarray(PIL.Image.open(path)) for path in paths]
    # plt.figure(figsize=(6, 6.5))
    draw(imgs[:10],2)
    # plt.figure(figsize=(6, 6.5))
    draw(imgs[10:],2)
    plt.show()