import os

import threading

import multiprocessing

length_of_folder = 0

def copyfile(Path):
    if os.path.isdir(Path):
        print("-----------%s" % ("Testfortherading_" + '/' + Path[length_of_folder:]))
        os.makedirs("Testforthreading_" + '/' + Path[length_of_folder:])
        filenames = os.listdir(Path)
        for filename in filenames:
            if os.path.isdir(Path + '/' + filename):
                #ps = "Testforthreading_" +"/" + Path[length_of_folder:]
                #print("%s" % (ps + '/' + filename))
                #os.mkdir(ps + '/' + filename)
                temp = Path + '/' + filename
                t = threading.Thread(target=copyfile , args=(temp,))
                t.start()
            else:
                f = open(Path + '/' + filename , 'rb')
                content = f.read()
                F = open('Testforthreading_' + '/' + Path[length_of_folder:]+ '/' + filename , 'wb')
                F.write(content)
                f.close()
                F.close()
def main():

    """"""
    foldername = input("Please input the folder you want to copy:")
    length_of_folder = len(foldername)
    if os.path.isdir("Testforthreading_"):
        os.removedirs("Testforthreading_")
        os.mkdir("Testforthreading_")
        copyfile(foldername)
    #p = multiprocessing.Pool(10)
    #que = multiprocessing.Manager().Queue()
if __name__ == "__main__":
    main()

# ps：Python多进程递归复制文件夹中的文件

import multiprocessing
import os
import reimport, time

# 源文件夹地址、目标文件夹地址

SOUR_PATH = ""
DEST_PATH = ""
# 源文件列表 文件夹列表
SOUR_FILE_LIST = list()
SOUR_DIR_LIST = list()
def traverse(source_path):
    """递归遍历源文件夹，获取文件夹列表、文件列表
    :param source_path: 用户指定的源路径
    """
    if os.path.isdir(source_path):
        SOUR_DIR_LIST.append(source_path)
        for temp in os.listdir(source_path):
            new_source_path = os.path.join(source_path, temp)
            traverse(new_source_path)
    else:
        SOUR_FILE_LIST.append(source_path)
def copy_files(queue, sour_file, dest_file):
    """复制文件列表中的文件到指定文件夹
    :param queue: 队列，用于监测进度
    :param sour_file:
    :param dest_file:
    """
    # time.sleep(0.1)
    try:
        old_f = open(sour_file, "rb")
        new_f = open(dest_file, "wb")
    except Exception as ret:
        print(ret)
    else:
        content = old_f.read()
        new_f.write(content)
    old_f.close()
    new_f.close()
    queue.put(sour_file)
def main():
    source_path = input("请输入需要复制的文件夹的路径：\n")
    SOUR_PATH = source_path
    DEST_PATH = SOUR_PATH + "[副本]"
    # dest_path = input("请输入目标文件夹路径")
    # DEST_PATH = dest_path
    print(">>>源文件夹路径：", SOUR_PATH)
    print(">目标文件夹路径：", DEST_PATH)
    print("开始计算文件...")
    queue = multiprocessing.Manager().Queue()
    po = multiprocessing.Pool(5)
    traverse(source_path)
    print("创建目标文件夹...")
    for sour_dir in SOUR_DIR_LIST:
        dest_dir = sour_dir.replace(SOUR_PATH, DEST_PATH)
        try:
            os.mkdir(dest_dir)
        except Exception as ret:
            print(ret)
        else:
            print("\r目标文件夹 %s 创建成功" % DEST_PATH, end="")
            print()
            print("开始复制文件")
    for sour_file in SOUR_FILE_LIST:
        dest_file = sour_file.replace(SOUR_PATH, DEST_PATH)
        po.apply_async(copy_files, args=(queue, sour_file, dest_file))
        count_file = len(SOUR_FILE_LIST)
        count = 0
        while True:
            q_sour_file = queue.get()
            if q_sour_file in SOUR_FILE_LIST:
                count += 1
                rate = count * 100 / count_file
                print("\r文件复制进度： %.2f%% %s" % (rate, q_sour_file), end="")
            if rate >= 100:
                break
    print()
    ret = re.match(r".*\\([^\\]+)", SOUR_PATH)
    name = ret.group(1)
    print("文件夹 %s 复制完成" % name)
if __name__ == '__main__':

    main()