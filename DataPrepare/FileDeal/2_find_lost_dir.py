import os

"""用于比对文件夹，寻找缺少的子文件夹"""
root_path1 = r'H:\sdfy_data\Pelvic2\CBCT'
root_path2 = r'H:\sdfy_data\Pelvic2\CT'

items1 = os.listdir(root_path1)
items2 = os.listdir(root_path2)
items_all = list(set(items1 + items2))

for item in items_all:
    if item not in items1:
        print('路径1中缺少子文件夹:%s' % item)
    if item not in items2:
        print('路径2中缺少子文件夹:%s' % item)
    # end if
# end for

