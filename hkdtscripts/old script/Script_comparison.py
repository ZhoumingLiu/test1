# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 14:29:27 2024

@author: SJ CHAI
"""

import difflib


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.readlines()

# 读取文件内容
start20230728 = read_file('Start20230728.py')
start = read_file('Start.py')

# 比较文件
differences = difflib.unified_diff(start20230728, start, fromfile='Start20230728.py', tofile='Start.py')

# 输出差异
for line in differences:
    print(line)
