#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 11/13/25
Description: np_where_condition_example
"""

import numpy as np
from numpy.ma.core import indices


def demo_np_where():
    # 1.基本用法，大于 25的标记为1，否则为0
    arr = np.array([10, 20, 30, 40, 50])
    results = np.where(arr > 25, 1, 0)
    print("示例1：基本用法", results)

    # 2. 替换负数为0
    arr = np.array([1, -3, 8, -9, 0, 12])
    results2 = np.where(arr < 0, 0, arr)
    print("example 2:", results2)

    # 3. 使用不同的数组作为x, y
    a = np.array([1, 2, 3])
    b = np.array([100, 200, 300])
    condition = np.array([True, False, True])
    results3 = np.where(condition, a, b)
    print("example 3:", results3)
    # 4. 获取索引 位置
    arr2 = np.array([10, 20, 30, 40, 50])
    indices = np.where(arr2 > 25)
    print("example 4:", indices)


def main():
    demo_np_where()


if __name__ == "__main__":
    main()
