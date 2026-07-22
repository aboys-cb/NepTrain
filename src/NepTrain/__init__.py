#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 16:22
# @Author  : 兵
# @email    : 1747193328@qq.com
import configparser
import os
import shutil



from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("NepTrain")
except PackageNotFoundError:
    __version__ = "0+unknown"
config_path = os.path.join(os.path.expanduser("~"), ".NepTrain")

module_path = os.path.dirname(__file__)

if not os.path.exists(config_path)  :
    shutil.copy(os.path.join(module_path,"config.ini"), config_path)

Config = configparser.RawConfigParser()
Config.read(config_path,encoding="utf8")



#
# if platform.is_linux():
#     # wsl测试默认的有问题  强行切换到poll机制
#     observer = PollingObserver()
# else:
#     observer = Observer()


try:
    from watchdog.observers import Observer

    observer = Observer()
except ImportError:
    observer = None
