#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 16:01
# @Author  : 兵
# @email    : 1747193328@qq.com
import glob
import os
import shutil
import subprocess
import traceback
from contextlib import contextmanager
import datetime

from pathlib import Path
from typing import Generator, Union

from ase.io import read as ase_read
from rich import get_console
from rich.progress import track

#前面几个0是为了让元素编号和索引对的上 避免了见一
radius_table = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96,
                'B': 0.85, 'C': 0.76, 'N': 0.71, 'O': 0.66,
                'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41,
                'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05,
                'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76,
                'Sc': 1.7, 'Ti': 1.6, 'V': 1.53, 'Cr': 1.39,
                'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24,
                'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.2,
                'As': 1.19, 'Se': 1.2, 'Br': 1.2, 'Kr': 1.16,
                'Rb': 2.2, 'Sr': 1.95, 'Y': 1.9, 'Zr': 1.75,
                'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46,
                'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
                'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38,
                'I': 1.39, 'Xe': 1.4, 'Cs': 2.44, 'Ba': 2.15,
                'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01,
                'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96,
                'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89,
                'Tm': 1.9, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75,
                'Ta': 1.7, 'W': 1.62, 'Re': 1.51, 'Os': 1.44,
                'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32,
                'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.4,
                'At': 1.5, 'Rn': 1.5, 'Fr': 2.6, 'Ra': 2.21,
                'Ac': 2.15, 'Th': 2.06, 'Pa': 2.0, 'U': 1.96,
                'Np': 1.9, 'Pu': 1.87, 'Am': 1.8, 'Cm': 1.69,
                'Bk': 1.5, 'Cf': 1.5, 'Es': 1.5, 'Fm': 1.5,
                'Md': 1.5, 'No': 1.5, 'Lr': 1.5, 'Rf': 1.5,
                'Db': 1.5, 'Sg': 1.5, 'Bh': 1.5, 'Hs': 1.5,
                'Mt': 1.5, 'Ds': 1.5, 'Rg': 1.5, 'Cn': 1.5,
                'Nh': 1.5, 'Fl': 1.5, 'Mc': 1.5, 'Lv': 1.5,
                'Ts': 1.5, 'Og': 1.5}

def print(*msg, **kwargs):

    get_console().print(f"[{datetime.datetime.now().strftime( '%Y-%m-%d %H:%M:%S' )}] -- ",*msg, **kwargs)


def print_warning(*msg):
    print(*msg, style="#fc5531")

def print_msg(*msg):
    print(*msg )

def print_tip(*msg):
    print(*msg)

def print_success(*msg):
    print(*msg, style="green")


def merge_yaml(yaml_a, yaml_b):
    result = yaml_a.copy()  # 复制a的内容
    for key, value in yaml_b.items():  # 遍历b的键值对
        if key in yaml_a and isinstance(yaml_a[key], dict) and isinstance(yaml_b[key], dict):
            result[key] = merge_yaml(yaml_a[key], yaml_b[key])  # 递归合并字典
        else:
            result[key] = value  # 否则直接覆盖
    return result


def get_config_path():
    return os.path.join(os.path.expanduser('~'),".NepTrain")

def verify_path(path):
    if not os.path.exists(os.path.expanduser(path)):

        os.makedirs(os.path.expanduser(path))

def copy(rc, dst,   follow_symlinks=True):
    if  rc is None or not os.path.exists(rc):
        return

    parent_path=(os.path.dirname(dst))
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    if os.path.isdir(rc):
        shutil.copytree(rc, dst ,dirs_exist_ok=True)
    else:
        try:
            shutil.copy(rc, dst,  follow_symlinks=follow_symlinks)
        except shutil.SameFileError:
            pass
def copy_files(src_dir, dst_dir):
    # 遍历源目录中的所有文件
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(dst_dir, filename)

        # 确保是文件而不是目录
        if os.path.isfile(src_file):
            # 复制文件
            shutil.copy2(src_file, dst_file)

def remove_file_by_re(src):


    src_file_list =  glob.glob(src)
    for file in src_file_list:
        Path(file).unlink()



def cat(files,out_file):


    # 获取所有以 'file' 开头的文件名
    if isinstance(files,str):
        file_list = glob.glob(files)
    else:
        file_list = files

    # 打开目标文件用于写入
    with open(out_file, 'wb') as outfile:
        for filename in file_list:

            with open(filename, 'rb') as infile:
                # 读取文件内容并写入到目标文件
                outfile.write(infile.read())


@contextmanager
def cd(path: Union[str, Path]) -> Generator:
    """


        with cd("/my/path/"):
            do_something()

    Args:
        path: Path to cd to.
    """
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)
def iter_path_to_atoms(glob_strs: list,show_progress=True,**kkwargs):
    def decorator(func):
        def wrapper(path: Path | str, *args, **kwargs):
            if isinstance(path, str):
                path = Path(path)
            paths=[]
            if path.is_dir():
                parent = path
                for glob_str in glob_strs:
                    for i in parent.glob(glob_str):
                        paths.append(i)
            else:
                paths = [path]
            result =[]

            filter_path_list=[]

            for i in paths:
                try:
                    atoms=ase_read(i.as_posix(),index=":")
                except Exception as e:
                    print_warning(f"文件：{i.as_posix()}读取错误!报错原因：{e}")
                    continue
                if isinstance(atoms,list):

                    filter_path_list.extend(atoms)
                else:
                    filter_path_list.append(atoms)
            if len(filter_path_list)==0:
                print_warning(f"Structure file not found: {path}")
                return result
            if show_progress:
                iter_obj=track(filter_path_list,
                              **kkwargs
                              )
            else:
                iter_obj=filter_path_list

            for i in iter_obj:

                try:
                    result.append(func(i, *args, **kwargs))
                except KeyboardInterrupt:
                    return result
                except Exception as e:

                    print_warning(traceback.format_exc())
                    pass
            return result
        return wrapper

    return decorator




def get_command_result(cmd):
    try:
        # 使用 subprocess 调用 which 命令，并捕获输出

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # 检查命令是否成功执行

        if result.returncode == 0:
            # 返回命令的路径
            return result.stdout.strip()
        else:
            # 如果命令未找到，返回 None 或抛出异常
            return None
    except Exception as e:

        return None

def is_file_empty(file_path):
    # 检查文件是否存在

    if file_path is None:
        return True

    if not os.path.exists(file_path):
        print_warning(f"The file {file_path} does not exist.")
        return True
        # raise FileNotFoundError(f"文件 {file_path} 不存在。")

    # 检查文件大小
    return os.path.getsize(file_path) == 0
def is_diff_path(path,path1):
    return os.path.abspath(path)!=os.path.abspath(path1)

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


