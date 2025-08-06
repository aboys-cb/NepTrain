#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 15:42
# @Author  : 兵
# @email    : 1747193328@qq.com

import os
import re
import shutil
import subprocess

from rich.progress import Progress

from watchdog.events import FileSystemEventHandler


from NepTrain import utils, Config, observer
from .utils import read_symbols_from_file




class NepFileMoniter(FileSystemEventHandler):
    def __init__(self,file_path,total):

        self.file_path = file_path
        self.progress = Progress( )
        self.current_steps=0
        self.total=int(total)
        self.progress.start()
        self.pbar=self.progress.add_task(total=int(total),description="NEP training")
    def on_modified(self, event):

        if not utils.is_diff_path(event.src_path , self.file_path):
            with open(self.file_path,'r',encoding="utf8") as f:
                lines = f.readlines()
                if not lines:
                    return
                last_line=lines[-1]
                current_steps=int(last_line.split(" ")[0])

                self.progress.advance(self.pbar,current_steps-self.current_steps)
                self.current_steps=current_steps

    def finish(self):

        if self.progress.finished:
            self.progress.advance(self.pbar,self.total-self.current_steps)


        self.progress.stop()



class RunInput:

    def __init__(self,train_xyz_path,nep_in_path=None,test_xyz_path=None):
        self.nep_in_path = nep_in_path
        self.train_xyz_path = train_xyz_path
        self.test_xyz_path = test_xyz_path
        self.run_in={"generation":100000}


        self.restart=False
        if self.nep_in_path is not None and os.path.exists(self.nep_in_path):
            self.read_run(self.nep_in_path)
        self.command=Config.get('environ','nep_path')

    def read_run(self,file_name):
        with open(file_name,'r',encoding="utf8") as f:
            # groups=re.findall("(\w+)\s+(.*?)\n",f.read()+"\n")
            groups=re.findall("^([A-Za-z_]+)\s+(.*)",f.read() ,re.MULTILINE)

            for group in groups:
                self.run_in[group[0].strip()]=group[1].strip()

    def set_restart(self,file_path,steps):
        if file_path and os.path.exists(file_path):
            self.restart_nep_path=file_path
            self.run_in["generation"]=steps
            self.run_in["lambda_1"]=0
            self.restart=True


    def build_run(self):
        """
        如果runin 不存在 就遍历训练集  然后找出所有的元素

        :return:
        """
        symbols = read_symbols_from_file(self.train_xyz_path)
        self.run_in["type"]=f"{len(symbols)} {' '.join(symbols)}"

    def write_run(self,file_name):
        if  "type" not in   self.run_in :
            self.build_run()
        with open(file_name,'w',encoding="utf8") as f:
            for k,v in self.run_in.items():

                f.write(f"{k}     {v}\n" )


    def calculate(self,directory,show_progress=True):
        utils.verify_path(directory)
        if self.restart:
            # utils.print_tip("Start the restart mode!")
            if utils.is_diff_path(self.restart_nep_path,os.path.join(directory,"nep.restart")):

                utils.copy(self.restart_nep_path,os.path.join(directory,"nep.restart"))


        self.write_run(os.path.join(directory,"nep.in"))
        if self.train_xyz_path is   None or not  os.path.exists(self.train_xyz_path):
            raise ValueError("A valid train.xyz must be specified.")
        if utils.is_diff_path(self.train_xyz_path ,os.path.join(directory,"train.xyz")):

            shutil.copy(self.train_xyz_path,os.path.join(directory,"train.xyz"))
        if self.test_xyz_path is not None and os.path.exists(self.test_xyz_path):
            if utils.is_diff_path(self.test_xyz_path, os.path.join(directory, "test.xyz")):

                shutil.copy(self.test_xyz_path, os.path.join(directory, "test.xyz"))
        if show_progress:

            handler=NepFileMoniter(os.path.join(directory,"loss.out"),self.run_in["generation"])
            watch=observer.schedule(handler, os.path.abspath(directory) , recursive=False)


            if not observer.is_alive():

                observer.start()

        with   open(os.path.join(directory,"nep.out"), "w") as f_std, open(os.path.join(directory,"nep.err"), "w", buffering=1) as f_err:

            errorcode = subprocess.call(self.command,
                                        shell=True,
                                        stdout=f_std,
                                        stderr=f_err,
                                        cwd=directory)


        if show_progress:

            handler.finish()
            observer.unschedule(watch)
            observer.stop()

class PredictionRunInput(RunInput):
    def __init__(self,nep_txt_path,*args,**kwargs):
        self.nep_txt_path=nep_txt_path
        super().__init__(*args,**kwargs)

    def write_run(self,file_name):

        self.run_in["prediction"]=1
        super().write_run(file_name)

    def calculate(self,directory,show_progress=False ):
        utils.verify_path(directory)

        if self.nep_txt_path is not None and os.path.exists(self.nep_txt_path):
            if utils.is_diff_path(self.nep_txt_path, os.path.join(directory, "nep.txt")):

                shutil.copy(self.nep_txt_path, os.path.join(directory, "nep.txt"))
        else:
            raise ValueError("In prediction mode, a potential function must be specified, please specify it via the `--nep nep_path` option.")
        super().calculate(directory,show_progress)
if __name__ == '__main__':
    run=RunInput("./train1.xyz")
    # run.read_run("./nep.in")
    run.write_run("./nep.out")
    run.calculate("./")