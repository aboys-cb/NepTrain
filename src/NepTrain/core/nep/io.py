#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 15:42
# @Author  : 兵
# @email    : 1747193328@qq.com

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

from rich.progress import Progress

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:  # pragma: no cover - optional progress dependency
    FileSystemEventHandler = object
    Observer = None

from .utils import read_symbols_from_file




class NepFileMoniter(FileSystemEventHandler):
    def __init__(self,file_path,total):

        self.file_path = Path(file_path).resolve()
        self.progress = Progress( )
        self.current_steps=0
        self.total=int(total)
        self.progress.start()
        self.pbar=self.progress.add_task(total=int(total),description="NEP training")
    def on_modified(self, event):

        if Path(event.src_path).resolve() == self.file_path:
            with self.file_path.open('r',encoding="utf8") as f:
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
        self.command = os.environ.get("NEPTRAIN_NEP_COMMAND", "nep")

    def read_run(self,file_name):
        with open(file_name,'r',encoding="utf8") as f:
            # groups=re.findall("(\w+)\s+(.*?)\n",f.read()+"\n")
            groups=re.findall(r"^([A-Za-z_]+)\s+(.*)",f.read() ,re.MULTILINE)

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
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if self.restart:
            restart_target = directory / "nep.restart"
            restart_source = Path(self.restart_nep_path).expanduser().resolve()
            if restart_source != restart_target:
                shutil.copy2(restart_source, restart_target)


        self.write_run(directory / "nep.in")
        if self.train_xyz_path is   None or not  os.path.exists(self.train_xyz_path):
            raise ValueError("A valid train.xyz must be specified.")
        train_source = Path(self.train_xyz_path).expanduser().resolve()
        train_target = directory / "train.xyz"
        if train_source != train_target:
            shutil.copy2(train_source, train_target)
        if self.test_xyz_path is not None and os.path.exists(self.test_xyz_path):
            test_source = Path(self.test_xyz_path).expanduser().resolve()
            test_target = directory / "test.xyz"
            if test_source != test_target:
                shutil.copy2(test_source, test_target)
        observer = Observer() if show_progress and Observer is not None else None
        if observer is not None:

            handler=NepFileMoniter(directory / "loss.out",self.run_in["generation"])
            watch=observer.schedule(handler, str(directory), recursive=False)


            if not observer.is_alive():

                observer.start()

        with (directory / "nep.out").open("w") as f_std, (directory / "nep.err").open("w", buffering=1) as f_err:

            completed = subprocess.run(
                shlex.split(self.command),
                stdout=f_std,
                stderr=f_err,
                cwd=directory,
                check=False,
            )

        if completed.returncode != 0:
            raise RuntimeError(
                "GPUMD NEP training failed with exit code "
                f"{completed.returncode}; "
                f"see {directory / 'nep.err'}"
            )


        if show_progress and observer is not None:

            handler.finish()
            observer.unschedule(watch)
            observer.stop()

if __name__ == '__main__':
    run=RunInput("./train1.xyz")
    # run.read_run("./nep.in")
    run.write_run("./nep.out")
    run.calculate("./")
