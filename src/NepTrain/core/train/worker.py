#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/25 20:51
# @Author  : 兵
# @email    : 1747193328@qq.com
import re
import subprocess
import sys
import time
from threading import Thread

from NepTrain import utils


class Worker():
    pass
    def __init__(self, mode):
        self.mode = mode

        self._running=False

    @property
    def running(self):
        return self._running

    def sub_job(self,command,job_path):
        raise NotImplementedError

    def wait(self):
        pass

class LocalWorker(Worker):
    def __init__(self,  ):
        super().__init__("local")
    def sub_job(self,command,job_path,**kwargs):
        utils.verify_path(job_path)

        # with  open("job.out", "w") as f_std, open("job.err", "w", buffering=1) as f_err:
        errorcode = subprocess.call(command,
                                    shell=True,
                                    stdout=sys.stdout,
                                    stderr=sys.stderr,
                                    cwd=job_path)



class SlurmWorker(Worker):
    def __init__(self,vasp_sh,gpumd_sh  ):
        super().__init__("slurm")
        self.vasp_sh = vasp_sh
        self.gpumd_sh = gpumd_sh
        #创建一个线程  定时检查任务状态？
        self.job_id=[]
        self._thread=Thread(target=self.check_job_state)
    def sub_job(self,command,job_path,job_type="vasp"):
        utils.verify_path(job_path)

        if job_type == "vasp":
            job_command=["sbatch",self.vasp_sh,command]
        else:
            job_command=["sbatch",self.gpumd_sh,command]
        # print(command)

        result = subprocess.run(job_command, capture_output=True, text=True, check=True,cwd=job_path)
        job_id=int(result.stdout.replace("Submitted batch job ",""))
        self.job_id.append(job_id)
        utils.print_msg(f"Task submitted: {job_id}",)


    def wait(self):
        utils.print_msg("Waiting for all tasks to finish...")
        while  self.job_id:
            for job in self.job_id.copy():

                if not self.check_job_state(job):
                    self.job_id.remove(job)
            time.sleep(5)
        utils.print_success("All tasks have finished computation.")

    def check_job_state(self,job_id):

        result = subprocess.run(['squeue','-j',f"{job_id}"], capture_output=True, text=True, check=True)
        match = re.search(r'JOBID.*?(\d+) ', result.stdout, re.S)
        #
        # 如果找到匹配项，打印作业ID
        if match:
            # job_id = match.group(1)
            return True


        else:
            return False

