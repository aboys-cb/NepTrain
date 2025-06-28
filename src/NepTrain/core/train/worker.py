import asyncio

from dpdispatcher import Machine, Resources, Task, Submission
from pathlib import Path
def remove_sub_file(work_path="./"):
    all_file = Path(work_path).glob("????????????????????????????????????????.sub.*")

    for file in all_file:

        file.unlink()
def submit_job( machine_dict,
                resources_dict,
                task_dict_list,
                submission_dict
               ):
    machine = Machine.load_from_dict(machine_dict)
    resources = Resources.load_from_dict(resources_dict)
    task_list=[]
    for task_dict in task_dict_list:
        task1 = Task(**task_dict)
        task_list.append(task1)
    submission = Submission(
        machine=machine,
        resources=resources,
        task_list=task_list,**submission_dict
    )
    submission.run_submission(clean=True)
    remove_sub_file()
    return submission

async def async_submit_job( machine_dict,
                resources_dict,
                task_dict_list,
                submission_dict
               ):
    machine = Machine.load_from_dict(machine_dict)
    resources = Resources.load_from_dict(resources_dict)
    task_list=[]
    for task_dict in task_dict_list:
        task1 = Task(**task_dict)
        task_list.append(task1)
    submission = Submission(
        machine=machine,
        resources=resources,
        task_list=task_list,**submission_dict
    )
    submission.async_run_submission(clean=True)
    background_task = asyncio.create_task(

    submission.async_run_submission(check_interval=2, clean=False)
 )
    return background_task