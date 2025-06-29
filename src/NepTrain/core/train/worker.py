import asyncio
import os

from dpdispatcher import Machine, Resources, Task, Submission
from pathlib import Path
from NepTrain import utils


def remove_sub_file(work_path: str = "./"):
    """Remove temporary submission files."""
    all_file = Path(work_path).glob("????????????????????????????????????????.sub.*")
    for file in all_file:
        file.unlink()
    all_file = Path(work_path).glob("????????????????????????????????????????.sub")
    for file in all_file:
        file.unlink()

def submit_job(
    machine_dict: dict,
    resources_dict: dict,
    task_dict_list: list,
    submission_dict: dict,
) -> Submission:
    """Submit a job synchronously using dpdispatcher."""
    machine = Machine.load_from_dict(machine_dict)
    resources = Resources.load_from_dict(resources_dict)
    task_list = [Task(**task_dict) for task_dict in task_dict_list]
    submission = Submission(
        machine=machine,
        resources=resources,
        task_list=task_list,
        **submission_dict,
    )
    submission.run_submission(clean=False)
    job_id = getattr(submission, 'job_id', getattr(submission, 'submission_id', 'UNKNOWN'))
    work_path = getattr(submission, 'work_base', submission_dict.get('work_base', './'))
    utils.print_msg(f"Submitted job {job_id} in {work_path}")
    remove_sub_file()
    return submission


async def async_submit_job(
    machine_dict: dict,
    resources_dict: dict,
    task_dict_list: list,
    submission_dict: dict,
) -> None:
    """Submit a job asynchronously using dpdispatcher."""
    machine = Machine.load_from_dict(machine_dict)
    resources = Resources.load_from_dict(resources_dict)
    task_list = [Task(**task_dict) for task_dict in task_dict_list]
    submission = Submission(
        machine=machine,
        resources=resources,
        task_list=task_list,
        **submission_dict,
    )
    await submission.async_run_submission(check_interval=2, clean=False)
    job_id = getattr(submission, 'job_id', getattr(submission, 'submission_id', 'UNKNOWN'))
    work_path = getattr(submission, 'work_base', submission_dict.get('work_base', './'))
    utils.print_msg(f"Submitted job {job_id} in {work_path}")
    remove_sub_file()
