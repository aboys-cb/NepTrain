#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 14:33
# @Author  : 兵
# @email    : 1747193328@qq.com
import argparse
import sys
sys.path.append('../../')
from NepTrain.core import *
from NepTrain import __version__
import warnings
from dpdispatcher.dlog import dlog_stdout, dlog
dlog.removeHandler(dlog_stdout)
# 禁用所有 UserWarning
warnings.simplefilter('ignore', UserWarning)
def check_kpoints_number(value):
    """检查值是否为单个数字或三个数字的字符串"""

    if isinstance(value, str):
        values = value.split(',')

        if len(values) == 3 and all(v.isdigit() for v in values):
            return list(map(int, values))
        elif len(values) == 1 and value.isdigit():
            return [int(value),int(value),int(value)]
        else:
            raise argparse.ArgumentTypeError("The ka parameter must be a single number or three numbers separated by `,`.")
    elif isinstance(value, int):
        return value
    else:
        raise argparse.ArgumentTypeError("The ka parameter must be a single number or three numbers separated by `,`.")

def build_init(subparsers):
    parser_init = subparsers.add_parser(
        "init",
        help="Initialize some file templates",
    )
    parser_init.add_argument("type",
                             type=str,
                            choices=["bohrium","slurm","pbs","local"],default="slurm",
                             help=" ")

    parser_init.add_argument("-f", "--force", action='store_true',
                             default=False,
                             help="Force overwriting of generated templates"
                             )

    parser_init.set_defaults(func=init_template)



def build_perturb(subparsers):
    parser_perturb = subparsers.add_parser(
        "perturb",
        help="Generate perturbed structures.",
    )

    parser_perturb.set_defaults(func=run_perturb)

    parser_perturb.add_argument("model_path",
                             type=str,

                             help="The structure path or structure file required for calculation only supports files in xyz and vasp formats.")
    parser_perturb.add_argument("--num","-n",
                             type=int,
                                default=20,
                             help="The number of perturbations for each structure, if a folder is input, the final number generated should be the number of structures multiplied by num.default 20.")

    parser_perturb.add_argument("--cell", "-c",
                                dest="cell_pert_fraction",
                                type=float,
                                default=0.03,
                                help="The deformation ratio,default 0.03.")

    parser_perturb.add_argument("--distance", "-d",
                                type=float,
                                dest="min_distance",
                                default=0.1,
                                help="Min atom distance, unit Å, default 0.1.")

    parser_perturb.add_argument("--out", "-o",
                             dest="out_file_path",
                             type=str,
                             help="Output file for perturbed structures, default ./perturb.xyz.",
                             default="./perturb.xyz"
                             )
    parser_perturb.add_argument("--append", "-a",
                             dest="append", action='store_true', default=False,
                             help="Write to out_file_path in append mode, default False.",

                             )

def build_vasp(subparsers):
    parser_vasp = subparsers.add_parser(
        "vasp",
        help="Calculate single-point energy using VASP.",
    )
    parser_vasp.set_defaults(func=run_vasp)

    parser_vasp.add_argument("model_path",
                             type=str,

                             help="The required structure path or structure file only supports files in xyz and vasp formats.")
    parser_vasp.add_argument("--directory", "-dir",

                             type=str,
                             help="Set the VASP calculation path. default ./cache/vasp.",
                             default="./cache/vasp"
                             )

    parser_vasp.add_argument("--out", "-o",
                             dest="out_file_path",
                             type=str,
                             help="Structure output file after calculation. default ./vasp_scf.xyz",
                             default="./vasp_scf.xyz"
                             )

    parser_vasp.add_argument("--append", "-a",
                             dest="append", action='store_true', default=False,
                             help="Write to out_file_path in append mode, default False.",

                             )
    parser_vasp.add_argument("--gamma", "-g",
                             dest="use_gamma", action='store_true', default=False,
                             help="Default to using Monkhorst-Pack k-points, add -g to use Gamma-centered k-point scheme.",

                             )
    parser_vasp.add_argument("-n", "-np",
                             dest="n_cpu",
                             default=1,
                             type=int,
                             help="Set the number of CPU cores, default 1.")

    parser_vasp.add_argument("--incar",

                             help="Input path for INCAR file, default is ./INCAR.",default="./INCAR")



    k_group = parser_vasp.add_mutually_exclusive_group(required=False)
    k_group.add_argument("--kspacing", "-kspacing",

                         type=float,
                         help="Set kspacing, which can also be defined in the INCAR _template.")
    k_group.add_argument("--ka", "-ka",
                         default=[1, 1, 1],
                         type=check_kpoints_number,
                         help="ka takes 1 or 3 numbers (comma-separated), sets k-points to (k[0]/a, k[1]/b, k[2]/c). default 1.")


def build_nep(subparsers):
    parser_nep = subparsers.add_parser(
        "nep",
        help="Train potential functions using NEP.",
    )
    parser_nep.set_defaults(func=run_nep)


    parser_nep.add_argument("--directory", "-dir",

                             type=str,
                             help="Set the path for NEP calculations. default ./cache/nep",
                             default="./cache/nep"
                             )
    parser_nep.add_argument("--in", "-in",
                            dest="nep_in_path",
                             type=str,
                             help="Set the path for the nep.in file; if not present, generate it based on train.xyz. default ./nep.in",
                             default="./nep.in"
                             )

    parser_nep.add_argument("--train", "-train",
                             dest="train_path",

                             type=str,
                             help="Set the path for the train.xyz file, default  ./train.xyz.",
                             default="./train.xyz"
                             )
    parser_nep.add_argument("--test", "-test",
                             dest="test_path",
                             type=str,
                             help="Set the path for the test.xyz file, default is ./test.xyz.",
                             default="./test.xyz"
                             )
    parser_nep.add_argument("--nep", "-nep",
                            dest="nep_txt_path",
                             type=str,
                             help="restart and prediction require the use of a potential function, default is ./nep.txt.",
                             default="./nep.txt"
                             )
    parser_nep.add_argument("--prediction", "-pred","--pred",

                             action="store_true",
                             help="Set the forecast mode，default False",
                             default=False
                             )
    parser_nep.add_argument("--restart_file", "-restart","--restart",

                            type=str,

                            help="To restart running, simply provide a valid path; default is None.",
                             default=None
                             )
    parser_nep.add_argument("--continue_step", "-cs",

                            type=int,

                            help="If a restart_file is provided, this parameter will take effect, continuing for continue_step steps, with a default value of 10000.",
                             default=10000
                             )
def build_gpumd(subparsers):
    parser_gpumd = subparsers.add_parser(
        "gpumd",
        help="run molecular dynamics using GPUMD.",
    )
    parser_gpumd.set_defaults(func=run_gpumd)

    parser_gpumd.add_argument("model_path",
                             type=str,

                             help="The required structure path or structure file only supports files in xyz and vasp formats.")
    parser_gpumd.add_argument("--directory", "-dir",

                             type=str,
                             help="Set the GPUMD calculation path, default is ./cache/gpumd.",
                             default="./cache/gpumd"
                             )
    parser_gpumd.add_argument("--in","-in",dest="run_in_path", type=str,
                              help="The filename for the command _template file, default is ./run.in.", default="./run.in")

    parser_gpumd.add_argument("--nep", "-nep",
                            dest="nep_txt_path",
                             type=str,
                             help="Potential function path, default is ./nep.txt.",
                             default="./nep.txt"
                             )
    parser_gpumd.add_argument("--time", "-t", type=int, help="Molecular dynamics time, unit ps, default 10 ps.", default=10)
    parser_gpumd.add_argument("--temperature", "-T", type=int, help="Molecular dynamics temperature in Kelvin,multiple integers can be input. default is 300 K", nargs="*", default=[300])

    parser_gpumd.add_argument("--out", "-o",
                               dest="out_file_path",

                               type=str,
                               default="./trajectory.xyz",
                               help="Output path for structures."
                               )

def build_train(subparsers):
    parser_train = subparsers.add_parser(
        "train",
        help="Automatic training.",
    )
    parser_train.set_defaults(func=train_nep)

    parser_train.add_argument("config_path",
                             type=str,

                             help="The required structure path or structure file only supports files in XYZ and VASP formats.")


def build_select(subparsers):
    parser_select = subparsers.add_parser(
        "select",
        help="Select samples.",
    )
    parser_select.set_defaults(func=run_select,decomposition='pca')

    parser_select.add_argument("trajectory_paths",
                              nargs="+",
                             help="The trajectory files needed for sampling is in xyz format.")


    parser_select.add_argument("--base", "-base",
                               type=str,
                               default="train.xyz",
                               help="Provide a path to base.xyz, and sample the trajectory based on base.xyz, default is train.xyz."
                               )
    parser_select.add_argument("--nep", "-nep",
                               type=str,
                               default="./nep.txt",
                               help="Provide a path to a nep.txt file to extract descriptors for the structure, default is ./nep.txt. If the file does not exist, use SOAP descriptors."
                               )
    parser_select.add_argument("--max_selected", "-max", type=int,
                               help="Maximum number of structures to select, default is 20.",
                               default=20)
    parser_select.add_argument("--min_distance","-d", type=float,
                               help="Minimum bond length for farthest-point sampling, default is 0.01.",
                               default=0.01)
    parser_select.add_argument("--filter", "-f", type=float,
                               const=0.6,nargs='?',
                               help="Whether to filter based on covalent radius, the default is False. If True, the default coefficient is 0.6, and a coefficient can be passed in",
                               default=False)

    dc_group = parser_select.add_mutually_exclusive_group(required=False)
    dc_group.add_argument('-pca',"--pca", action='store_const', const='pca', dest='decomposition',
                       help='Use PCA for decomposition')
    dc_group.add_argument('-umap',"--umap", action='store_const', const='umap', dest='decomposition',
                       help='Use UMAP for decomposition')

    parser_select.add_argument("--out", "-o",
                               dest="out_file_path",

                               type=str,
                               default="./selected.xyz",
                               help="Output path for selected structures.default ./selected.xyz"
                               )

    group= parser_select.add_argument_group("SOAP","SOAP Parameters")

    group.add_argument("--r_cut", "-r", type=float, help="A cutoff for local region in angstroms,default 6", default=6)
    group.add_argument("--n_max", "-n", type=int, help="The number of radial basis functions,default 8", default=8)
    group.add_argument("--l_max", "-l", type=int, help="The maximum degree of spherical harmonics,default 6", default=6)

def main():
    parser = argparse.ArgumentParser(
        description="""
        NepTrain is a tool for automatically training NEP potential functions""",

    )
    parser.add_argument(
        "-v", "--version", action="version", version=__version__
    )



    subparsers = parser.add_subparsers()


    build_init(subparsers)

    build_perturb(subparsers)

    build_select(subparsers)

    build_vasp(subparsers)

    build_nep(subparsers)
    build_gpumd(subparsers)
    build_train(subparsers)



    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:

        pass


    args = parser.parse_args()

    try:
        _ = args.func
    except AttributeError as exc:
        parser.print_help()
        raise SystemExit("Please specify a command.") from exc
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
