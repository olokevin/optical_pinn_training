import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
# sys.path.append('../TFONet')
from core.TFONet import start_from_ini
from dataio import dataio_init_from_config
from val_func import val_func_init_from_config
from loss_func import loss_func_init_from_config
from eval_func import eval_func_init_from_config


argparser = argparse.ArgumentParser(description='Process some integers.')
argparser.add_argument('-m', '--mode', type=str, default='both', help='Running mode: train/eval/both')
argparser.add_argument('-c', '--config_file_path', type=str, default='./pinns.ini',
                       help='Path to config file (.ini)')
argparser.add_argument('-y', '--yml_config_path', metavar='FILE', 
                       help='Path to config file (.yml)')
argparser.add_argument('-o', '--overwrite', type=bool, default=False,
                       help='Whether overwrite the existing experiment folder.')
argparser.add_argument('-e', '--eval_epoch_list', action='store', type=int, nargs='*', default=[0],
                       help="The list of epochs that are desired to evaluate at.")
args = argparser.parse_args()

start_from_ini(sys_args=args, dataio_from_config=dataio_init_from_config,
               loss_fn_from_config=loss_func_init_from_config, val_fn_from_config=val_func_init_from_config,
               eval_fn_from_config=eval_func_init_from_config)
