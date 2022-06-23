import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--pd_dir',  type=str, help='the PD dataset path' )
parser.add_argument('--additional_dir',  type=str, help='moving and all object masks dir')

cameras = ['camera_01', 'camera_05', 'camera_06', 'camera_07', 'camera_08', 'camera_09']

def merge(pd_path, add_path):
    scenes = os.listdir(add_path)
    for sc in scenes:
        cmd = 'mv {}/{}/moving_masks {}/{}/moving_masks/'.format(pd_path, sc, add_path, sc)
        os.system(cmd)
        cmd = 'mv {}/{}/ari_masks {}/{}/ari_masks/'.format(pd_path, sc, add_path, sc)
        os.system(cmd)

def main():
    opt = parser.parse_args()
    merge(opt.pd_dir, opt.additional_dir)

    

if __name__ == '__main__':
    main()
