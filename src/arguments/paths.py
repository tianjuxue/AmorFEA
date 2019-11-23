"""Args defining output and data saving path"""

import os

def add_args(parser):
    rt_path = 'data'
    sub_paths = ['images', 'solutions', 'numpy', 'torch', 'models', 'others']

    parser.add_argument('--root_path', type=str, default=rt_path)
    parser.add_argument('--images_path', type=str, default=sub_paths[0])
    parser.add_argument('--solutions_path', type=str, default=sub_paths[1])
    parser.add_argument('--numpy_path', type=str, default=sub_paths[2])
    parser.add_argument('--torch_path', type=str, default=sub_paths[3])
    parser.add_argument('--model_path', type=str, default=sub_paths[4])    
    parser.add_argument('--others_path', type=str, default=sub_paths[5])

    if not os.path.exists(rt_path):
        os.mkdir(rt_path)

    for sub_path in sub_paths:
        if not os.path.exists(rt_path + '/' + sub_path):
            os.mkdir(rt_path + '/' + sub_path)