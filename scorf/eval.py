import os
import numpy as np
import imageio
import random
import time
import torch
from tqdm.auto import tqdm, trange

import matplotlib.pyplot as plt

from dataset import load_llff_data, load_blender_data
from models import build_model
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--basedir", type=str, default='./result/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern/test', 
                        help='input data directory')
    parser.add_argument("--model", type=str, default='nerf', 
                        help='model type')

    # training options
    parser.add_argument("--netdepth", type=int, default=2, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32 ,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ckpt", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_fine_samples", type=int, default=64,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    
        ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # logging/saving options
    parser.add_argument("--i_testset", action='store_true', 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   action='store_true', 
                        help='frequency of render_poses video saving')
    parser.add_argument("--i_typing",   action='store_true', 
                        help='frequency of typing pose saving')
    parser.add_argument("--x",   type=float, default=0.,
                        help='x-axis(theta, 각도), 값이 작아질 수록 왼쪽으로')
    parser.add_argument("--y",   type=float, default=0.,
                        help='y-axis(phi, 각도), 값이 작아질 수록 아래로')
    parser.add_argument("--z",   type=float, default=0.,
                        help='z-axis(radius, 이동값), 값이 작아질 수록 가까워짐, -3이 넘어가면 뒤집히고 멀어짐')

    return parser


def evaluation():

    parser = config_parser()
    args = parser.parse_args()
    print('==================================================\n')

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses = load_llff_data(args.datadir, args.factor, recenter=True,
                                                          bd_factor=.75, spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold, '\n')
            i_test = np.arange(images.shape[0])[::args.llffhold]

        print('DEFINING BOUNDS')
            
        near = 0.
        far = 1.
        print('NEAR FAR :', near, far)
        
    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        _, _, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
    
    print('==================================\n')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    # Create log dir and copy the config file
    basedir = './eval1'
    expname = args.expname
    
    ckpts = args.ckpt
    ckpts_name = str(int(ckpts.split('/')[-1].split('.')[0])) + '_iter'
    print('Found ckpts :', ckpts_name)

    # Create nerf model
    _, render_kwargs_test, _, _, _, best_psnr = build_model(args, device, ckpts, mode='eval')

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    if args.i_testset:
        poses = torch.Tensor(poses).to(device)
        images = torch.Tensor(images).to(device)
        
    print('==================================================\n')

    print('Test Begin')
    
    if args.i_video:
        # Turn on testing mode
        test_chunk = args.chunk // 8
        with torch.no_grad():
            rgbs, depths, _, _ = render_path(render_poses, hwf, K, test_chunk, render_kwargs_test, None)
        tqdm.write('Done, saving')
        tqdm.write(f"RGB_map shape : {rgbs.shape}") 
        tqdm.write(f"Depth_map shape : {depths.shape}\n")

        os.makedirs(os.path.join(basedir, expname, 'movie'), exist_ok=True)
        moviebase = os.path.join(basedir, expname, 'movie', '{}_spiral_{}_'.format(expname, ckpts_name))
        imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=10)
        imageio.mimwrite(moviebase + 'depth.mp4', to8b(depths), fps=30, quality=10)
        #imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

#             # use viewdirs to make movie
#             render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
#             with torch.no_grad():
#                 rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
#             render_kwargs_test['c2w_staticcam'] = None
#             imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

    if args.i_testset:
        os.makedirs(os.path.join(basedir, expname, 'testsavedir'), exist_ok=True)
        testsavedir = os.path.join(basedir, expname, 'testsavedir', 'testset_{}'.format(ckpts_name))
        os.makedirs(testsavedir, exist_ok=True)
        
        print('TEST views are', i_test)

        with torch.no_grad():
            render_path(torch.Tensor(poses[i_test]).to(device), hwf, args.chunk, render_kwargs_test, best_psnr, gt_imgs=images[i_test], savedir=testsavedir)
        tqdm.write('Saved test set')
    
    
    if args.i_typing:
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])

        def pose_spherical(theta, phi, radius):
            c2w = trans_t(radius)
            c2w = rot_phi(phi/180.*np.pi) @ c2w
            c2w = rot_theta(theta/180.*np.pi) @ c2w
            c2w = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) @ c2w
            return c2w


        c2w = torch.Tensor(pose_spherical(args.x, args.y, args.z))
        with torch.no_grad():
            rgb8, _, _, _ = render(H, W, focal, chunk=args.chunk, c2w=c2w[:3,:4], **render_kwargs_test)
        rgb8 = rgb8.cpu().numpy()
        rgb8 = to8b(rgb8)

#         plt.figure(2, figsize=(20,6))
#         plt.imshow(rgb8)
#         plt.show()
        
        # --theta 180.00 --phi -90.00 --radius -0.00 (default)
        
        os.makedirs(os.path.join(basedir, expname, 'type_savedir'), exist_ok=True)
        filename = os.path.join(basedir, expname, 'type_savedir', 'test_{}_{}.png'.format(ckpts_name, 'x:'+str(args.x)+'_y:'+str(args.y)+'_z:'+str(args.z)))
        imageio.imwrite(filename, rgb8)



if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    evaluation()
