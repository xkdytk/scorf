import os
import sys
import numpy as np
import imageio
import random
import time
import torch
import cv2
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
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
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
    parser.add_argument("--chunk", type=int, default=1024*32, 
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
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    # training options
    parser.add_argument("--N_epoch", type=int, default=70,
                        help='number of train iteration')
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    parser.add_argument("--z_loss_weight", type=float,
                        default=1e-2, help='z_loss_weight') 
    parser.add_argument("--z_loss_iter", type=int,
                        default=500, help='z_loss_iter') 

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
    parser.add_argument("--n_ckpt",   type=int, default=1, 
                        help='number of checkpoints to save')
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=6000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=1000000000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=500000000, 
                        help='frequency of render_poses video saving')

    return parser


def train():

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

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR :', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

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
    basedir = args.basedir
    expname = args.expname
    
    os.makedirs(os.path.join(basedir, expname, 'config'), exist_ok=True)
    f = os.path.join(basedir, expname, 'config', 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config', 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
            
    # Load checkpoints
    os.makedirs(os.path.join(basedir, expname, 'ckpt'), exist_ok=True)
    
    if args.ckpt is not None and args.ckpt!='None':
        ckpts = [args.ckpt]
        print_ckpts = str(int(args.ckpt.split('/')[-1].split('.')[0])) + '_iter'
    else:
        ckpts = [os.path.join(basedir, expname, 'ckpt', f) for f in sorted(os.listdir(os.path.join(basedir, expname, 'ckpt'))) if 'tar' in f]
        print_ckpts = [str(int(f.split('.')[0])) + ' iter' for f in sorted(os.listdir(os.path.join(basedir, expname, 'ckpt'))) if 'tar' in f]

    print('Found ckpts', print_ckpts)

    # Create nerf model
    render_kwargs_train,render_kwargs_test, optimizer, start, epoch, best_psnr = build_model(args, device, ckpts, mode='train')
                                              
    print('==================================================\n')
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching

    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done\n')
    
        i_batch = 0
        rays_num = rays_rgb.shape[0]
    else:
        rays_tmp = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0)
        rays_tmp = np.stack([rays_tmp[i] for i in i_train], 0)
        rays_tmp = np.reshape(rays_tmp, [-1,2,3])
        rays_num = rays_tmp.shape[0]

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)
        
    print('==================================================\n')
    
    N_iters = rays_num//N_rand * args.N_epoch
    print('Train Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val, '\n')
    
    start = start + 1
    for i in trange(start, N_iters):
        # Sample random ray batch
        
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]
        
            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                tqdm.write("Shuffle data after an epoch!")
                tqdm.write(f"[TRAIN] epoch: {epoch} Loss: {loss.item()}  PSNR: {psnr.item()}")
                epoch += 1
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, _, z_diff, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                retraw=True, **render_kwargs_train)
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)
        
        # z_loss = torch.mean(torch.relu(-z_diff))
        
        # z_loss = torch.mean(torch.max(z_diff, -1)[0])
        z_loss = asc_loss(-z_diff)
        
        # z_loss_weight = args.z_loss_weight if i < args.z_loss_iter else 1.
        
        # loss = loss + 1e-2 * z_loss
        loss = loss + args.z_loss_weight * z_loss
        
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        # loss.backward(retain_graph=True)
        # z_loss.backward()
        loss.backward()
            
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1  # 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        if i % args.i_print==0 and i>0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1

        if i % args.i_weights==0 and i>0:
            
            os.makedirs(os.path.join(basedir, expname, 'ckpt'), exist_ok=True)

            path = os.path.join(basedir, expname, 'ckpt', '{:08d}.tar'.format(i))
            if args.N_fine_samples > 0:
                torch.save({
                    'epoch' : epoch,
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_psnr' : best_psnr,
                }, path)
            else:
                torch.save({
                    'epoch' : epoch,
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_psnr' : best_psnr,
                }, path)

            ckpt_list = os.listdir(os.path.join(basedir, expname, 'ckpt'))

            if len(ckpt_list) > args.n_ckpt:
                sort_ckpt_list = sorted(ckpt_list)
                print(sort_ckpt_list)

                for j in range (len(ckpt_list)-args.n_ckpt):
                    os.remove(os.path.join(basedir, expname, 'ckpt', '{}'.format(sort_ckpt_list[j])))

            print('\n')
            print('Saved checkpoints at', path, '\n')

            os.makedirs(os.path.join(basedir, expname, 'testsavedir'), exist_ok=True)
            testsavedir = os.path.join(basedir, expname, 'testsavedir', 'testset_{:08d}'.format(i))
            logdir = os.path.join(basedir, expname)
            os.makedirs(testsavedir, exist_ok=True)

            with torch.no_grad():
                test_chunk = args.chunk // 8
                if use_batching:
                    _, _, best_psnr, best_flag = render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, test_chunk, render_kwargs_test, best_psnr, gt_imgs=images[i_test], savedir=testsavedir, logdir=logdir)
                else:
                    _, _, best_psnr, best_flag = render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, test_chunk, render_kwargs_test, best_psnr, gt_imgs=torch.Tensor(images[i_test]).to(device), savedir=testsavedir, logdir=logdir)
            tqdm.write('Saved test set')

            os.makedirs(os.path.join(basedir, expname, 'best_ckpt'), exist_ok=True)

            if best_flag and best_psnr > 2.0:
                path = os.path.join(basedir, expname, 'best_ckpt', '{:08d}_{}.tar'.format(i, best_psnr.item()))
                torch.save({
                    'epoch' : epoch,
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_psnr' : best_psnr,
                }, path)

                print('Saved best checkpoints at', path)

                best_list = os.listdir(os.path.join(basedir, expname, 'best_ckpt'))

                if len(best_list) > 1:
                    sort_best_list = sorted(best_list)
                    print(sort_best_list)

                    for j in range (len(best_list)-1):
                        os.remove(os.path.join(basedir, expname, 'best_ckpt', '{}'.format(sort_best_list[j])))


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()

