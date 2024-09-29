import os, sys
import numpy as np
import time
import imageio
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.auto import tqdm

from .rays import get_rays, ndc_rays
from .util import sample_pdf
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt
import lpips
# Misc
img2ssim = lambda x,y : ssim(x, y, data_range=y.max() - y.min(), channel_axis=-1)
img2mse = lambda x, y : torch.mean((x - y) ** 2)
# asc_loss = lambda x : torch.mean(torch.sum(x.clamp(min=0), -1))
asc_loss = lambda x : torch.mean(torch.sum(F.relu(x), -1))
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    
    # dists = dists.clamp(min=0)
    dists = F.relu(dists)
    
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    # dists = torch.cat([z_vals[..., 0].unsqueeze(-1), dists], -1) # 수정했습니다.
    # dists[...,-1] = torch.Tensor([1e10]).expand(dists[...,-1].shape)

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    #x = np.linspace(1, alpha.shape[1], alpha.shape[1])
    #plt.scatter(x, alpha[0].detach().cpu().numpy())
    #plt.savefig('alpha_fig.png')
    #plt.clf()
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)

    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                model_type,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_fine_samples=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                ):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_fine_samples: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)

    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand
    
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    raw, t_vals = network_query_fn(pts, viewdirs, network_fn)
    
    # t_vals = t_vals.clamp(0, 1)
    # z_diff = t_vals[...,1:] - t_vals[...,:-1]
    
    # max_previous = z_vals.cummax(dim=1).values
    z_diff = t_vals[...,1:] - t_vals[...,:-1]

    # z_diff = max_previous - z_vals
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    #print('\nraw_shape : ', raw.shape, '\n')
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    if N_fine_samples > 0:
        
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_fine_samples, det=(perturb==0.))
        z_samples = z_samples.detach()
        
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_fine_samples, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw , z_vals = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map, 'z_diff' : z_diff}
    if retraw:
        ret['raw'] = raw
    if N_fine_samples > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    return ret

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., use_viewdirs=False, c2w_staticcam=None, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays
    
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
        
    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map', 'z_diff']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, best_psnr, gt_imgs=None, savedir=None, logdir=None):

    H, W, focal = hwf

    loss_fn_alex = lpips.LPIPS(net='vgg')

    best_flag = False
    psnr_list = []
    ssim_list = []
    rgbs = []
    disps = []
    depths = []
    lpips_arr = []

    t = time.time()
    if logdir is not None:
        tqdm.write(f"\n")
        f = open(os.path.join(logdir,'stdout.txt'), 'a')
        print(savedir, file=f)
    for i, c2w in enumerate(tqdm(render_poses)):
        t = time.time()
        rgb, disp, acc, depth, _, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        depths.append(depth.cpu().numpy())

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            depth8 = depths[-1]
            filename = os.path.join(savedir, '{:03d}_{typ}.png'.format(i, typ='depth'))
            imageio.imwrite(filename, depth8)
            disp8 = disps[-1]
            filename = os.path.join(savedir, '{:03d}_{typ}.png'.format(i, typ='disp'))
            imageio.imwrite(filename, disp8)
            
        if gt_imgs is not None:
            loss = img2mse(rgb, gt_imgs[i])
            psnr = mse2psnr(loss)
            psnr_list.append(psnr)
            lpip = loss_fn_alex(torch.permute(torch.Tensor(gt_imgs[i]), (2, 0, 1)), torch.permute(rgb, (2, 0, 1)))
            lpips_arr.append(lpip.item())
            rgb = rgb.cpu().numpy()
            gt = gt_imgs[i].cpu().numpy()
            ssim = img2ssim(gt, rgb)
            ssim_list.append(ssim)
            print(f"PSNR : {psnr.item()},  SSIM : {ssim}, LPIPS : {lpip.item()}", file=f)
            tqdm.write(f"PSNR : {psnr.item()},  SSIM : {ssim}, LPIPS : {lpip.item()}")

    if gt_imgs is not None:
        psnr_average = sum(psnr_list) / len(psnr_list)
        ssim_average = sum(ssim_list) / len(ssim_list)
        lpips_average = sum(lpips_arr)/len(lpips_arr)

        print(f"psnr average : {psnr_average.item()}", file=f)
        print(f"ssim average : {ssim_average}", file=f)
        print(f"lpips average : {lpips_average}", file=f)
        tqdm.write(f"PSNR Average : {psnr_average.item()},  SSIM Average : {ssim_average}, LPIPS Average : {lpips_average}")
    
        if best_psnr < psnr_average:
            best_psnr = psnr_average
            best_flag = True

    if logdir is not None:
        print("\n", file=f)
        f.close()

    rgbs = np.stack(rgbs, 0)
    depths = np.stack(depths, 0)

    return rgbs, depths, best_psnr, best_flag
