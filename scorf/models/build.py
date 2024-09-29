from .nerf import *
from .CNN import *

import torchsummary
from torchsummary import summary as summary

import os

def build_model(args, device, ckpts, mode):
    model_type = args.model
    device = device
    ckpts = ckpts
    mode = mode
    
    if model_type == 'nerf':
        """Instantiate NeRF's MLP model.
        """
        
        embed_fn, input_ch = get_embedder(args.multires)
        
        input_ch_views = 0
        embeddirs_fn = None
        if args.use_viewdirs:
            embeddirs_fn, input_ch_views = get_embedder(args.multires_views)

        
        skips = [4]
        model = NeRF(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, output_ch=4, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars = list(model.parameters())

        model_fine = None
        if args.N_fine_samples > 0:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                              input_ch=input_ch, output_ch=4, skips=skips,
                              input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
            grad_vars += list(model_fine.parameters())

        network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)

        # Create optimizer
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        start = 0
        epoch = 0
        best_psnr = 0
        basedir = args.basedir
        expname = args.expname

        ##########################

        # Load checkpoints and apply on model
        if len(ckpts) > 0 and not args.no_reload and mode=='train':
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path, '\n')
            ckpt = torch.load(ckpt_path)

            start = ckpt['global_step']
            epoch = ckpt['epoch']
            best_psnr = ckpt['best_psnr']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            # Load model
            model.load_state_dict(ckpt['network_fn_state_dict'])
            if model_fine is not None:
                model_fine.load_state_dict(ckpt['network_fine_state_dict'])
                
        if len(ckpts) > 0 and mode=='eval':
            print('Reloading from', ckpts, '\n')
            ckpt = torch.load(ckpts)

            start = ckpt['global_step']
            epoch = ckpt['epoch']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            # Load model
            model.load_state_dict(ckpt['network_fn_state_dict'])
            if model_fine is not None:
                model_fine.load_state_dict(ckpt['network_fine_state_dict'])
                
        ##########################

        render_kwargs_train = {
            'network_query_fn' : network_query_fn,
            'perturb' : args.perturb,
            'N_fine_samples' : args.N_fine_samples,
            'network_fine' : model_fine,
            'N_samples' : args.N_samples,
            'network_fn' : model,
            'raw_noise_std' : args.raw_noise_std,
            'white_bkgd' : args.white_bkgd,
            'use_viewdirs' : args.use_viewdirs,
            'model_type' : args.model,
        }

        # NDC only good for LLFF-style forward facing data
        if args.dataset_type != 'llff' or args.no_ndc:
            print('Not ndc!\n')
            render_kwargs_train['ndc'] = False
            render_kwargs_train['lindisp'] = args.lindisp

        render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
        render_kwargs_test['perturb'] = False
        render_kwargs_test['raw_noise_std'] = 0.

        return render_kwargs_train,render_kwargs_test, optimizer, start, epoch, best_psnr
    
    elif model_type == 'CNN':
        embed_fn, input_ch = get_embedder(args.multires, device)
        embeddirs_fn = None
        
        if args.use_viewdirs:
            embeddirs_fn, input_ch_views = get_embedder(args.multires_views, device)
        
        model = CNN(in_channels=1, block_num=args.netdepth, dims=args.netwidth, 
                    drop_path_rate=0.2, input_ch=input_ch, input_ch_views=input_ch_views, sample=args.N_samples, layer_scale_init_value=1e-6, 
                    head_init_scale=1.).to(device)
        
        grad_vars = list(model.parameters())
        
        # logdir = os.path.join(args.basedir, args.expname)
        # f = open(os.path.join(logdir,'stdout.txt'), 'a')
        # report, _ = torchsummary.summary_string(model, (args.N_samples, 91))
        # print(report, file=f)
        # print("\n", file=f)
        # f.close()
        # summary(model, (128, 90))
        
        model_fine = None
        if args.N_fine_samples > 0:
            model_fine = CNN(in_channels=1, block_num=args.netdepth, dims=args.netwidth, 
                    drop_path_rate=0.4, input_ch=input_ch, input_ch_views=input_ch_views, sample=128, layer_scale_init_value=1e-6, 
                    head_init_scale=1.).to(device)
            grad_vars += list(model_fine.parameters())
            
            summary(model_fine, (128, 90))
        
        network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)
        
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
        
        start = 0
        epoch = 0
        best_psnr = 0
        
        if len(ckpts) > 0 and not args.no_reload and mode=='train':
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path, '\n')
            ckpt = torch.load(ckpt_path)

            start = ckpt['global_step']
            epoch = ckpt['epoch']
            best_psnr = ckpt['best_psnr']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            # Load model
            model.load_state_dict(ckpt['network_fn_state_dict'])
            if model_fine is not None:
                model_fine.load_state_dict(ckpt['network_fine_state_dict'])
            
        if len(ckpts) > 0 and mode=='eval':
            print('Reloading from', ckpts, '\n')
            ckpt = torch.load(ckpts)

            start = ckpt['global_step']
            epoch = ckpt['epoch']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            # Load model
            model.load_state_dict(ckpt['network_fn_state_dict'])
            if model_fine is not None:
                model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        
        render_kwargs_train = {
            'network_query_fn' : network_query_fn,
            'perturb' : args.perturb,
            'N_fine_samples' : args.N_fine_samples,
            'network_fine' : model_fine,#.train(),
            'N_samples' : args.N_samples,
            'network_fn' : model.train(),
            'raw_noise_std' : args.raw_noise_std,
            'use_viewdirs' : args.use_viewdirs,
            'white_bkgd' : args.white_bkgd,
            'model_type' : args.model,
        }

        if args.dataset_type != 'llff':
            print('Not ndc!\n')
            render_kwargs_train['ndc'] = False
            render_kwargs_train['lindisp'] = args.lindisp
        
        render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
        # render_kwargs_test['network_fine'] = model_fine.eval()
        render_kwargs_test['network_fn'] = model.eval()
        render_kwargs_test['perturb'] = False
        render_kwargs_test['raw_noise_std'] = 0.
        
        return render_kwargs_train,render_kwargs_test, optimizer, start, epoch, best_psnr

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
