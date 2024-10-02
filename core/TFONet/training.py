import torch
import os
import logging
import time
import ast
import numpy as np
from core.optimizer import FLOPSOptimizer, MixedTrainOptimizer, ScheduledOptim, ZO_SGD_mask, ZO_SCD_mask
from core.GraSP.utils.common_utils import PresetLRScheduler

from core.ZO_Estim.ZO_Estim_entry import build_obj_fn


def train(config, model, dataset, optimizer, scheduler, ZO_Estim, epochs, epochs_til_checkpoints, model_dir, loss_fn,
          val_fn=None, device=None, start_epoch=0, lr_decay=True, epochs_til_val=500, verbose=True, log=True, in_logger=None, writer=None):

    # if start_epoch > 0:
    #     model_path = os.path.join(model_dir, 'checkpoints', 'model_epoch_{}.pth'.format(start_epoch))
    #     checkpoint = torch.load(model_path)
    #     model.load_state_dict(checkpoint['model'])

    #     if isinstance(optimizer, ScheduledOptim):
    #         optimizer._optimizer.load_state_dict(checkpoint['optim'])
    #     else:
    #         optimizer.load_state_dict(checkpoint['optim'])

    #     if scheduler is not None:
    #         scheduler.load_state_dict(checkpoint['scheduler'])

    #     model.train()
    #     assert (start_epoch == checkpoint['epoch'])

    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    figure_dir = os.path.join(model_dir, 'figures')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    if in_logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        fh = logging.FileHandler('{}/log.log'.format(model_dir))
        fh.setLevel(logging.CRITICAL)  # or any level you want
        logger.addHandler(fh)
    else:
        logger = in_logger

    if start_epoch == 0:
        # init_msg = "Num of params: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad))

        if log:
            logger.critical(str(model_dir))
            logger.critical(os.getpid())
            logger.critical(torch.initial_seed())
            num_param = sum(p.numel() for p in model.parameters())
            num_param_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.critical("Num of params: {}".format(num_param))
            logger.critical("Num of params require grad: {}".format(num_param_grad))

            logger.critical(model)

        # if verbose:
        #     print(init_msg)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    best_test_loss = float('inf')
    if config['training'].get('optimizer') == 'ZO_mix': 
        patience = config['training'].getint('patience') 
    
    ### Random initialization
    if val_fn:
        model.eval()
        with torch.no_grad():
            test_loss = val_fn(model, dataset, 0, figure_dir, device)
        writer.add_scalar('test/loss', test_loss, 0)
        
        msg = 'epoch: 0, lr: {:.6f}, training loss: , test loss: {:.4e}, num_forward: 0'.format(optimizer.state_dict()['param_groups'][0]['lr'], test_loss.item())
    else:
        msg = 'epoch: 0, lr: {:.6f}, training loss: , num_forward: 0'.format(optimizer.state_dict()['param_groups'][0]['lr'])

    if log:
        logger.critical(msg)

    if verbose:
        print(msg)
    
    for epoch in range(start_epoch, epochs + 1):

        model.train()
        
        if isinstance(scheduler, PresetLRScheduler):
            scheduler(optimizer, epoch)
            now_lr = scheduler.get_lr(optimizer)
        else:
            now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        optimizer.zero_grad()
        # train_loss = loss_fn(model=model, dataset=dataset)
        # train_loss.backward()

        if ZO_Estim is not None:
            obj_fn_type = ZO_Estim.obj_fn_type
            with torch.no_grad():
                inputs = dataset.train()
                train_loss = loss_fn(model=model, dataset=dataset, inputs=inputs)
                ### pinn obj_fn
                obj_fn = build_obj_fn(obj_fn_type, model=model, dataset=dataset, loss_fn=loss_fn, inputs=inputs)
                ZO_Estim.update_obj_fn(obj_fn)
                ZO_Estim.estimate_grad(old_loss=train_loss)
            
            ###### Test ######
            # import copy
            # ZO_grad = copy.deepcopy(model.net[2].weight.factors.factor_0.grad.data)
            
            # # this_layer = model.net[2].tt_cores[0]
            # # ZO_grad = copy.deepcopy(this_layer.phase_S.grad.data)
            # # ZO_grad = - this_layer.phase_S.grad.data / this_layer.phase_S.data.sin().mul_(this_layer.S_scale)
            # # ZO_grad[torch.isinf(ZO_grad)] = 0
            
            # grad_list = []
            # for splited_param in ZO_Estim.splited_param_list:
            #     if 'phase_S' in splited_param.name:
            #         grad_S = (- splited_param.param.grad.data / splited_param.param.data.sin().mul_(splited_param.layer.S_scale)).view(-1)
            #         grad_S[torch.isinf(grad_S)] = 0
            #         grad_list.append(grad_S)
            #     else:
            #         grad_list.append(splited_param.param.grad.data.view(-1))
            
            # model_ZO_grad = torch.cat(grad_list, dim=0)
            
            # optimizer.zero_grad()
            # FO_loss = loss_fn(model=model, dataset=dataset, inputs=inputs)
            # FO_loss.backward()
            
            # FO_grad = copy.deepcopy(model.net[2].weight.factors.factor_0.grad.data)
            # # FO_grad = copy.deepcopy(this_layer.S.grad.data)
            
            # grad_list = []
            # for splited_param in ZO_Estim.splited_param_list:
            #     grad_list.append(splited_param.param.grad.data.view(-1))
            
            # model_FO_grad = torch.cat(grad_list, dim=0)
            
            # print(f'layer cos sim {torch.nn.functional.cosine_similarity(ZO_grad.view(-1), FO_grad.view(-1), dim=0)}')
            # print(f'layer MSE loss {torch.nn.functional.mse_loss(ZO_grad, FO_grad)}')
            # print(f'layer FO_grad_norm {torch.norm(FO_grad)}')
            # print(f'layer ZO_grad_norm {torch.norm(ZO_grad)}')
            
            # print(f'model cos sim {torch.nn.functional.cosine_similarity(model_ZO_grad.view(-1), model_FO_grad.view(-1), dim=0)}')
            # print(f'model MSE loss {torch.nn.functional.mse_loss(model_ZO_grad, model_FO_grad)}')
            # print(f'model FO_grad_norm {torch.norm(model_FO_grad)}')
            # print(f'model ZO_grad_norm {torch.norm(model_ZO_grad)}')
            
            ###### Test ######
            
            optimizer.step()
            if ZO_Estim.en_param_commit:
                for splited_param in ZO_Estim.splited_param_list:
                    if hasattr(splited_param, 'commit_fn'):
                        splited_param.commit_fn()
        
        else:
            # test, check real grads
            en_debug = config['training'].getboolean('debug')

            if isinstance(optimizer, (ZO_SCD_mask, ZO_SGD_mask, FLOPSOptimizer, MixedTrainOptimizer)):
                pass
            else:
                train_loss = loss_fn(model=model, dataset=dataset)
                ##### Calculate the penalty for weights outside the range
                try:
                    if config['training'].getint('loss_penalty') > 0: 
                        max_val = 0.9
                        min_val = 0.02
                        # min_val = -0.9
                        penalty = torch.tensor(0.0, device=device)
                        for name, param in model.named_parameters():
                            if 'weight' in name:
                                penalty += torch.sum(torch.relu(param - max_val) + torch.relu(min_val - param))
                        train_loss = train_loss + penalty
                except:
                    pass
                
                train_loss.backward()

            if isinstance(optimizer, (ZO_SCD_mask)):
                with torch.no_grad():
                    outputs, train_loss, grads = optimizer.step_pinn(model, dataset, loss_fn, en_debug=en_debug)
            elif isinstance(optimizer, ZO_SGD_mask):
                with torch.no_grad():
                    outputs, train_loss, grads = optimizer.step_pinn(model, dataset, loss_fn, en_debug=en_debug)   
            elif isinstance(optimizer, (FLOPSOptimizer, MixedTrainOptimizer)):
                with torch.no_grad():
                    outputs, train_loss = optimizer.step_pinn(model, dataset, loss_fn, en_debug=en_debug)   
            elif isinstance(optimizer, ScheduledOptim):
                optimizer.step_and_update_lr()
            else:
                optimizer.step()
            
            if epoch == start_epoch:
                if isinstance(optimizer, (ZO_SCD_mask, ZO_SGD_mask)):
                    logger.critical("ZO forward call per iter: {}".format(optimizer.get_forward_cnt()))
                    logger.critical("ZO optimizer dimension: {}".format(optimizer.get_param_num()))
                # end.record()
                # torch.cuda.synchronize()
                # logger.critical("iter time: {}".format(start.elapsed_time(end)))

        ##### Weight clamping
        try:
            if config['training'].getboolean('weight_clamp') == True: 
                max_val = 0.9
                # min_val = 0.02
                min_val = -0.9
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        param.data.clamp_(min_val, max_val)
        except:
            pass

        if lr_decay and isinstance(scheduler, PresetLRScheduler)==False:
            scheduler.step()

        writer.add_scalar('train/loss', train_loss, epoch)
        
        msg = 'epoch: {:d}, training loss: {:.4e}'.format(epoch, train_loss.item())
        if verbose:
            logger.critical(msg)

        # validation
        # if not epoch % epochs_til_val:
        if (epoch+1) % epochs_til_val == 0:
            if isinstance(optimizer, (ZO_SCD_mask, ZO_SGD_mask, FLOPSOptimizer, MixedTrainOptimizer)):
                num_forward = optimizer.get_forward_cnt()
            elif ZO_Estim is not None:
                num_forward = ZO_Estim.get_forward_cnt()
            else:
                num_forward = epoch+1

            writer.add_scalar('forward num', num_forward, epoch)
            
            if val_fn:
                model.eval()
                with torch.no_grad():
                    test_loss = val_fn(model, dataset, epoch, figure_dir, device)
                writer.add_scalar('test/loss', test_loss, epoch)
                
                msg = 'epoch: {:d}, lr: {:.6f}, training loss: {:.4e}, test loss: {:.4e}, num_forward: {:d}'.format(epoch+1, now_lr, train_loss.item(), test_loss.item(), num_forward)
            else:
                msg = 'epoch: {:d}, lr: {:.6f}, training loss: {:.4e}, num_forward: {:d}'.format(epoch+1, now_lr, train_loss.item(), num_forward)

            if log:
                logger.critical(msg)

            if verbose:
                print(msg)
            
            # ================= save best checkpoint =================
            if val_fn and test_loss < best_test_loss:
                best_test_loss = test_loss    
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                }

                checkpoint['optim'] = optimizer.state_dict()

                if scheduler is not None:
                    checkpoint['scheduler'] = scheduler.state_dict()

                torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pth'))

            if val_fn and test_loss < best_test_loss:
                best_test_loss = test_loss
            else:
                if config['training'].get('optimizer') == 'ZO_mix' and patience > 0: 
                    patience -= 1
                    if patience == 0:
                        logger.critical('Optimizer changed')
                        raw_opt_layers_strs = config['training'].get('opt_layers_strs') if not None else None
                        if raw_opt_layers_strs is not None:
                            opt_layers_strs = ast.literal_eval(raw_opt_layers_strs)
                        optimizer = ZO_SCD_mask(
                            model = model, 
                            criterion = loss_fn,
                            masks = None,
                            lr = config['training'].getfloat('lr'),
                            grad_sparsity = config['training'].getfloat('grad_sparsity'),
                            h_smooth = config['training'].getfloat('h_smooth') if not None else 0.1,
                            grad_estimator = config['training'].get('grad_estimator') if not None else 'sign',
                            opt_layers_strs = opt_layers_strs,
                            STP = config['training'].getboolean('STP') if not None else False,
                            momentum = config['training'].getfloat('momentum') if not None else 0
                        )

                        epochs_til_decay = config['training'].getint('epochs_til_decay')
                        gamma=config['training'].getfloat('gamma') if not None else 0.9
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs_til_decay, gamma=gamma)  

        # save checkpoint
        if not epoch % epochs_til_checkpoints and epoch:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict()
            }

            if isinstance(optimizer, ScheduledOptim):
                checkpoint['optim'] = optimizer._optimizer.state_dict()
            else:
                checkpoint['optim'] = optimizer.state_dict()

            if scheduler is not None:
                if isinstance(scheduler, PresetLRScheduler):
                    pass
                else:
                    checkpoint['scheduler'] = scheduler.state_dict()

            torch.save(checkpoint, os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch)))

    if isinstance(optimizer, (ZO_SCD_mask, ZO_SGD_mask)):
        logger.critical("ZO forward call to converge: {}".format(optimizer.get_forward_cnt()))
