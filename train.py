import os
import time
import numpy as np

from models import create_model
from data import CreateDataLoader
from util.visualizer import Visualizer
from options.train_options import TrainOptions


def print_log(logger,message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')


if __name__ == '__main__':
    opt = TrainOptions().parse()
    
    # Load train data
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print(f'Training set size: {len(data_loader)}')
    
    # Load val data
    opt.phase = 'val'
    data_loader_val = CreateDataLoader(opt)
    dataset_val = data_loader_val.load_data()
    print(f'Validation set size: {len(data_loader_val)}')
    
    # Set logger
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger = open(os.path.join(save_dir, 'log.txt'), 'w+')
    print_log(logger,opt.name)
    logger.close()
    
    if opt.model=='cycle_gan':
        L1_avg=np.zeros([2,opt.niter + opt.niter_decay,len(dataset_val)])      
    else:
        L1_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)])      
    
    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        
        # Training phase
        opt.phase = 'train'
        
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                
                if opt.dataset_mode=='aligned_mat':
                    temp_visuals=model.get_current_visuals()
                    visualizer.display_current_results(temp_visuals, epoch, save_result)
                
                elif opt.dataset_mode=='unaligned_mat':   
                    temp_visuals=model.get_current_visuals()
                    temp_visuals['real_A']=temp_visuals['real_A'][:,:,0:3]
                    temp_visuals['real_B']=temp_visuals['real_B'][:,:,0:3]
                    temp_visuals['fake_A']=temp_visuals['fake_A'][:,:,0:3]
                    temp_visuals['fake_B']=temp_visuals['fake_B'][:,:,0:3]
                    temp_visuals['rec_A']=temp_visuals['rec_A'][:,:,0:3]
                    temp_visuals['rec_B']=temp_visuals['rec_B'][:,:,0:3]
                    
                    if opt.lambda_identity>0:
                        temp_visuals['idt_A']=temp_visuals['idt_A'][:,:,0:3]
                        temp_visuals['idt_B']=temp_visuals['idt_B'][:,:,0:3]                    
                   
                    visualizer.display_current_results(temp_visuals, epoch, save_result)                    
                
                else:
                    temp_visuals=model.get_current_visuals()
                    visualizer.display_current_results(temp_visuals, epoch, save_result)                    
                    
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / len(data_loader), opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()
        
        # Validaiton phase
        if epoch % opt.save_epoch_freq == 0:
            logger = open(os.path.join(save_dir, 'log.txt'), 'a')
            print(opt.dataset_mode)
            opt.phase='val'
            
            for i, data_val in enumerate(dataset_val):
                model.set_input(data_val)
                model.test()
                fake_im=model.fake_B.cpu().data.numpy()
                real_im=model.real_B.cpu().data.numpy() 
                real_im=real_im*0.5+0.5
                fake_im=fake_im*0.5+0.5
                
                if real_im.max() <= 0:
                    continue
                
                L1_avg[epoch-1,i]=abs(fake_im-real_im).mean()

            l1_avg_loss = np.mean(L1_avg[epoch-1])

            print_log(logger,'Epoch %3d   l1_avg_loss: %.5f' % (epoch, l1_avg_loss))
            print_log(logger,'')
            logger.close()
            print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))

            # Save model
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        
        # Update learning rate
        model.update_learning_rate()
