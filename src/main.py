from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import time
import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from callbacks import DefaultModelCallback, TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  #logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  # opt.device = torch.device('cpu') # if using only CPU 
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  # Callbacks
  log_dir = opt.save_dir + '/logs_{}'.format(time.strftime('%Y-%m-%d-%H-%M'))
  tb_writer = SummaryWriter(log_dir=log_dir)
  trainer.register_callback(DefaultModelCallback(visualization_dir=log_dir))
  trainer.register_callback(TensorBoardCallback(tb_writer=tb_writer))

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  trainer.notify_callbacks("on_training_start", opt.num_epochs)
  
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    trainer.notify_callbacks("on_epoch_start", epoch, len(train_loader))
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    trainer.notify_callbacks("on_epoch_end", epoch, log_dict_train['loss'])
    
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      trainer.notify_callbacks("on_evaluation_start", len(val_loader))
      save_model(os.path.join(log_dir, 'model_{}.pth'.format(mark)),
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
        trainer.notify_callbacks("on_training_iteration_end", log_dict_train['loss'], log_dict_val['loss'])
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(log_dir, 'model_best.pth'),
                   epoch, model)
    else:
      save_model(os.path.join(log_dir, 'model_last.pth'),
                 epoch, model, optimizer)
    if epoch in opt.lr_step:
      save_model(os.path.join(log_dir, 'model_{}.pth'.format(epoch)),
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
if __name__ == '__main__':
  opt = opts().parse()
  main(opt)