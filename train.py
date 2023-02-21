import os
import random

print(os.getpid())
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options
import pickle

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
import src.data_generator_cls_name
from transformers import AutoTokenizer, AutoModel

from src.model import AlbertCrossAttention, albert_atten_config_obj

def train(model, optimizer, scheduler, step, eval_dataset, opt, best_dev_em, checkpoint_path):
    str_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time() + 60*60*8))
    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path('runs')/opt.name/str_time)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    # k-shot, k-query
    src.data_generator_cls_name.args.update_batch_size = opt.update_batch_size
    src.data_generator_cls_name.args.update_batch_size_eval = opt.update_batch_size_eval

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank

    # load dict_data
    if opt.datasource == 'huffpost':
        with open("data_fewshot/dict_data_sts_retrieved_100.pkl", 'rb') as file_pkl:
            data_pkl = pickle.load(file_pkl)
    elif opt.datasource == 'amazonreview':
        with open("data_amzn/amazon_500_dict_data2.pkl", 'rb') as file_pkl:
            data_pkl = pickle.load(file_pkl)
    else:
        logger.info("invalid opt.datasource, please check!")
        exit(1)

    # train dataset
    huffpost_train = src.data_generator_cls_name.Huffpost(src.data_generator_cls_name.args,
                     'train', opt.text_maxlength, None, opt.n_context, opt.answer_maxlength,
                     data=data_pkl, opt=opt)
    # test dataset
    # 5-way * 5-query = 25
    src.data_generator_cls_name.args.update_batch_size_eval = 5
    huffpost_test = src.data_generator_cls_name.Huffpost(src.data_generator_cls_name.args,
                     'test', opt.text_maxlength, None, opt.n_context, opt.answer_maxlength,
                     data=data_pkl, opt=opt)

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    # support的cls抽取器
    support_cls_model = AutoModel.from_pretrained("albert-base-v1").cuda()
    args = src.data_generator_cls_name.args
    # 冻结encoder参数
    if opt.freeze_bert:
        for p in support_cls_model.parameters():
            p.requires_grad = False
    # 新的decoder
    decoder2 = AlbertCrossAttention(albert_atten_config_obj).cuda()
    decoder2.load_state_dict(
        support_cls_model.encoder.albert_layer_groups[0].albert_layers[0].attention.state_dict()
    )
    model.decoder2 = decoder2
    # 新的optimizer
    # 冻住encoder参数
    if not opt.freeze_bert:  # 如果不冻结参数
        optimizer.add_param_group({'params': support_cls_model.parameters(), 'lr': opt.lr})
    optimizer.add_param_group({'params': decoder2.parameters(), 'lr': opt.lr*2})
    # scheduler
    from src.util import WarmupLinearScheduler
    scheduler = WarmupLinearScheduler(optimizer, warmup_steps=opt.warmup_steps,
scheduler_steps=src.data_generator_cls_name.args.metatrain_iterations,min_ratio=0., fixed_lr=opt.fixed_lr)
    # 统计
    acc = []

    data_time = []
    train_time = []
    for step, (x_spt, y_spt, y_true_sqt, x_qry, y_qry, y_true_qry) in enumerate(huffpost_train):
        time1 = time.time()
        if step:
            data_time.append(time1-time3)
        if step > src.data_generator_cls_name.args.metatrain_iterations:
            break
        task_losses = []
        task_acc = []
        for meta_batch in range(src.data_generator_cls_name.args.meta_batch_size):
            x_sqt_batch, y_sqt_batch, y_true_sqt_batch,\
            x_qry_batch, y_qry_batch, y_true_qry_batch = \
                x_spt[meta_batch], y_spt[meta_batch], y_true_sqt[meta_batch], \
                x_qry[meta_batch], y_qry[meta_batch], y_true_qry[meta_batch]
            # support的样本，去掉passages的维度，从[5,1,250]变成[5,250]
            x_sqt_batch_processed = x_sqt_batch[3].reshape(x_sqt_batch[3].shape[0], x_sqt_batch[3].shape[2]), \
                                    x_sqt_batch[4].reshape(x_sqt_batch[3].shape[0], x_sqt_batch[3].shape[2])
            # query的样本
            x_qry_batch_processed = x_qry_batch[3], x_qry_batch[4]
            # support
            z_proto = support_cls_model(x_sqt_batch_processed[0].cuda(),
                                        x_sqt_batch_processed[1].cuda(),
                                        torch.zeros_like(x_sqt_batch_processed[0]).cuda())
            # 取cls
            # z_proto = z_proto[0][:, 0, :]
            # 取字平均，结合掩码
            z_proto = [z_proto[0][i][x_sqt_batch_processed[1][i]].mean(dim=0) for i in range(z_proto[0].shape[0])]
            z_proto = torch.stack(z_proto)

            z_proto = z_proto.reshape([args.num_classes, args.update_batch_size, z_proto.shape[-1]])
            z_proto = z_proto.mean(1)
            z_proto = z_proto.unsqueeze(dim=0)
            # z_proto = z_proto.repeat(x_qry_batch_processed[0].shape[0], 1, 1)

            model.encoder2 = support_cls_model
            model_return = model(
                input_ids=x_qry_batch_processed[0].cuda(),
                attention_mask=x_qry_batch_processed[1].cuda(),
                # labels=labels.cuda()
                decoder_inputs_embeds=z_proto,
                labels=y_qry_batch.cuda()
            )
            train_loss = model_return[0]
            predict_cur = model_return[1].squeeze()
            _, predict_cur_indices = predict_cur.detach().cpu().max(dim=-1)
            acc.append((predict_cur_indices == y_qry_batch).sum() / x_qry_batch_processed[0].shape[0])

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            support_cls_model.zero_grad()
        time2 = time.time()
        train_time.append(time2-time1)
        if (step + 1) % opt.eval_freq == 0:
            cur_acc = torch.stack(acc)
            index = step//opt.eval_freq
            cur_acc_batch = cur_acc[index*opt.eval_freq: (index+1)*opt.eval_freq].mean()
            print("step: {}, cur_acc_batch: {}".format(step, cur_acc_batch))
            for i, param in enumerate(optimizer.param_groups):
                print('[ lr{}: {:.7f}'.format(i, param['lr']), end='    ')
                tb_logger.add_scalar('lr{}'.format(i), param['lr'], step)
            print(' ') # 换行
            tb_logger.add_scalar('train_loss', train_loss, step)
            tb_logger.add_scalar('cur_acc_batch', cur_acc_batch, step)

            # 保存参数信息
            tb_logger.add_scalar('update_batch_size_eval', huffpost_train.k_query, step) # k-query
            tb_logger.add_scalar('update_batch_size', huffpost_train.k_shot, step)   # k-shot
            tb_logger.add_scalar('text_maxlength', opt.text_maxlength, step)  # text_maxlength
            tb_logger.add_scalar('n-context', opt.n_context, step)   # n-context
        if (step + 1) % (opt.eval_freq) == 0:
            test(src.data_generator_cls_name.args, huffpost_test, model, support_cls_model, step, tb_logger)
            # loss_val, acc_val = model(x_spt[meta_batch], y_spt[meta_batch], y_true_sqt[meta_batch],
            #                              x_qry[meta_batch], y_qry[meta_batch], y_true_qry[meta_batch])
            # task_losses.append(loss_val)
            # task_acc.append(acc_val)
        if (step + 1) % opt.eval_freq == 0:
            logger.info("data_time: " + str(sum(data_time)))
            logger.info("train_time: " + str(sum(train_time)))
        time3 = time.time()
    tb_logger.close()

if False:
    while step < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            (idx, labels, _, context_ids, context_mask) = batch

            model_return = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                # labels=labels.cuda()
                decoder_inputs_embeds=torch.randn((context_ids.shape[0], 5, 768)).cuda(),
                labels=torch.tensor([random.choice(range(5)) for i in range(context_ids.shape[0])]).cuda()
            )
            train_loss = model_return[0]

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.eval_freq == 0:
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break

def test(args, dataset, t5_model, support_cls_model, train_step, logger, type='test'):
    res_acc = []

    for step, (x_spt, y_spt, y_true_sqt, x_qry, y_qry, y_true_qry) in enumerate(dataset):
        if step > 300:  # 训练3000次，测试300次*25个query
            break
        # x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to("cuda"), y_spt.squeeze(0).to("cuda"), \
        #                              x_qry.squeeze(0).to("cuda"), y_qry.squeeze(0).to("cuda")
        with torch.no_grad():
            x_sqt_batch, y_sqt_batch, y_true_sqt_batch, \
            x_qry_batch, y_qry_batch, y_true_qry_batch = \
                x_spt[0], y_spt[0], y_true_sqt[0], \
                x_qry[0], y_qry[0], y_true_qry[0]
            # support的样本
            x_sqt_batch_processed = x_sqt_batch[3].reshape(x_sqt_batch[3].shape[0], x_sqt_batch[3].shape[2]), \
                                    x_sqt_batch[4].reshape(x_sqt_batch[3].shape[0], x_sqt_batch[3].shape[2])
            # query的样本
            x_qry_batch_processed = x_qry_batch[3], x_qry_batch[4]
            # support
            z_proto = support_cls_model(x_sqt_batch_processed[0].cuda(),
                                        x_sqt_batch_processed[1].cuda(),
                                        torch.zeros_like(x_sqt_batch_processed[0]).cuda())
            # 取cls
            # z_proto = z_proto[0][:, 0, :]
            # 取字平均
            z_proto = [z_proto[0][i][x_sqt_batch_processed[1][i]].mean(dim=0) for i in range(z_proto[0].shape[0])]
            z_proto = torch.stack(z_proto)

            z_proto = z_proto.reshape([args.num_classes, args.update_batch_size, z_proto.shape[-1]])
            z_proto = z_proto.mean(1)
            z_proto = z_proto.unsqueeze(dim=0)
                # .repeat(x_qry_batch_processed[0].shape[0], 1, 1)

            model.encoder2 = support_cls_model
            model_return = model(
                input_ids=x_qry_batch_processed[0].cuda(),
                attention_mask=x_qry_batch_processed[1].cuda(),
                # labels=labels.cuda()
                decoder_inputs_embeds=z_proto,
                labels=y_qry_batch.cuda()
            )
            train_loss = model_return[0]
            predict_cur = model_return[1].squeeze()
            _, predict_cur_indices = predict_cur.detach().cpu().max(dim=-1)
            res_acc.append((predict_cur_indices == y_qry_batch).sum() / x_qry_batch_processed[0].shape[0])

    res_acc = torch.stack(res_acc)

    print('                                     {}_epoch is {}, acc is {}'.format(type, train_step, res_acc.mean()))
    logger.add_scalar('test_acc', res_acc.mean(), train_step)

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    options.add_few_shot_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    #load data

    # use golbal rank and world size to split the eval set on multiple gpus

    # use golbal rank and world size to split the eval set on multiple gpus

    # eval_examples = src.data.load_data(
    #     opt.eval_data,
    #     global_rank=opt.global_rank,
    #     world_size=opt.world_size,
    # )
    # eval_dataset = src.data.DatasetFewShot(eval_examples, opt.n_context)
    if True:
        model = src.model.Fewshot()
        model.lm_head = torch.nn.Linear(in_features=768, out_features=1, bias=True).cuda()
        model = model.to(opt.local_rank)
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        step, best_dev_em = 0, 0.0

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")

    train(
        model,
        optimizer,
        None,
        step,
        None,
        opt,
        best_dev_em,
        checkpoint_path
    )
