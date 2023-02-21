import os
print(os.getpid())
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import argparse
import csv
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import transformers

import src.data
import src.util
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

def embed_passages(opt, passages, model):
    batch_size = opt.per_gpu_batch_size * opt.world_size
    dataset = src.data.PureTextDataset(passages)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=20)
    total = 0
    allids, allembeddings = [], []
    with torch.no_grad():
        for k, (ids, texts) in enumerate(dataloader):
            allembeddings.append(model.encode(texts, convert_to_tensor=True).cpu())
            total += len(ids)
            allids.append(ids)
            if (k + 1) % 100 == 0:
                logger.info('Encoded passages %d', total)

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    allids = [x for idlist in allids for x in idlist]
    return allids, allembeddings


def main(opt):
    logger = src.util.init_logger(is_main=True)
    model = SentenceTransformer('msmarco-distilroberta-base-v2').cuda()

    all_passages = src.util.load_passages(args.passages)
    start = opt.start
    end = opt.end if opt.end > 0 and opt.end < len(all_passages) else len(all_passages)
    passages = all_passages[start: end]

    logger.info(f'Embedding generation for {len(all_passages)} passages from idx {start} to {end}')

    allids, allembeddings = embed_passages(opt, passages, model)

    output_path = Path(args.output_path)
    save_file = output_path.parent / (output_path.name + f'_start{start}_end{end}')
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    logger.info(f'Saving {len(allids)} passage embeddings to {save_file}')
    with open(save_file, mode='wb') as f:
        pickle.dump((allids, allembeddings), f, protocol=4)

    logger.info(f'Total passages processed {len(allids)}. Written to {save_file}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    --passages
    open_domain_data/psgs_w100.tsv
    --shard_id
    0
    --num_shards
    1
    --per_gpu_batch_size
    2000
    '''
    parser.add_argument('--passages', type=str, default=None, help='Path to passages (.tsv file)')
    parser.add_argument('--output_path', type=str, default='wikipedia_embeddings/passages', help='prefix path to save embeddings')
    parser.add_argument('--shard_id', type=int, default=0, help="Id of the current shard")
    parser.add_argument('--num_shards', type=int, default=1, help="Total number of shards")
    parser.add_argument('--per_gpu_batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    parser.add_argument('--passage_maxlength', type=int, default=200, help="Maximum number of tokens in a passage")
    parser.add_argument('--model_path', type=str, help="path to directory containing model weights and config file")
    parser.add_argument('--no_fp16', action='store_true', help="inference in fp32")
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    args = parser.parse_args()

    main(args)
