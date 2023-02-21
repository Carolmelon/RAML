import os

from tqdm import tqdm

print(os.getpid())
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import transformers

import src.slurm
import src.util
import src.data
import src.index

from sentence_transformers import SentenceTransformer, util

from torch.utils.data import DataLoader

from src.evaluation import calculate_matches

logger = logging.getLogger(__name__)

def embed_questions(opt, data, model):
    batch_size = opt.per_gpu_batch_size * opt.world_size
    dataset = src.data.DatasetFewShot(data, question_prefix='') # amazon数据集不要前缀
    coll = src.data.CollatorFewshotSimple()
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=0, collate_fn=coll)
    model.eval()
    embedding = []
    with torch.no_grad():
        for k, batch in tqdm(enumerate(dataloader)):
            (idx, targets, texts) = batch
            output = model.encode(texts, convert_to_tensor=True, show_progress_bar=False).cpu()
            embedding.append(output)
            if opt.debug2:
                break

    embedding = torch.cat(embedding, dim=0)
    logger.info(f'Questions embeddings shape: {embedding.size()}')

    return embedding.cpu().numpy()

def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        logger.info(f'Loading file {file_path}')
        with open(file_path, 'rb') as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)
            print("allembeddings.shape[0]: {}".format(allembeddings.shape[0]))
            if args.debug:
                break
        if args.debug:
            break

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)
        if args.debug:
            break

    logger.info('Data indexing completed.')

def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits] 
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return match_stats.questions_doc_hits


def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    merged_data = []
    if not args.debug2:
        assert len(data) == len(top_passages_and_scores)
    for i, d in tqdm(enumerate(data), total=len(data)):
        # print("i: {}".format(i))
        # print('len(top_passages_and_scores): {}'.format(len(top_passages_and_scores)))
        # print("len(data): {}".format(len(data)))
        if i >= len(top_passages_and_scores):
            break
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d['ctxs'] =[
                {
                    'id': results_and_scores[0][c],
                    'title': docs[c][1],
                    'text': docs[c][0],
                    'score': scores[c],
                } for c in range(ctxs_num)
        ]

def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex['ctxs']):
            d['hasanswer'] = hasanswer[i][k]


def main(opt):
    src.util.init_logger(is_main=True)
    data = src.data.load_data_fewshot(opt.data)
    model = SentenceTransformer('msmarco-distilroberta-base-v2').cuda()

    index = src.index.Indexer(model.get_sentence_embedding_dimension(), opt.n_subquantizers, opt.n_bits)

    # index all passages
    input_paths = glob.glob(args.passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = Path(input_paths[0]).parent
    index_path = embeddings_dir / 'index.faiss'
    if args.save_or_load_index and index_path.exists():
        start = time.time()
        logger.info("start index.deserialize")
        index.deserialize_from(embeddings_dir)
        logger.info("end index.deserialize")
    else:
        logger.info(f'Indexing passages from files {input_paths}')
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, opt.indexing_batch_size)
        logger.info(f'Indexing time: {time.time()-start_time_indexing:.1f} s.')
        if args.save_or_load_index:
            index.serialize(embeddings_dir)

    questions_embedding = embed_questions(opt, data, model)

    # get top k results
    start_time_retrieval = time.time()
    top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs) 
    logger.info(f'Search time: {time.time()-start_time_retrieval:.1f} s.')

    passages = src.util.load_passages(args.passages)
    passages = {x[0]:(x[1], x[2]) for x in passages}

    add_passages(data, passages, top_ids_and_scores)
    # hasanswer = validate(data, args.validation_workers)
    # add_hasanswer(data, hasanswer)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, 'w') as fout:
        json.dump(data, fout, indent=4)
    with open(args.output_path + '.pkl', 'wb') as fout:
        pickle.dump(data, fout)
    logger.info(f'Saved results to {args.output_path} and {args.output_path + ".pkl"}')

'''/data/lrs/anaconda3/envs/fid3/bin/python /data/lrs/test/fid/passage_retrieval.py --passages open_domain_data/psgs_w100.tsv --data data_fewshot/Pure_Huffpost.json --passages_embeddings wikipedia_embeddings/passages_00 --output_path wikipedia_embeddings/TQA_retrieved_data_train.json --n-docs 10'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
python passage_retrieval.py 
    --model_path <model_dir> 
    --passages open_domain_data/psgs_w100.tsv 
    --data open_domain_data/TQA/train.json
    --passages_embeddings wikipedia_embeddings/passages_00 
    --output_path wikipedia_embeddings/retrieved_data.json 
    --n-docs 100 
    '''
    parser.add_argument('--data', required=True, type=str, default=None, 
                        help=".json file containing question and answers, similar format to reader data")
    parser.add_argument('--passages', type=str, default=None, help='Path to passages (.tsv file)')
    parser.add_argument('--passages_embeddings', type=str, default=None, help='Glob path to encoded passages')
    parser.add_argument('--output_path', type=str, default=None, help='Results are written to output_path')
    parser.add_argument('--n-docs', type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument('--validation_workers', type=int, default=32,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--per_gpu_batch_size', type=int, default=64, help="Batch size for question encoding")
    parser.add_argument("--save_or_load_index", action='store_true', 
                        help='If enabled, save index and load index if it exists')
    parser.add_argument('--model_path', type=str, help="path to directory containing model weights and config file")
    parser.add_argument('--no_fp16', action='store_true', help="inference in fp32")
    parser.add_argument('--passage_maxlength', type=int, default=200, help="Maximum number of tokens in a passage")
    parser.add_argument('--question_maxlength', type=int, default=40, help="Maximum number of tokens in a question")
    parser.add_argument('--indexing_batch_size', type=int, default=50000, help="Batch size of the number of passages indexed")
    parser.add_argument("--n-subquantizers", type=int, default=0, 
                        help='Number of subquantizer used for vector quantization, if 0 flat index is used')
    parser.add_argument("--n-bits", type=int, default=8, 
                        help='Number of bits per subquantizer')
    parser.add_argument("--debug", type=int, default=0,
                        help='reduce parameters quantity')
    parser.add_argument("--debug2", type=int, default=1,
                        help='reduce parameters quantity')
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument('--amzn', type=int, default=0)

    args = parser.parse_args()
    print("debug: {}".format(args.debug))
    print("debug2: {}".format(args.debug2))
    print(f"save_or_load_index: {args.save_or_load_index}")
    print(f'amzn: {args.amzn}')
    main(args)