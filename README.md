This repository contains code for:
- Retrieval-Augmented Meta Learning for Low-Resource Text Classification (RAML).

# Dependencies

- Python 3
- [PyTorch](http://pytorch.org/)
- [Transformers](http://huggingface.co/transformers/)
- [NumPy](http://www.numpy.org/)

# Data

### Download data
- Huffpost and Amazon review: Download in [Google Drive](https://drive.google.com/drive/folders/16CZAi9_FgiulK7m7bXrnldkDMzjjnRzA).
- External knowledge for retrieval: Wikipedia dump which are splited in 100 words for each passage, click [here](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz) to download (about 4.4GB).

### Data format

The expected data format is a list of entry examples, where each entry example is a dictionary containing
- `headline`: query text.
- `target`: label corresponding to query text, whose type is number.
- `ctxs`: a list of passages where each item is a dictionary containing
        - `title`: article title
            - `text`: passage text

Entry example:
```python
{
  'question': 'Live It Up',
  'target': '3',	# the label of query text.
  'ctxs': [
            {
                "title": "Live It Up (Nicky Jam song)",
                "text": "Live It Up (Nicky Jam song) \"Live It Up\" is a song by American singer Nicky Jam featuring American rapper Will Smith and Kosovar singer Era Istrefi. It was chosen as the official song for the 2018 FIFA World Cup. The track is produced by Diplo, The Picard Brothers, and Free School, and was released on May 25, 2018. On May 21, 2018, Will Smith posted on social media that he and singer Nicky Jam will collaborate for the official 2018 World Cup theme song. The message is: \"One Life to Live. Live it Up\". Nicky Jam also stated that:"
            },
            {
                "title": "Icon (Jaden Smith song)",
                "text": "Icon (Jaden Smith song) \"Icon\" is a song by American rapper-singer Jaden Smith from his debut studio album \"Syre\" (2017). The song was released on November 17, 2017. Written by Jaden Smith, Melvin Lewis and Omarr Rambert, the song is Jaden's most successful single as lead artist. The official music video for the song was released on 17 November 2017. It features Jaden Smith performed next to a black Tesla Model X with all the doors open. An official remix of the song featuring American singer-rapper Nicky Jam, with an accompanied music video was released on May 25, 2018. Another"
            }
          ]
}
```

# Pretrained models.

Retrieval: 

- I use sentence-BERT to get embeddings of query and Wikipedia passages. See [Sentence-BERT](https://github.com/UKPLab/sentence-transformers). 
- Specially, I use the pre-trained model [`msmarco-distilroberta-base-v2`](https://www.sbert.net/docs/pretrained-models/msmarco-v3.html) of Sentence-BERT. For the detailed process of retrieval, refer to the [`generate_passage_embeddings_sts_transformer.py`](generate_passage_embeddings_sts_transformer.py) and [`passage_retrieval_sts_transformer.py`](passage_retrieval_sts_transformer.py).

Word-level embedding:

- I use the ALBERT, specifically the huggingface's `albert-base-v1` model.

- 

# Knowledge source indexing

The Sentence-BERT is used to index a Knowledge source, Wikipedia in our case.

```shell
python3 generate_passage_embeddings_sts_transformer.py
        --passages
        path-to-passages/psgs_w100.tsv	#.tsv file
        --shard_id
        0
        --num_shards
        1
        --per_gpu_batch_size
        2000
        --output_path
        wikipedia_embeddings/sts-trans-passages
```

# Passage retrieval

After indexing, given an input query, passages can be efficiently retrieved:


```shell
python passage_retrieval.py
	--passages
    path-to-passages/psgs_w100.tsv	#.tsv file
    --data
    data_fewshot/Pure_Huffpost.json		# few-shot dataset, Huffpost or Amazon review.
    --passages_embeddings
    wiki_embeddings2/*					# the output file of the previous process. 
    --output_path
    wikipedia_embeddings/huffpost_retrieved_data.json
    --n-docs
    100
    --debug2
    0
    --save_or_load_index
```

#  Train

```shell
python train.py
    --lr
    0.00002
    --optim
    adamw
    --scheduler
    linear
    --weight_decay
    0.01
    --text_maxlength
    170
    --per_gpu_batch_size
    1
    --n_context
    5
    --total_step
    15000
    --warmup_step
    1000
    --eval_freq
    100
    --update_batch_size		# k-shot
    5
    --update_batch_size_eval	# k-query
    5
```

