import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import model
import pickle
import time
import numpy as np
import os
import json
import random
from compare_mt.rouge.rouge_scorer import RougeScorer
from transformers import BartTokenizer, BartConfig, BartModel, BartForConditionalGeneration, PegasusTokenizer
from utils import Recorder
from data_utils import to_cuda, collate_mp_bart, BartDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
from model import RankingLoss, BartMixReRanker
import math
import logging
from label_smoothing_loss import label_smoothing_loss
from nltk import sent_tokenize, word_tokenize
from datetime import datetime

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)


def base_setting(args):
    args.batch_size = getattr(args, 'batch_size', 2)
    args.epoch = getattr(args, 'epoch', 100)
    args.ext_lr = getattr(args, "ext_lr", 1e-5)
    args.lr = getattr(args, "lr", 1e-4)
    args.report_freq = getattr(args, "report_freq", 100)
    args.accumulate_step = getattr(args, "accumulate_step", 8)
    args.margin = getattr(args, "margin", 0.001)
    args.gold_margin = getattr(args, "gold_margin", 0)
    args.gold_weight = getattr(args, "gold_weight", 0)
    args.mle_weight = getattr(args, "mle_weight", 1)
    args.rank_weight = getattr(args, "rank_weight", 1)
    args.model_type = getattr(args, "model_type", "google/pegasus-xsum")
    args.warmup_steps = getattr(args, "warmup_steps", 10000)
    args.normalize = getattr(args, "normalize", True)
    args.grad_norm = getattr(args, "grad_norm", 0)
    args.seed = getattr(args, "seed", 970903)
    args.no_gold = getattr(args, "no_gold", False)
    args.pretrained = getattr(args, "pretrained", None)
    args.max_lr = getattr(args, "max_lr", 2e-3)
    args.scale = getattr(args, "scale", 0.01)
    args.score_mode = getattr(args, "score_mode", "log")
    args.datatype = getattr(args, "datatype", "cased_fine_large")
    args.dataset = getattr(args, "dataset", "xsum")
    args.max_len = getattr(args, "max_len", 80)
    args.max_num = getattr(args, "max_num", 16)
    args.smooth = getattr(args, "smooth", 0.1)
    args.total_len = getattr(args, "total_len", 512)
    args.length_penalty = getattr(args, "length_penalty", 0.6)
    args.do_sample = getattr(args, "do_sample", True)
    args.gen_max_len = getattr(args, "gen_max_len", 62)
    args.gen_min_len = getattr(args, "gen_min_len", 11)
    args.is_pegasus = getattr(args, "is_pegasus", True)
    args.adding = getattr(args, "adding", 0)


def evaluation(args):
    # load data
    base_setting(args)
    if args.is_pegasus:
        tok = PegasusTokenizer.from_pretrained(args.model_type)
    else:
        tok = BartTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp_bart, pad_token_id=tok.pad_token_id, is_test=True)
    test_set = BartDataset(f"./{args.dataset}/{args.datatype}/test", args.model_type, is_test=True, maxlen=512,
     is_sorted=False, maxnum=args.max_num, is_untok=True, total_len=args.total_len, is_pegasus=args.is_pegasus)
    dataloader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    scorer = BartMixReRanker(model_path, tok.pad_token_id, args.is_pegasus)
    if args.cuda:
        scorer = scorer.cuda()

    scorer.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=f'cuda:{args.gpuid[0]}'))
    device = f'cuda:{args.gpuid[0]}'
    scorer.eval()

    model_name = args.model_pt.split("/")[0]

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    print(model_name)
    root_dir = "./result/%s"%model_name
    mkdir(root_dir)
    mkdir("./result/%s/reference"%model_name)
    mkdir("./result/%s/candidate"%model_name)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    cnt = 0
    acc = 0
    scores = []
    records = []
    do_reranking = True
    mle_loss = 0
    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    if do_reranking:
        with torch.no_grad():
            for (i, batch) in enumerate(dataloader):
                if args.cuda:
                    to_cuda(batch, args.gpuid[0])
                samples = batch["data"]
                output = scorer(batch["src_input_ids"], batch["candidate_ids"], args.normalize, args.score_mode, args.length_penalty, adding=args.adding)
                similarity, gold_similarity = output['score'], output['summary_score']
                similarity = similarity.cpu().numpy()
                probs = output["probs"][:, :-1]  # truncate last token
                gold = batch["candidate_ids"][:, 0, 1:]  # shift right
                mle_loss += mle_fn(probs.transpose(1, 2), gold)
                # mle_loss += output["loss"]
                if i % 100 == 0:
                    print(f"test similarity: {similarity[0]}")
                    print(f"mle loss: {mle_loss / i}")
                max_ids = similarity.argmax(1)
                scores.extend(similarity.tolist())
                # acc += (max_ids == batch["scores"].cpu().numpy().argmax(1)).sum()
                for j in range(similarity.shape[0]):
                    sample = samples[j]
                    sents = sample["candidates"][max_ids[j]][0]
                    # print(" ".join(sents), file=f_out)
                    score = rouge_scorer.score("\n".join(sample["abstract"]), "\n".join(sents))
                    rouge1 += score["rouge1"].fmeasure
                    rouge2 += score["rouge2"].fmeasure
                    rougeLsum += score["rougeLsum"].fmeasure
                    with open("./result/%s/candidate/%d.dec"%(model_name, cnt), "w") as f:
                        for s in sents:
                            print(s, file=f)
                    with open("./result/%s/reference/%d.ref"%(model_name, cnt), "w") as f:
                        for s in sample["abstract"]:
                            print(s, file=f)
                    cnt += 1
        rouge1 = rouge1 / cnt
        rouge2 = rouge2 / cnt
        rougeLsum = rougeLsum / cnt
        print(f"accuracy: {acc / cnt}")
        print("rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))
        print(f"mle loss: {mle_loss / i}")
    # generation
    tokenizer = tok
    max_length = 140
    min_length = 55
    count = 1
    bsz = 4
    scorer.model.model.scoring_mode = False
    do_inference = False
    print(scorer.model.model.config)
    if do_inference:
        with open(f'./{args.dataset}/test.document') as source, open(os.path.join(root_dir, "test.out"), 'w') as fout:
            sline = source.readline().strip()
            slines = [sline]
            for sline in source:
                if count % 100 == 0:
                    print(count, flush=True)
                if count % bsz == 0:
                    with torch.no_grad():
                        dct = tokenizer.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                        summaries = scorer.generate(
                            input_ids=dct["input_ids"].to(device),
                            attention_mask=dct["attention_mask"].to(device),
                            length_penalty=args.length_penalty,
                        )
                        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                    cands = []
                    for hypothesis in dec:
                        hypothesis = hypothesis.replace("\n", " ")
                        fout.write(hypothesis + '\n')
                        fout.flush()
                    slines = []
                sline = sline.strip()
                if len(sline) == 0:
                    sline = " "
                slines.append(sline)
                count += 1
            if slines != []:
                with torch.no_grad():
                    dct = tokenizer.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                    summaries = scorer.generate(
                            input_ids=dct["input_ids"].to(device),
                            attention_mask=dct["attention_mask"].to(device),
                            length_penalty=args.length_penalty,
                    )
                    dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                    for hypothesis in dec:
                        hypothesis = hypothesis.replace("\n", " ")
                        fout.write(hypothesis + '\n')
                        fout.flush()


def test(dataloader, scorer, args, gpuid, tok, root_dir=None, do_sample=False, rank=0):
    scorer.eval()
    loss = 0
    cnt = 0
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    mle_loss = 0
    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, gpuid)
            samples = batch["data"]
            output = scorer(batch["src_input_ids"], batch["candidate_ids"], args.normalize, args.score_mode, args.length_penalty, adding=args.adding)
            similarity, gold_similarity = output['score'], output['summary_score']
            similarity = similarity * args.scale
            gold_similarity = gold_similarity * args.scale
            similarity = similarity.cpu().numpy()
            probs = output["probs"]  # [bz, seq_len, word_num]
            probs = output["probs"][:, :-1]  # truncate last token
            gold = batch["candidate_ids"][:, 0, 1:]  # shift right
            mle_loss += mle_fn(probs.transpose(1, 2), gold)
            if i % 1000 == 0:
                print(f"test similarity: {similarity[0]}")
            max_ids = similarity.argmax(1)
            for j in range(similarity.shape[0]):
                cnt += 1
                sample = samples[j]
                sents = sample["candidates"][max_ids[j]][0]
                score = rouge_scorer.score("\n".join(sample["abstract"]), "\n".join(sents))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    mle_loss = mle_loss / cnt
    loss = 1 - 2 * (rouge1 * rouge2) / (rouge1 + rouge2 + 1e-10)

    if len(args.gpuid) > 1:
        loss = torch.FloatTensor([loss]).to(gpuid)
        dist.all_reduce(loss, op=dist.reduce_op.SUM)
        loss = loss.item() / len(args.gpuid)

    if do_sample:
        device = scorer.device
        if len(args.gpuid) > 1:
            model = scorer.module
            scorer.module.model.model.scoring_mode = False
        else:
            model = scorer
            scorer.model.model.scoring_mode = False
        tokenizer = tok
        max_length = 140
        min_length = 55
        count = 1
        bsz = 8
        with open(f'./{args.dataset}/val.document') as source:
            lines = source.readlines()
        num = len(lines)
        local_num = num // len(args.gpuid)
        with open(os.path.join(root_dir, f"val.ours.source.{rank}"), "w") as f:
            if rank == len(args.gpuid) - 1:
                f.writelines(lines[local_num * rank:])
            else:
                f.writelines(lines[local_num * rank: local_num * (rank + 1)])
        with open(f'./{args.dataset}/val.summary') as source:
            lines = source.readlines()
        num = len(lines)
        local_num = num // len(args.gpuid)
        with open(os.path.join(root_dir, f"val.ours.target.{rank}"), "w") as f:
            if rank == len(args.gpuid) - 1:
                f.writelines(lines[local_num * rank:])
            else:
                f.writelines(lines[local_num * rank: local_num * (rank + 1)])

        with open(os.path.join(root_dir, f"val.ours.source.{rank}")) as source, open(os.path.join(root_dir, f"tmp.out.{rank}"), 'w') as fout:
            sline = source.readline().strip()
            slines = [sline]
            for sline in source:
                if count % bsz == 0:
                    with torch.no_grad():
                        dct = tokenizer.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                        summaries = model.generate(
                            input_ids=dct["input_ids"].to(device),
                            attention_mask=dct["attention_mask"].to(device),
                            length_penalty=args.length_penalty,
                        )
                        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                    cands = []
                    for hypothesis in dec:
                        hypothesis = hypothesis.replace("\n", " ")
                        fout.write(hypothesis + '\n')
                        fout.flush()
                    slines = []
                sline = sline.strip()
                if len(sline) == 0:
                    sline = " "
                slines.append(sline)
                count += 1
            if slines != []:
                with torch.no_grad():
                    dct = tokenizer.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                    summaries = model.generate(
                            input_ids=dct["input_ids"].to(device),
                            attention_mask=dct["attention_mask"].to(device),
                            length_penalty=args.length_penalty,
                    )
                    dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                    for hypothesis in dec:
                        hypothesis = hypothesis.replace("\n", " ")
                        fout.write(hypothesis + '\n')
                        fout.flush()
        with open(os.path.join(root_dir, f"val.ours.target.{rank}")) as f:
            refs = f.readlines()
        with open(os.path.join(root_dir, f"tmp.out.{rank}")) as f:
            cands = f.readlines()
        def process(x):
            return sent_tokenize(" ".join(word_tokenize(x.strip())))
        sample_rouge1, sample_rouge2, sample_rougeLsum = 0, 0, 0
        for (x, y) in zip(refs, cands):
            x = process(x)
            y = process(y)
            score = rouge_scorer.score("\n".join(x), "\n".join(y))
            sample_rouge1 += score["rouge1"].fmeasure / count
            sample_rouge2 += score["rouge2"].fmeasure / count
            sample_rougeLsum += score["rougeLsum"].fmeasure / count
        mle_loss = 1 - 2 * (sample_rouge1 * sample_rouge2) / (sample_rouge1 + sample_rouge2 + 1e-10)
        mle_loss = torch.FloatTensor([mle_loss]).to(gpuid)

        print(f"mle rouge-1: {sample_rouge1}, rouge-2: {sample_rouge2}, rouge-L: {sample_rougeLsum}")
        if len(args.gpuid) > 1:
            scorer.module.model.model.scoring_mode = True
        else:
            scorer.model.model.scoring_mode = True

    scorer.train()
    print(f"rouge-1: {rouge1}, rouge-2: {rouge2}, rouge-L: {rougeLsum}")

    if len(args.gpuid) > 1:
        dist.all_reduce(mle_loss, op=dist.reduce_op.SUM)
        mle_loss = mle_loss / len(args.gpuid)
    return loss, mle_loss.item()


def run(rank, args):
    base_setting(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = args.gpuid[rank]
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    if is_master:
        id = len(os.listdir("./cache")) + 2
        recorder = Recorder(id, args.log)
    EPS = 1e-10
    if args.is_pegasus:
        tok = PegasusTokenizer.from_pretrained(args.model_type)
    else:
        tok = BartTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp_bart, pad_token_id=tok.pad_token_id, is_test=False)
    collate_fn_val = partial(collate_mp_bart, pad_token_id=tok.pad_token_id, is_test=True)
    train_set = BartDataset(f"./{args.dataset}/{args.datatype}/train", args.model_type, maxlen=args.max_len, maxnum=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)
    val_set = BartDataset(f"./{args.dataset}/{args.datatype}/val", args.model_type, is_test=True, maxlen=512, is_sorted=False, maxnum=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	 train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
    	 val_set, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn_val, sampler=val_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    scorer = model.BartMixReRanker(model_path, tok.pad_token_id, is_pegasus=args.is_pegasus)
    if len(args.model_pt) > 0:
        scorer.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=f'cuda:{gpuid}'))
    if args.cuda:
        if len(args.gpuid) == 1:
            scorer = scorer.cuda()
        else:
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            scorer = nn.parallel.DistributedDataParallel(scorer.to(gpuid), [gpuid], find_unused_parameters=False)
    scorer.train()
    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    init_lr = args.lr / args.warmup_steps
    if args.single:
        init_lr = args.lr
    s_optimizer = optim.Adam(scorer.parameters(), lr=init_lr)
    if is_master:
        recorder.write_config(args, [scorer], __file__)
    minimum_ranking_loss = 100
    minimum_mle_loss = 1e5
    all_step_cnt = 0
    now = datetime.now()
    date = now.strftime("%y-%m-%d")
    if len(args.gpuid) > 1:
        if is_master:
            id = torch.FloatTensor([id]).to(gpuid)
        else:
            id = torch.zeros(1).to(gpuid)
        dist.all_reduce(id, op=dist.reduce_op.SUM)
        id = int(id.item())
    root_dir = f"./cache/{date}-{id}"
    # start training
    for epoch in range(args.epoch):
        s_optimizer.zero_grad()
        avg_ranking_loss = 0
        avg_mle_loss = 0
        step_cnt = 0
        epoch_step = 0
        avg_loss = 0
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, gpuid)
            step_cnt += 1
            output = scorer(batch["src_input_ids"], batch["candidate_ids"], args.normalize, args.score_mode, args.length_penalty, adding=args.adding)
            similarity, gold_similarity = output['score'], output['summary_score']
            similarity = similarity * args.scale
            gold_similarity = gold_similarity * args.scale
            ranking_loss = RankingLoss(similarity, gold_similarity, args.margin, args.gold_margin, args.gold_weight)
            probs = output["probs"]  # [bz, seq_len, word_num]
            probs = output["probs"][:, :-1]  # truncate last token
            gold = batch["candidate_ids"][:, 0, 1:]  # shift right
            mle_loss = mle_fn(probs.transpose(1, 2), gold)
            loss = args.rank_weight * ranking_loss + args.mle_weight * mle_loss
            loss = loss / args.accumulate_step
            avg_loss += loss.item()
            avg_mle_loss += mle_loss.item() / args.accumulate_step
            avg_ranking_loss += ranking_loss.item() / args.accumulate_step
            loss.backward()
            if step_cnt == args.accumulate_step:
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(scorer.parameters(), args.grad_norm)
                step_cnt = 0
                epoch_step += 1
                all_step_cnt += 1
                lr = args.lr
                if not args.single:
                    lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                    for param_group in s_optimizer.param_groups:
                        param_group['lr'] = lr
                s_optimizer.step()
                s_optimizer.zero_grad()
            if epoch_step % args.report_freq == 0 and step_cnt == 0 and is_master:
                print("id: %d"%id)
                print(f"similarity: {similarity[:, :10]}")
                if not args.no_gold:
                    print(f"gold similarity: {gold_similarity}")
                recorder.print("epoch: %d, batch: %d, avg loss: %.6f, avg ranking loss: %.6f, avg mle loss: %.6f"
                %(epoch+1, epoch_step, avg_loss / args.report_freq, avg_ranking_loss / args.report_freq, avg_mle_loss / args.report_freq))
                recorder.print(f"learning rate: {lr:.6f}")
                recorder.plot("loss", {"loss": avg_loss / args.report_freq}, all_step_cnt)
                recorder.print()
                avg_mle_loss, avg_ranking_loss, avg_loss = 0, 0, 0
            del similarity, gold_similarity, loss, mle_loss, ranking_loss, output, probs

            if all_step_cnt % 1000 == 0 and all_step_cnt != 0 and step_cnt == 0 and not args.single:
                loss, mle_loss = test(val_dataloader, scorer, args, gpuid, tok, root_dir, args.do_sample, rank)
                if loss < minimum_ranking_loss and is_master:
                    minimum_ranking_loss = loss
                    if is_mp:
                        recorder.save(scorer.module, "scorer.bin")
                    else:
                        recorder.save(scorer, "scorer.bin")
                    recorder.print("best ranking loss - epoch: %d, batch: %d"%(epoch, i / args.accumulate_step))
                if is_master:
                    recorder.print("val rouge: %.6f"%(1 - loss))

                if mle_loss < minimum_mle_loss and is_master:
                    minimum_mle_loss = mle_loss
                    if is_mp:
                        recorder.save(scorer.module, "scorer_mle.bin")
                    else:
                        recorder.save(scorer, "scorer_mle.bin")
                    recorder.print("best mle loss - epoch: %d, batch: %d"%(epoch, i / args.accumulate_step))
                if is_master:
                    recorder.print("mle val rouge: %.6f"%(1 - mle_loss))


def main(args):
    # set env
    if len(args.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args,), nprocs=len(args.gpuid), join=True)
    else:
        run(0, args)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpuid", nargs='+', type=int, default=0)
    parser.add_argument("-e", "--evaluate", action="store_true")
    parser.add_argument("-l", "--log", action="store_true")
    parser.add_argument("-p", "--port", type=int, default=12355)
    parser.add_argument("--model_pt", default="", type=str)
    args = parser.parse_args()
    if args.cuda is False:
        if args.evaluate:
            evaluation(args)
        else:
            main(args)
    else:
        if args.evaluate:
            with torch.cuda.device(args.gpuid[0]):
                evaluation(args)
        elif len(args.gpuid) == 1:
            with torch.cuda.device(args.gpuid[0]):
                main(args)
        else:
            main(args)
