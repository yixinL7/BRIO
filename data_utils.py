from torch.utils.data import Dataset
import os
import json
import torch
from transformers import BartTokenizer, PegasusTokenizer


def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data":
            batch[n] = batch[n].to(gpuid)


class BrioDataset(Dataset):
    def __init__(self, fdir, model_type, max_len=-1, is_test=False, total_len=512, is_sorted=True, max_num=-1, is_untok=True, is_pegasus=False, num=-1):
        """ data format: article, abstract, [(candidiate_i, score_i)] """
        self.isdir = os.path.isdir(fdir)
        if self.isdir:
            self.fdir = fdir
            if num > 0:
                self.num = min(len(os.listdir(fdir)), num)
            else:
                self.num = len(os.listdir(fdir))
        else:
            with open(fdir) as f:
                self.files = [x.strip() for x in f]
            if num > 0:
                self.num = min(len(self.files), num)
            else:
                self.num = len(self.files)
        if is_pegasus:
            self.tok = PegasusTokenizer.from_pretrained(model_type, verbose=False)
        else:
            self.tok = BartTokenizer.from_pretrained(model_type, verbose=False)
        self.maxlen = max_len
        self.is_test = is_test
        self.total_len = total_len
        self.sorted = is_sorted
        self.maxnum = max_num
        self.is_untok = is_untok
        self.is_pegasus = is_pegasus

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self.isdir:
            with open(os.path.join(self.fdir, "%d.json"%idx), "r") as f:
                data = json.load(f)
        else:
            with open(self.files[idx]) as f:
                data = json.load(f)
        if self.is_untok:
            article = data["article_untok"]
        else:
            article = data["article"]
        src_txt = " ".join(article)
        src = self.tok.batch_encode_plus([src_txt], max_length=self.total_len, return_tensors="pt", pad_to_max_length=False, truncation=True)
        src_input_ids = src["input_ids"]
        src_input_ids = src_input_ids.squeeze(0)
        if self.is_untok:
            abstract = data["abstract_untok"]
        else:
            abstract = data["abstract"]
        if self.maxnum > 0:
            candidates = data["candidates_untok"][:self.maxnum]
            _candidates = data["candidates"][:self.maxnum]
            data["candidates"] = _candidates
        if self.sorted:
            candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
            _candidates = sorted(_candidates, key=lambda x:x[1], reverse=True)
            data["candidates"] = _candidates
        if not self.is_untok:
            candidates = _candidates
        cand_txt = [" ".join(abstract)] + [" ".join(x[0]) for x in candidates]
        cand = self.tok.batch_encode_plus(cand_txt, max_length=self.maxlen, return_tensors="pt", pad_to_max_length=False, truncation=True, padding=True)
        candidate_ids = cand["input_ids"]
        if self.is_pegasus:
            # add start token
            _candidate_ids = candidate_ids.new_zeros(candidate_ids.size(0), candidate_ids.size(1) + 1)
            _candidate_ids[:, 1:] = candidate_ids.clone()
            _candidate_ids[:, 0] = self.tok.pad_token_id
            candidate_ids = _candidate_ids
        result = {
            "src_input_ids": src_input_ids, 
            "candidate_ids": candidate_ids,
            }
        if self.is_test:
            result["data"] = data
        return result


def collate_mp_brio(batch, pad_token_id, is_test=False):
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result

    src_input_ids = pad([x["src_input_ids"] for x in batch])
    candidate_ids = [x["candidate_ids"] for x in batch]
    max_len = max([max([len(c) for c in x]) for x in candidate_ids])
    candidate_ids = [pad(x, max_len) for x in candidate_ids]
    candidate_ids = torch.stack(candidate_ids)
    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "candidate_ids": candidate_ids,
        }
    if is_test:
        result["data"] = data
    return result