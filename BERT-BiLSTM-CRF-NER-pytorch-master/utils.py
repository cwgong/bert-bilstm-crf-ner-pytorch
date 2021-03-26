import io
import logging
import os
import sys
import torch
import pickle
import copy
import math
from torch.utils.data import TensorDataset
from tqdm import tqdm
import re
logger = logging.getLogger(__name__)
ts=['\ue456','\ue246','\ue312','\ue292','\ue3da','\ue2c7','\ue0ec', 
    '\ue02e','\ue264','\ue060','\ue093','\ue0be', '\ue0ce', '\ue431',
    '\ue0bc','\ue0e9','\ue0df','\ue002','\ue027','\ue000','\ue456',
    '\ue0de','\ue787','\ue207','\ue383','\ue2be', '\ue1e6','\ue468',
    '\ue46a', '\ue0ed','\ue1fb','\ue49a','\ue1d8','\ue800','\ue262',
    '\ue034','\ue0d6','\ue13f','\ue132', '\ue432','\ue0ee'
    ]
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text 
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ori_tokens):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ori_tokens = ori_tokens


class NerProcessor(object):
    def read_data(self, input_file):
        """Reads a BIO data."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            words = []
            labels = []
            
            for line in f.readlines():   
                contends = line.strip()
                tokens = line.strip().split("\t")
                if len(tokens) == 2:
                    s=re.sub(r'\\u.{4}','',tokens[0].__repr__())
                    s=eval(s)
                    if not len(s):
                        tokens[0]='#'                    
                    #if 'ue' in tokens[0]:
                        #tokens[0]='#'
                    #else:
                        #if tokens[0] in ts:
                            #tokens[0]='#'                           
                    words.append(tokens[0])
                    labels.append(tokens[1])
                elif len(tokens) == 3:                
                    s=re.sub(r'\\u.{4}','',tokens[0].__repr__())
                    s=eval(s)
                    if not len(s):
                        tokens[0]='#'
                    #if 'ue' in tokens[0]:
                        #tokens[0]='#' 
                    #else:
                        #if tokens[0] in ts:
                            #tokens[0]='#'   
                    words.append(tokens[0])
                    labels.append(tokens[1])                    
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
            
            return lines
    
    def get_labels(self, args):
        labels = set()
        if os.path.exists(os.path.join(args.output_dir, "label_list.pkl")):
            logger.info(f"loading labels info from {args.output_dir}")
            with open(os.path.join(args.output_dir, "label_list.pkl"), "rb") as f:
                labels = pickle.load(f)
        else:
            # get labels from train data
            logger.info(f"loading labels info from train file and dump in {args.output_dir}")
            with open(args.train_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    tokens = line.strip().split("\t")

                    if len(tokens) == 2:
                        labels.add(tokens[1])
                    elif len(tokens) == 3:
                        labels.add(tokens[1])
                    

            if len(labels) > 0:
                with open(os.path.join(args.output_dir, "label_list.pkl"), "wb") as f:
                    pickle.dump(labels, f)
            else:
                logger.info("loading error and return the default labels B,I,O")
                labels = {"O", "B", "I"}
        
        return labels 

    def get_examples(self, input_file):
        examples = []
        
        lines = self.read_data(input_file)

        for i, line in enumerate(lines):
            guid = str(i)
            text = line[1]
            label = line[0]

            examples.append(InputExample(guid=guid, text=text, label=label))
        
        return examples
    def read_pred_data(self, input_file):
        """Reads a BIO data."""
        file_paths = []
        for name in os.listdir(input_file):
            path = os.path.join(input_file,name)
            if  os.path.isfile(path):
                file_paths.append(path)   
        lines = []
        for file_path in file_paths:
            file_name = file_path.split("/")[-1]
            title = file_name.split(".")[0]            
            with open(file_path, "r", encoding="utf-8") as f:

                for line in f.readlines():  
                    if line:
                        text=line
                        if len(line)<512:
                            print(line)
                            word=line.replace(' ','#')
                            words=[]
                            for w in word:
                                s=re.sub(r'\\u.{4}','',w.__repr__())
                                s=eval(s)
                                if not len(s):
                                    w='#'  
                                words.append(w)
                                
                            label=[title]*len(words)
                            lines.append([' '.join(label), ' '.join(words)])
                        else:
                            if title in ['1390']:
                                print(text)
                            news_contents=[]
                            content=copy.deepcopy(text)
                            strlen=len(content)
                            cnt=math.ceil(strlen/400) 
                            tcnt=copy.deepcopy(cnt)
                            per_cnt=math.ceil(strlen/cnt)  
                            while cnt:
                                if per_cnt*(tcnt-cnt+1)<strlen and content[per_cnt*(tcnt-cnt+1)] in [' ','。','！','？']:
                                    if per_cnt*(tcnt-cnt)==0:
                                        tc=content[:per_cnt*(tcnt-cnt+1)+1]
                                        assert len(tc)<512
                                        news_contents.append(content[:per_cnt*(tcnt-cnt+1)+1])
                                    else:
                                        tc=content[per_cnt*(tcnt-cnt)+1:per_cnt*(tcnt-cnt+1)+1]
                                        assert len(tc)<512
                                        news_contents.append(content[per_cnt*(tcnt-cnt)+1:per_cnt*(tcnt-cnt+1)+1])
                                    cnt=cnt-1
                                elif per_cnt*(tcnt-cnt+1)>=strlen:
                                    tc=content[per_cnt*(tcnt-cnt)+1:]
                                    assert len(tc)<512                                    
                                    news_contents.append(content[per_cnt*(tcnt-cnt)+1:])
                                    cnt=cnt-1                
                                else:
                                    startpos=(tcnt-cnt)*per_cnt+1
                                    if startpos==1:
                                        while content[per_cnt] not in [' ','。','！','？']:
                                            per_cnt+=1
                                        tc=content[:per_cnt*(tcnt-cnt+1)+1]
                                        assert len(tc)<512                                            
                                        news_contents.append(content[:per_cnt*(tcnt-cnt+1)+1])
                                        cnt=cnt-1                    
                                    else:
                                        if per_cnt*(tcnt-cnt+1)>=strlen:
                                            tc=content[startpos:]
                                            assert len(tc)<512                                               
                                            news_contents.append(content[startpos:])
                                            cnt=cnt-1                        
                                        else:
                                            while content[per_cnt*(tcnt-cnt+1)] not in  [' ','。','！','？']:
                                                per_cnt+=1
                                            tc=content[startpos:per_cnt*(tcnt-cnt+1)+1]
                                            assert len(tc)<512                                               
                                            news_contents.append(content[startpos:per_cnt*(tcnt-cnt+1)+1])
                                            cnt=cnt-1  
                            tmp=''
                            for word in news_contents:
                                word=word.replace(' ','#')
                                #tmp+=word
                                words=[]
                                for w in word:
                                    s=re.sub(r'\\u.{4}','',w.__repr__())
                                    s=eval(s)
                                    if not len(s):
                                        w='#'  
                                    words.append(w)
                                tmp+= ''.join(words)
                                label=[title]*len(words)
                                lines.append([' '.join(label), ' '.join(words)])  
                            assert len(tmp)==len(text)
        return lines        
    def get_pred_examples(self,input_file):
        examples = []
        
        lines = self.read_pred_data(input_file)

        for i, line in enumerate(lines):
            guid = str(i)
            text = line[1]
            label = line[0]

            examples.append(InputExample(guid=guid, text=text, label=label))
        
        return examples        


def convert_examples_to_features(args, examples, label_list, max_seq_length, tokenizer):

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples"):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        textlist = example.text.split(" ")
        labellist = example.label.split(" ")
        assert len(textlist) == len(labellist)
        tokens = []
        labels = []
        ori_tokens = []

        for i, word in enumerate(textlist):
            # 防止wordPiece情况出现，不过貌似不会
            
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            ori_tokens.append(word)

            # 单个字符不会出现wordPiece
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    if label_1 == "O":
                        labels.append("O")
                    else:
                        labels.append("I")
            
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
            labels = labels[0:(max_seq_length - 2)]
            ori_tokens = ori_tokens[0:(max_seq_length - 2)]

        ori_tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
        
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["O"])

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if not label_map.get(labels[i]):
                _id=label_map.get('O')
            else:
                _id=label_map.get(labels[i])
            
            label_ids.append(_id)

        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["O"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)   
        
        input_mask = [1] * len(input_ids)

        assert len(ori_tokens) == len(ntokens), f"{len(ori_tokens)}, {len(ntokens)}, {ori_tokens}"

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(0)
            ntokens.append("**NULL**")

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in ntokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        # if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        #     with open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
        #         pickle.dump(label_map, w)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              ori_tokens=ori_tokens))

    return features


def get_Dataset(args, processor, tokenizer, mode="train"):
    if mode == "train":
        filepath = args.train_file
    elif mode == "eval":
        filepath = args.eval_file
    elif mode == "test":
        filepath = args.test_file
    elif mode == "test":
        filepath = args.inference_file    
    else:
        raise ValueError("mode must be one of train, eval, or test")

    examples = processor.get_examples(filepath)
    label_list = args.label_list

    features = convert_examples_to_features(
        args, examples, label_list, args.max_seq_length, tokenizer
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return examples, features, data

def get_Pred_Dataset(args, processor, tokenizer, mode="inference"):
    if mode == "train":
        filepath = args.train_file
    elif mode == "eval":
        filepath = args.eval_file
    elif mode == "test":
        filepath = args.test_file
    elif mode == "inference":
        filepath = args.inference_file    
    else:
        raise ValueError("mode must be one of train, eval, or test")

    examples = processor.get_pred_examples(filepath)
    label_list = args.label_list

    features = convert_examples_to_features(
        args, examples, label_list, args.max_seq_length, tokenizer
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return examples, features, data