import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
# from transformers import BertTokenizer
from transformers import AutoTokenizer
from create_dataset import MOSI, MOSEI, UR_FUNNY, PAD

# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', mirror='https://huggingface.co/mirrors')

...
_TOKENIZERS = {}


def get_tokenizer(name):
    """Load each tokenizer once per process, for whichever encoder the run uses."""
    if name not in _TOKENIZERS:
        _TOKENIZERS[name] = AutoTokenizer.from_pretrained(name)
    return _TOKENIZERS[name]


class MSADataset(Dataset):
    def __init__(self, config):
        # Fetch dataset
        if config.data.lower() == "mosi":
            dataset = MOSI(config)
        elif config.data.lower() == "mosei":
            dataset = MOSEI(config)
        elif config.data.lower() == "ur_funny": # Jack Add From MISA Code
            dataset = UR_FUNNY(config)
        else:
            print("Dataset not defined correctly")
            exit()

        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        self.len = len(self.data)
        # self.label = np.abs(np.array(self.data)[:, 1])
        self.label = np.abs(np.array(self.data, dtype=object)[:, 1])

        config.visual_size = self.data[0][0][1].shape[1]
        config.acoustic_size = self.data[0][0][2].shape[1]

        config.word2id = self.word2id
        config.pretrained_emb = self.pretrained_emb

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def get_loader(config, shuffle=True):
    """Load DataLoader of given DialogDataset"""
    dataset = MSADataset(config)
    bert_tokenizer = get_tokenizer(config.text_encoder)
    print(config.mode)
    config.data_len = len(dataset)

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things

        if config.task == 'emotion':
            labels = torch.cat([torch.from_numpy(sample[1][:, 1:7]) for sample in batch], dim=0)  # 6 emotion columns -> [B, 6]
        else:
            labels = torch.cat([torch.from_numpy(sample[1][:, 0:1]) for sample in batch], dim=0)  # sentiment column -> [B, 1]
        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD) # UR_FUUNY: torch.Size([46, 32, 300])
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch]) # ([46, 32, 75]) # Jack Change FloatTensor -> LongTensor
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch]) # ([46, 32, 81]) # Jack Change

        ## BERT-based features input prep

        SENT_LEN = sentences.size(0)
        # Create bert indices using tokenizer

        # UR_FUNNY 不需要进行bert特征提取
        try:
            bert_details = []
            for sample in batch:
                text = " ".join(sample[0][3]) # len(text)= 179 len(sample[0][3])=39 ['and', 'um', 'i', 'have', 'to', 'admit', 'i', 'was', 'watching', 'this', 'i', 'put', 'it', 'on', ...]
                # encoded_bert_sent = bert_tokenizer.encode_plus(
                #     text, max_length=SENT_LEN + 2, add_special_tokens=True, pad_to_max_length=True)
                # encoded_bert_sent = bert_tokenizer(text, max_length=SENT_LEN + 2, add_special_tokens=True,
                #     padding='max_length', truncation=True)
                encoded_bert_sent = bert_tokenizer(text, max_length=min(2 * SENT_LEN + 2, 512), add_special_tokens=True, padding='max_length', truncation=True)
                bert_details.append(encoded_bert_sent)
                

            # Bert things are batch_first
            bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details]) # bert_sentences.shape=([64, 41]) 堆叠(64): len(41): [101, 1045, 16755, 2009, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]
            # bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details]) # bert_sentence_types.shape=([64, 41]) 堆叠(64): len(41): [0, 0 ...]
            bert_sentence_types = torch.LongTensor([sample.get("token_type_ids", [0] * len(sample["input_ids"])) for sample in bert_details]) # roberta returns no token_type_ids -> substitute zeros
            bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details]) # bert_sentence_att_mask.shape=([64, 41]) 堆叠(64): len(41): [1, 0 ...]
        except: # 将V1 版本中提取的特征直接sum好赋值过来即可 使用随机mask会出错,acc为1, loss为nan!
            labels = labels.float() # UR_FUNNY 要将label也变为 'torch.FloatTensor'
            sum_sentences = torch.sum(sentences, dim=0).float() # ([64, 32, 300]) -> [32, 300] 消除第一个特征维度 64 保留对齐的300和32 bathsize 维度
            bert_sentences = (sum_sentences-torch.min(sum_sentences))/(torch.max(sum_sentences)-torch.min(sum_sentences)) # 将矩阵归一化到 0-1 之间 [32, 300]
            # 生成全一矩阵
            shape  = (bert_sentences.size(0), bert_sentences.size(1))
            bert_sentence_types = torch.ones(shape)
            bert_sentence_att_mask = torch.ones(shape)
            # # 进行bert获取
            # sum_result = torch.sum(sentences, dim=2, keepdim=True).squeeze(2).permute(1, 0) # torch.Size([46, 32, 300]) -> [32, 46]
            # bert_sentences = torch.clamp(sum_result, min=0).long()
            # # bert_sentences = torch.sum(sentences, dim=2, keepdim=True).squeeze(2).permute(1, 0).long()
            # # 随机mask
            # shape  = (bert_sentences.size(0), bert_sentences.size(1))
            # # 设置第一个矩阵的元素为0或1，其中0的占比为90%
            # torch.manual_seed(0)
            # threshold = 0.9
            # random_tensor = torch.randint(low=0, high=2, size=shape)
            # bert_sentence_types = (random_tensor >= threshold).int().to(torch.long)
            # # 设置第二个矩阵的元素为0或1，其中0的占比为10%
            # torch.manual_seed(1)
            # threshold = 0.1
            # random_tensor = torch.randint(low=0, high=2, size=shape)
            # bert_sentence_att_mask = (random_tensor >= threshold).int().to(torch.long)
            '''
            # ([64, 768])
            [[-0.0042, -0.2269,  0.2518,  ..., -0.3572,  0.0630, -0.1368],
            [ 0.0691,  0.0389,  0.4117,  ..., -0.2666,  0.3193, -0.0698],
            [-0.2460, -0.2926,  0.0414,  ..., -0.2327,  0.5835,  0.1047],
            ..., 
            [ 0.2548, -0.0438, -0.2142,  ..., -0.0204,  0.3280,  0.2674]],
            device='cuda:0', grad_fn=<DivBackward0>)               
            '''
        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
        
        # segment IDs (e.g. '-3g5yACwYnA_4') identify each clip, so saved predictions can be matched to CMU-MOSEI
        segment_ids = [sample[2] for sample in batch]

        return visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask, segment_ids

    # data_loader = DataLoader(
    #     dataset=dataset,
    #     batch_size=config.batch_size,
    #     shuffle=shuffle,
    #     collate_fn=collate_fn)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=shuffle)

    return data_loader
