import torch
from model_arithmetic import ModelArithmetic, Evaluation, PromptedLLM, enable_logging
from transformers import set_seed
import pandas as pd
from formulas_toxicity import *
from loguru import logger
import os
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
import json
from tqdm import tqdm


class BasicDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, i):
        return self.data_list[i]

    def __len__(self):
        return len(self.data_list)


class BatchDataLoader(DataLoader):
    def __init__(self,
            data_list=None, data_path=None, batch_size=None,
            collate_fn=None, shuffle=True, num_workers=16,
        ):
        if data_list is None:
            data_list = [json.loads(e) for e in open(data_path)]
        dataset = BasicDataset(data_list)
        basic_sampler = RandomSampler if shuffle else SequentialSampler
        sampler = BatchSampler(basic_sampler(dataset), batch_size=batch_size, drop_last=False)
        super().__init__(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)


def collate_fn(data):
    def process(values, name):
        processed_values = [value[name].replace('__start__', '').replace('__end__', '').strip() for value in values]
        return processed_values
    
    source = process(data, 'source')
    target = process(data, 'target')
    return {'source': source, 'target': target}


if __name__ == '__main__':
    model = 'EleutherAI/Pythia-70m'
    # model = "facebook/blenderbot-90M"
    default_model = None
    batch_size = 4
    num_return_sequences = 25

    formula = combo(0.04, -0.0, -0.96, c_model="/home/lyakhtin/repos/ctg/datasets/lm_arithmetic_checkpoints/finetune/toxicity_classifier", m_model=model)
    set_seed(42)
    if isinstance(formula, tuple):
        retroactive = [formula[1]]
        formula = formula[0]
    else:
        retroactive = []
    arithmetic = ModelArithmetic(formula, default_model=default_model, retroactive_operators=retroactive)
    arithmetic.save_pretrained("/home/lyakhtin/repos/ctg/datasets/lm_arithmetic_checkpoints/finetune/arithmetic")

    for split in ['valid']:
        print('Split:', split)
        infer_data_path = f'/home/lyakhtin/repos/ctg/datasets/click_checkpoints/data_bad/blender/{split}.txt'
        output_path = f'/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/model_arithmetics/{split}/gen.txt'
        # collator_name = 'text2text'
        # collate_fn = getattr(import_module('collators.' + collator_name), 'collate_fn')
        dataset = BatchDataLoader(
            data_path=infer_data_path,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )
        # for data in dataset:
        #     print(data)
        generated_samples = 0
        if os.path.exists(output_path):
            with open(output_path, 'r') as fin:
                lines = [x.strip() for x in fin]
                generated_samples = len(lines)
                if generated_samples % batch_size != 0:
                    generated_samples -= (generated_samples % batch_size)
                    lines = lines[:generated_samples]
        else:
            lines = []
        batch_idx = 0
        with open(output_path, 'w') as fout:
            for line in lines:
                print(line, file=fout)
            fout.flush()
            for batch in tqdm(dataset):
                if batch_idx * batch_size < generated_samples:
                    batch_idx += 1
                    continue
                # for source in batch['source']:
                source = batch['source']
                result = arithmetic.generate_text(source, max_length=32, num_return_sequences=num_return_sequences, do_speculation=False)
                for i in range(batch_size):
                    json_result = {"generation": result[i * num_return_sequences: (i + 1) * num_return_sequences]}
                    json.dump(json_result, fout)
                    fout.write('\n')
                fout.flush()
                batch_idx += 1
