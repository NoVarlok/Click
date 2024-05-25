import telebot
import logging
from importlib import import_module
import numpy as np

import torch
from torch import Tensor
from transformers.trainer_utils import set_seed
from accelerate import Accelerator

from utils.building_utils import boolean_string, build_model


class ModelParams:
    def __init__(self):
        self.model_name = 'blender'
        self.pretrained_model_path = '/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/experiments_contrast/combined_loss_embedding_pos_linear_prob_matching_copy_05/20.0'
        self.model_args = []


def cut_sequence_to_eos(seq, eos_token_id):
    ret = []
    for t in seq:
        if len(ret) > 0 and t == eos_token_id:
            break
        ret.append(t)
    return ret


def cut_label_to_golden(seq):
    ret = []
    for t in seq:
        if t == -100:
            if len(ret) == 0:
                continue
            else:
                break
        ret.append(t)
    return ret


def make_source(utterances, toker):
    utterances = [' ' + e.strip() for e in utterances]
    text = '  '.join(utterances) + toker.eos_token
    return text


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str, required=True)
    # parser.add_argument('--pretrained_model_path', type=str, required=True)
    # parser.add_argument('--model_args', type=str, nargs='+', default=[])
    # args = parser.parse_args()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)
    accelerator = Accelerator()

    toker, model = build_model(ModelParams(), checkpoint=None)
    model = accelerator.prepare(model)
    model.eval()

    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info('Number of parameter = {}'.format(total_params))

    eos_token_id = toker.eos_token_id
    generation_kwargs = {
        'max_new_tokens': 32,
        'min_length': 5,
        'do_sample': True,
        'temperature': 1,
        'top_k': 0,
        'top_p': 0.9,
        'num_beams': 1,
        'num_return_sequences': 25,
        'length_penalty': 1,
        'repetition_penalty': 1,
        'no_repeat_ngram_size': 0,
        'encoder_no_repeat_ngram_size': 0,
        'pad_token_id': eos_token_id,
        'eos_token_id': eos_token_id,
    }
    decode = lambda x: toker.decode(x, skip_special_tokens=False)
    collate_fn = getattr(import_module('collators.text2text'), 'collate_fn')
    bot = telebot.TeleBot("7188042487:AAFAeASZsxNve564fExpvjjFkN3lz-2xbjk")

    def generate(text):
        text = make_source(text, toker)
        text = collate_fn([{"source": text, "target": text,}],
                        toker,
                        max_input_length=128,
                        max_decoder_input_length=32,
                        infer=True)
        text['input_ids'] = text['input_ids'].cuda()
        text['attention_mask'] = text['attention_mask'].cuda()
        text.update(generation_kwargs)
        text.pop('references')
        with torch.no_grad():
            generations = model.generate(**text)
            generations = [cut_sequence_to_eos(each, eos_token_id) for each in generations.tolist()]
            generations  = [decode(g) for g in generations]
            for i, g in enumerate(generations):
                print(f'{i}) {g}')
        return generations[0]

    @bot.message_handler(content_types=['text'])
    def get_text_messages(message):
        print('message:')
        print(message.text)
        bot.send_message(message.from_user.id, generate(message.text))

    print('Telegram bot is running...')
    bot.polling(none_stop=True, interval=0)