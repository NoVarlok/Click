
from time import time
import json
from tqdm import tqdm
import numpy as np
from collections import Counter
from transformers import AutoTokenizer
from matplotlib import pyplot as plt

toker = AutoTokenizer.from_pretrained('/home/lyakhtin/repos/ctg/pretrained_models/blenderbot-90M', mask_token=None, use_fast=True)

MULTIPLE = 20
MAX_NEG_NUM = 5


##################################################################
# sentence embeddings
##################################################################
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
model = SentenceTransformer("all-mpnet-base-v2")


def is_negative(x):
    return x[0] > 0.5


def make_source(utterances):
    utterances = [' ' + e.strip() for e in utterances]
    text = '  '.join(utterances) + toker.eos_token
    return text


def make_target(utterance):
    text = toker.bos_token + ' ' + utterance.strip() + toker.eos_token
    return text


def remove_token(sentence):
    return sentence.replace(toker.bos_token, '').replace(toker.eos_token, '').strip()


st = time()
raw_data = [json.loads(e) for e in open(f"/home/lyakhtin/repos/ctg/datasets/click_checkpoints/data_bad/raw/train.txt")]
augmented_data = [[] for _ in range(len(raw_data))]
print('raw_data', time() - st)


losses = [json.loads(e)['loss'] for e in open(f'/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/blender/train/gen.txt')]
for idx in range(len(raw_data)):
    generation = raw_data[idx]['response']
    cls_label = raw_data[idx]['label']
    loss = losses[idx]
    if cls_label == 0:
        prediction = [1., 0.]
    else:
        prediction = [0., 1.]
    augmented_data[idx].append((generation, prediction, loss))


augmented_path = '/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/blender/train'
generations = [json.loads(e)['generation'] for e in open(f'{augmented_path}/gen.txt')]
predictions = [json.loads(e) for e in open(f'{augmented_path}/pred_list.txt')]
losses = [json.loads(e)['loss'] for e in open(f'{augmented_path}/loss_self_list.txt')]
assert len(generations) == len(raw_data)
assert len(predictions) % len(generations) == 0
assert MULTIPLE == len(predictions) // len(generations)
print('augmented_data', time() - st)

for idx in range(len(raw_data)):
    generation = generations[idx]
    if not isinstance(generation, list):
        generation = [generation]
    prediction = predictions[MULTIPLE * idx: MULTIPLE * (idx+1)]
    loss = losses[MULTIPLE * idx: MULTIPLE * (idx+1)]
    assert len(generation) == len(prediction)
    for x, y, z in zip(generation, prediction, loss):
        augmented_data[idx].append((x, y, z))
print('append', time() - st)


data = []
stats_binary = []
stats_num = []
fout = open('/home/lyakhtin/repos/ctg/datasets/click_checkpoints/data_bad/contrast_experiments/losses.txt', 'w')
for raw, augment in tqdm(zip(raw_data, augmented_data), total=len(raw_data), ncols=0):

    source = make_source(raw['context'])

    pos_targets = []
    pos_losses = []
    pos_probs = []
    neg_targets = []
    neg_losses = []
    neg_probs = []
    for idx in range(len(augment)):
        tmp_augment = augment[idx]

        if is_negative(tmp_augment[1]):
            neg_target = make_target(tmp_augment[0])
            neg_targets.append(neg_target)
            neg_losses.append(tmp_augment[2])
            neg_probs.append(tmp_augment[1][1])
        else:
            pos_target = make_target(tmp_augment[0])
            pos_targets.append(pos_target)
            pos_losses.append(tmp_augment[2])
            pos_probs.append(tmp_augment[1][1])

    if len(neg_targets) > 0:
        stats_binary.append(1)
        stats_num.append(len(neg_targets))
    else:
        stats_binary.append(0)

    if len(pos_targets) == 0:
        continue

    pos_generated_embeddings = [embedding for embedding in model.encode([remove_token(target) for target in pos_targets])]
    target_embedding = pos_generated_embeddings[0]
    filtered_pos_targets = []
    filtered_pos_embeddings = []
    for target, embedding, loss in zip(pos_targets, target_embeddings, pos_losses):
        if loss < 10:
            filtered_pos_targets.append(target)
            filtered_pos_embeddings.append(cosine(target_embeddings, embedding))

    if raw['label'] == 1:
        target = pos_targets[0]
    elif filtered_pos_targets:
        target = filtered_pos_targets[np.argmin(filtered_pos_distance)]
    else:
        target = toker.bos_token

    tmp_data = {
        'source': source,
        # 'target': pos_targets[0] if raw['label'] == 1 else toker.bos_token,
        'target': target,
        'pos_targets': [],
        'neg_targets': [],
    }

    ##################################################################
    # combined_loss_embedding_pos_linear_prob_matching
    ##################################################################
    def filter_data_by_loss(targets, losses, probs):
        LOSS_THRESHOLD = 100
        filtered_targets = []
        filtered_losses = []
        filtered_probs = []
        for target, loss, prob in zip(targets, losses, probs):
            if loss < LOSS_THRESHOLD:
                filtered_targets.append(target)
                filtered_losses.append(loss)
                filtered_probs.append(prob)
        return filtered_targets, filtered_losses, filtered_probs

    pos_targets, pos_losses, pos_probs = filter_data_by_loss(pos_targets, pos_losses, pos_probs)
    neg_targets, neg_losses, neg_probs = filter_data_by_loss(neg_targets, neg_losses, neg_probs)
    if len(pos_targets) == 0:
        continue

    pairs = sorted(zip(pos_targets, pos_losses), key=lambda x: x[1])
    pos_targets = [e[0] for e in pairs]
    pos_losses = [e[1] for e in pairs]

    # linear neg
    pairs = sorted(zip(neg_targets, neg_losses, neg_probs), key=lambda x: x[1])
    neg_targets = [e[0] for e in pairs]
    neg_losses = [e[1] for e in pairs]
    neg_probs = [e[2] for e in pairs]

    pos_embeddings = [embedding for embedding in model.encode([remove_token(target) for target in pos_targets])]
    neg_embeddings = [embedding for embedding in model.encode([remove_token(target) for target in neg_targets])]

    # linear
    pos_prob_weights = [1.5 - prob for prob in pos_probs]
    neg_prob_weights = [0.5 + prob for prob in neg_probs]


    for neg_target, neg_loss, neg_embedding, neg_prob_weigth in zip(neg_targets[:MAX_NEG_NUM], neg_losses[:MAX_NEG_NUM], neg_embeddings[:MAX_NEG_NUM], neg_prob_weights[:MAX_NEG_NUM]):
        distances = [cosine(neg_embedding, pos_embedding) for pos_embedding in pos_embeddings]
        pos_combined_loss = [distance * loss * prob_weight for distance, loss, prob_weight in zip(distances, pos_losses, pos_prob_weights)]
        idx_sorted = np.argsort(pos_combined_loss)
        pos_targets_ = [pos_targets[idx] for idx in idx_sorted]
        pos_losses_ = [pos_losses[idx] for idx in idx_sorted]
        for pos_target, pos_loss in zip(pos_targets_, pos_losses_):
            if pos_loss > neg_loss:
                break
            else:
                pos_target = pos_targets_[-1]
        tmp_data['pos_targets'].append(pos_target)
        tmp_data['neg_targets'].append(neg_target)

    data.append(tmp_data)

print('data', time() - st)


print(len(data))
with open('/home/lyakhtin/repos/ctg/datasets/click_checkpoints/data_bad/contrast_experiments/train.txt', 'w') as f:
    for d in data:
        f.write(json.dumps(d) + '\n')
with open('/home/lyakhtin/repos/ctg/datasets/click_checkpoints/data_bad/contrast_experiments/samples.txt', 'w') as f:
    for d in data[:50]:
        f.write(json.dumps(d) + '\n')
print('save', time() - st)


exit()
print(np.mean(stats_binary), np.mean(stats_num))
print(Counter(stats_num)[20])
plt.figure()
plt.hist(stats_num)
plt.tight_layout()
plt.savefig('./stats_num.png', dpi=300)
