import argparse
import glob
import logging
import os
import sys

import easydict
import librosa
import numpy as np
import requests
import torch
import tqdm
from sklearn import multiclass, svm

from dataloader import DataLoader
from models import FullModel


def init_logger(log_path, mode='w', stdout=True):
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'

    os.makedirs(os.path.split(log_path)[0], exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format=fmt, filename=log_path, filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        
def get_args():
    ## TODO: ここらへんを修正
    parser = argparse.ArgumentParser(description='研究用：音素に対して時間領域で処理して話者埋め込みを求めるやつ')
    parser.add_argument('--gpu', default='-1', type=str, metavar='N', help='GPU番号')
    parser.add_argument('--log-path', default='/tmp/vc-03.log', type=str, metavar='N', help='ログファイルのパス')
    parser.add_argument('--load-weights-path', type=str, metavar='N', help='読み込み元のウェイトファイルのパス')
    parser.add_argument('--weights-path', default='/tmp/vc-alpha.pth', type=str, metavar='N', help='保存先のウェイトファイルのパス')
    parser.add_argument('--nphonemes-path', default='resource/jvs_ver1_nphonemes_%(condition)s.txt', type=str, metavar='N', help='音素長データのパス')
    parser.add_argument('--patience', default=8, type=int, metavar='N', help='Early Stoppingまでの回数')
    parser.add_argument('--dataset-path', default='resource/jvs_ver1_phonemes/jvs%(person)03d/VOICEACTRESS100_%(voice)03d_%(deform_type)s.npz', type=str, metavar='N', help='データセットのパス')
    parser.add_argument('--batch-length-person', default=16, type=int, metavar='N', help='各バッチの話者数')
    parser.add_argument('--batch-length-phoneme', default=32, type=int, metavar='N', help='各バッチの音素数')
    parser.add_argument('--phonemes-length', default=32, type=int, metavar='N', help='音素の時間長')
    
    parser.add_argument('-d', '--deform-type', default='stretch', type=str, metavar='N', help='変形の種類')

    parser.add_argument('-sr', '--sampling-rate', default=24000, type=int, metavar='N', help='サンプリング周波数')
    parser.add_argument('--nfft', default=1024, type=int, metavar='N', help='STFTのウィンドウ幅（通常はPyWorldに依存）')
    parser.add_argument('--nhop', default=120, type=int, metavar='N', help='STFTのシフト幅（通常はPyWorldに依存）')

    # parser.add_argument('-xt', '--person-train-size', default=2048, type=int, metavar='N', help='学習に使用する音素数')
    # parser.add_argument('-yt', '--phoneme-train-size', default=64, type=int, metavar='N', help='学習に使用する話者数')
    # parser.add_argument('-xv', '--person-valid-size', default=512, type=int, metavar='N', help='検証に使用する音素数')
    # parser.add_argument('-yv', '--phoneme-valid-size', default=16, type=int, metavar='N', help='検証に使用する話者数')
    # parser.add_argument('-xe', '--person-eval-size', default=512, type=int, metavar='N', help='評価に使用する音素数')
    # parser.add_argument('-ye', '--phoneme-eval-size', default=16, type=int, metavar='N', help='評価に使用する話者数')
    
    parser.add_argument('-bx', '--person-batch-size', default=16, type=int, metavar='N', help='バッチ内の音素数')
    parser.add_argument('-by', '--phoneme-batch-size', default=16, type=int, metavar='N', help='バッチ内の話者数')
    
    args = vars(parser.parse_args(sys.argv[1:]))
    return args

def main(cfg):
    person_no_list = [8, 16, 17, 21, 23, 29, 35, 37, 42, 46, 47, 50, 54, 58, 59, 73, 88, 97]
    voice_no_list  = [5, 18, 24, 40, 42, 44, 46, 55, 56, 59, 60, 61, 63, 65, 71, 73, 75, 81, 84, 85, 87, 93, 94, 98]

    batch_size = (cfg.batch_length_person, cfg.batch_length_phoneme)

    known_person_list   = list(filter(lambda x:x not in person_no_list, np.arange(19)))
    unknown_person_list = list(filter(lambda x:x not in person_no_list, np.arange(19, 40)))
    train_voice_list    = list(filter(lambda x:x not in voice_no_list, np.arange(81)))
    check_voice_list    = list(filter(lambda x:x not in voice_no_list, np.arange(81, 95)))

    model = FullModel(1, cfg.nfft // 2, len(known_person_list)).to('cuda')
    if cfg.load_weights_path:
        load_weights(model, cfg.load_weights_path)

    train_loader = DataLoader(known_person_list, train_voice_list, batch_size, cfg.nphonemes_path, cfg.dataset_path, cfg.deform_type, cfg.phonemes_length)
    valid_loader = DataLoader(known_person_list, check_voice_list, batch_size, cfg.nphonemes_path, cfg.dataset_path, cfg.deform_type, cfg.phonemes_length)
    history = learn(model, (train_loader, valid_loader), cfg.weights_path, leaning_rate=1e-4, patience=cfg.patience)
    logging.info('History:\n' + history)

    known_train_loader   = DataLoader(known_person_list, train_voice_list, batch_size, cfg.nphonemes_path, cfg.dataset_path, cfg.deform_type, cfg.phonemes_length, onehot_mode=False)
    known_eval_loader    = DataLoader(known_person_list, check_voice_list, batch_size, cfg.nphonemes_path, cfg.dataset_path, cfg.deform_type, cfg.phonemes_length, onehot_mode=False)
    unknown_train_loader = DataLoader(unknown_person_list, train_voice_list, batch_size, cfg.nphonemes_path, cfg.dataset_path, cfg.deform_type, cfg.phonemes_length, onehot_mode=False)
    unknown_eval_loader  = DataLoader(unknown_person_list, check_voice_list, batch_size, cfg.nphonemes_path, cfg.dataset_path, cfg.deform_type, cfg.phonemes_length, onehot_mode=False)
    known_train_embed_pred, known_train_embed_true     = predict(model.embed, known_train_loader)
    known_eval_embed_pred, known_eval_embed_true       = predict(model.embed, known_eval_loader)
    unknown_train_embed_pred, unknown_train_embed_true = predict(model.embed, unknown_train_loader)
    unknown_eval_embed_pred, unknown_eval_embed_true   = predict(model.embed, unknown_eval_loader)
    known_svm_confusion_matrix   = calc_svm_matrix(known_train_embed_pred, known_train_embed_true, known_eval_embed_pred, known_eval_embed_true)
    unknown_svm_confusion_matrix = calc_svm_matrix(unknown_train_embed_pred, unknown_train_embed_true, unknown_eval_embed_pred, unknown_eval_embed_true)
    known_svm_acc_rate   = np.trace(known_svm_confusion_matrix).astype(np.int)
    unknown_svm_acc_rate = np.trace(unknown_svm_confusion_matrix).astype(np.int)
    logging.info(f'Known accuracy: {known_svm_acc_rate / len(known_eval_embed_true)} ({known_svm_acc_rate}/{len(known_eval_embed_true)})')
    logging.info(f'Unknown accuracy: {unknown_svm_acc_rate / len(unknown_eval_embed_true)} ({unknown_svm_acc_rate}/{len(unknown_eval_embed_true)})')

def load_weights(model, weights_path):
    existing_weights_paths = sorted(glob.glob(weights_path))
    if len(existing_weights_paths) == 0:
        return

    model.load_state_dict(torch.load(existing_weights_paths[-1]))

def learn(model, loaders, weights_path, leaning_rate=1e-4, patience=8):

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=leaning_rate)

    best_loss = np.Inf
    wait = 0
    history = { key: list() for key in ['train_loss', 'train_acc', 'valid_loss', 'valid_acc'] }
    for epoch in range(256):
        logging.info(f'[Epoch {epoch}]')

        train_loss, train_acc = train(model, loaders[0], optimizer, criterion)
        logging.info(f'Train loss {train_loss}, acc {100 * train_acc} %')

        valid_loss, valid_acc = valid(model, loaders[1], criterion)
        logging.info(f'Valid loss {valid_loss}, acc {100 * valid_acc} %')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)

        if valid_loss < best_loss:
            wait = 0
            best_loss = valid_loss
            logging.info(f'val_loss improved.')
            torch.save(model.state_dict(), weights_path)
        else:
            wait += 1
            logging.info(f'val_loss did not improve. {wait}/{patience}')
            if wait >= patience:
                logging.info(f'Early stopping.')
                model.load_state_dict(torch.load(weights_path))
                break

    return history

def train(model, loader, optimizer, criterion):
    model.train()

    train_loss = 0
    train_acc  = 0
    with tqdm.tqdm(loader, bar_format='{l_bar}{bar:24}| [{elapsed}<{remaining}{postfix}]') as bar:
        for idx, batch in enumerate(bar):
            data, true = batch
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc  += pred.argmax(dim=1).eq(true).sum().item()
            bar.set_postfix({'loss': '%.4f' % (train_loss / (idx + 1)), 'acc': '%.2f %%' % (100 * train_acc / ((idx + 1) * len(true)))})

    train_loss /= len(loader)
    train_acc  /= len(loader)
    return train_loss, train_acc

def valid(model, loader, criterion):
    model.eval()

    valid_loss = 0
    valid_acc  = 0
    for batch in loader:
        with torch.no_grad():
            data, true = batch
            pred = model(data)
            loss = criterion(pred, true)

            valid_loss += loss.item()
            valid_acc  += pred.argmax(dim=1).eq(true).sum().item()

    valid_loss /= len(loader)
    valid_acc  /= len(loader)
    return valid_loss, valid_acc

def predict(model, loader):
    '''
    Returns
    -------
    pred : [nbatch * batch_length, embed_dims]
    true : [nbatch * batch_length]
    '''
    model.eval()

    pred_list = list()
    true_list = list()
    for data, true in loader:
        with torch.no_grad():
            pred_data = model(data)
        pred_list.append(pred_data.to('cpu').detach().numpy().copy())
        true_list.append(true.to('cpu').detach().numpy().copy())
    pred = np.concatenate(pred_list)
    true = np.concatenate(true_list)
    return pred, true

def calc_confusion_matrix(pred, true, nclasses):
    confusion_matrix = np.zeros((nclasses, nclasses))
    for eval_true_label, pred_label in zip(true, pred):
        confusion_matrix[eval_true_label][pred_label] += 1
    return confusion_matrix

def calc_svm_matrix(train_data, train_true, eval_data, eval_true, nclasses):
    svc = svm.SVC(C=1., kernel='rbf', gamma=0.01, shrinking=False, verbose=False)
    classifier = multiclass.OneVsRestClassifier(svc)
    classifier.fit(train_data, train_true)

    eval_pred = classifier.predict(eval_data)
    confusion_matrix = calc_confusion_matrix(eval_pred, eval_true, nclasses)
    return confusion_matrix

if __name__ == '__main__':
    args = get_args()
    cfg = easydict.EasyDict(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    init_logger(cfg.log_path)
    logging.debug(args)

    main(cfg)
