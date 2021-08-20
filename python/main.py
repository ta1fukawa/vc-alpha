import argparse
import datetime
import glob
import logging
import os
import shutil
import sys
import json
import warnings
import csv

import easydict
import numpy as np
import torch
import tqdm
from sklearn import multiclass, svm

from dataloader import DataLoader
from models import FullModel
from omake import console_menu, console_inputarea

def get_args():
    ## TODO: ここらへんを修正
    parser = argparse.ArgumentParser(description='研究用：音素に対して時間領域で処理して話者埋め込みを求めるやつ')
    parser.add_argument('--gpu',      default=None,      type=str, metavar='N',    help='GPU番号')
    parser.add_argument('--dest-dir', default='dest',    type=str, metavar='PATH', help='出力先ディレクトリのパス')
    parser.add_argument('--code-id',  default='default', type=str, metavar='ID',   help='プログラムコードの識別コード')

    parser.add_argument('--nphonemes-path',   default='resource/jvs_ver1_nphonemes_%(condition)s.txt', type=str, metavar='PATH', help='音素長データの書式付きパス')
    parser.add_argument('--dataset-dir',      default='resource/jvs_ver1_phonemes',                    type=str, metavar='PATH', help='データセットの書式付きパス')
    parser.add_argument('--fast-dataset-dir', default=None,                                            type=str, metavar='PATH', help='データセットの書式付きパス（Fast）')
    
    parser.add_argument('--model-type',       default='stats_pooling', type=str, metavar='TYPE', help='モデルの種類')
    parser.add_argument('-x', '--model-dims', default=None,            type=int, metavar='N',    help='モデルのConv1d/Conv2dの選択')
    parser.add_argument('--patience',         default=4,               type=int, metavar='N',    help='Early Stoppingまでの回数')
    
    parser.add_argument('--no-load-weights', action='store_true', help='重み読み込みの有無')
    parser.add_argument('--no-learn',        action='store_true', help='学習の有無')

    parser.add_argument('--sampling-rate', default=24000, type=int, metavar='N', help='サンプリング周波数')
    parser.add_argument('--nfft',          default=1024,  type=int, metavar='N', help='STFTのウィンドウ幅（通常はPyWorldに依存）')
    parser.add_argument('--nhop',          default=120,   type=int, metavar='N', help='STFTのシフト幅（通常はPyWorldに依存）')

    parser.add_argument('-d', '--deform-type', default=None, type=str, metavar='TYPE', help='変形の種類（variableの場合はバッチサイズを(1, 1)にする）')

    parser.add_argument('-bs', '--batch-length-person',  default=1,  type=int, metavar='N', help='各バッチの話者数')
    parser.add_argument('-bp', '--batch-length-phoneme', default=1,  type=int, metavar='N', help='各バッチの音素数')
    parser.add_argument('--phonemes-length',             default=32, type=int, metavar='N', help='音素の時間長')

    parser.add_argument('-pk', '--person-known-size',        default=16, type=int, metavar='N', help='既知の話者として使用する話者数')
    parser.add_argument('-pu', '--person-unknown-size',      default=16, type=int, metavar='N', help='未知の話者として使用する話者数')
    parser.add_argument('-vt', '--voice-train-size',         default=64, type=int, metavar='N', help='学習に使用する音声ファイル数')
    parser.add_argument('-vc', '--voice-check-size',         default=8,  type=int, metavar='N', help='検証に使用する音声ファイル数')
    parser.add_argument('-svm-vt', '--svm-voice-train-size', default=8,  type=int, metavar='N', help='SVMの学習に使用する音声ファイル数')
    
    args = vars(parser.parse_args(sys.argv[1:]))
    return args

def backup_code(targets, dest_dir):
    for target in targets:
        code_files = sorted(glob.glob(target))
        for code_file in code_files:
            shutil.copyfile(code_file, os.path.join(dest_dir, os.path.split(code_file)[1]))

def init_logger(log_path, mode='w', stdout=True):
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=fmt, filename=log_path, filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def main(cfg):
    warnings.filterwarnings('ignore')
    logging.debug('Config:\n' + json.dumps(args, ensure_ascii=False, indent=4))

    person_no_list = np.array([8, 16, 17, 21, 23, 29, 35, 37, 42, 46, 47, 50, 54, 58, 59, 73, 88, 97])
    voice_no_list  = np.array([5, 18, 24, 40, 42, 44, 46, 55, 56, 59, 60, 61, 63, 65, 71, 73, 75, 81, 84, 85, 87, 93, 94, 98])
    
    logging.debug('person_no_list: ' + str(person_no_list))
    logging.debug('voice_no_list: '  + str(voice_no_list))

    batch_size = (cfg.batch_length_person, cfg.batch_length_phoneme)

    def calc_file_idx(no_list, base_size):
        last_append_size = 0
        while True:
            append_size = np.count_nonzero(no_list < base_size + last_append_size)
            if append_size == last_append_size:
                break
            last_append_size = append_size
        return base_size + last_append_size

    person_known_idx   = calc_file_idx(person_no_list, cfg.person_known_size)
    person_unknown_idx = calc_file_idx(person_no_list, cfg.person_known_size + cfg.person_unknown_size)
    voice_train_idx    = calc_file_idx(voice_no_list,  cfg.voice_train_size)
    voice_check_idx    = calc_file_idx(voice_no_list,  cfg.voice_train_size  + cfg.voice_check_size)

    known_person_list   = list(filter(lambda x:x not in person_no_list, np.arange(person_known_idx)))
    unknown_person_list = list(filter(lambda x:x not in person_no_list, np.arange(person_known_idx, person_unknown_idx)))
    train_voice_list    = list(filter(lambda x:x not in voice_no_list,  np.arange(voice_train_idx)))
    check_voice_list    = list(filter(lambda x:x not in voice_no_list,  np.arange(voice_train_idx, voice_check_idx)))

    model = FullModel(cfg.model_type, cfg.model_dims, cfg.nfft // 2, cfg.phonemes_length, len(known_person_list)).to('cuda')
    logging.info('Model:\n' + str(model))

    if not cfg.no_load_weights:
        load_weights(model, os.path.join(cfg.output_wild_dir, 'weights.pth'))

    if not cfg.no_learn:
        weights_path = os.path.join(cfg.output_dir, 'weights.pth')
        logging.info('Start learning: ' + weights_path)

        train_loader = DataLoader(known_person_list, train_voice_list, batch_size, cfg.nphonemes_path, cfg.fast_dataset_path, cfg.deform_type, cfg.phonemes_length, seed=0)
        valid_loader = DataLoader(known_person_list, check_voice_list, batch_size, cfg.nphonemes_path, cfg.fast_dataset_path, cfg.deform_type, cfg.phonemes_length, seed=0)
        history = learn(model, (train_loader, valid_loader), weights_path, leaning_rate=1e-4, patience=cfg.patience)
        logging.info('History:\n' + json.dumps(history, ensure_ascii=False, indent=4))

    logging.info('Start evaluation')

    # SVMは無駄に時間がかかりすぎるので……
    svm_train_voice_list = list(filter(lambda x:x not in voice_no_list, np.arange(calc_file_idx(voice_no_list, cfg.svm_voice_train_size))))

    known_train_loader   = DataLoader(known_person_list,  svm_train_voice_list, batch_size,  cfg.nphonemes_path, cfg.fast_dataset_path, cfg.deform_type, cfg.phonemes_length)
    known_eval_loader    = DataLoader(known_person_list,  check_voice_list, batch_size,      cfg.nphonemes_path, cfg.fast_dataset_path, cfg.deform_type, cfg.phonemes_length)
    unknown_train_loader = DataLoader(unknown_person_list, svm_train_voice_list, batch_size, cfg.nphonemes_path, cfg.dataset_path,      cfg.deform_type, cfg.phonemes_length)
    unknown_eval_loader  = DataLoader(unknown_person_list, check_voice_list, batch_size,     cfg.nphonemes_path, cfg.dataset_path,      cfg.deform_type, cfg.phonemes_length)
    known_train_embed_pred, known_train_embed_true     = predict(model.embed, known_train_loader)
    known_eval_embed_pred, known_eval_embed_true       = predict(model.embed, known_eval_loader)
    unknown_train_embed_pred, unknown_train_embed_true = predict(model.embed, unknown_train_loader)
    unknown_eval_embed_pred, unknown_eval_embed_true   = predict(model.embed, unknown_eval_loader)
    known_svm_confusion_matrix   = calc_svm_matrix(known_train_embed_pred,   known_train_embed_true,   known_eval_embed_pred,   known_eval_embed_true,   cfg.person_known_size)
    unknown_svm_confusion_matrix = calc_svm_matrix(unknown_train_embed_pred, unknown_train_embed_true, unknown_eval_embed_pred, unknown_eval_embed_true, cfg.person_unknown_size)
    known_svm_acc_rate   = np.trace(known_svm_confusion_matrix).astype(np.int)
    unknown_svm_acc_rate = np.trace(unknown_svm_confusion_matrix).astype(np.int)
    logging.info(f'Known accuracy: {known_svm_acc_rate / len(known_eval_embed_true)} ({known_svm_acc_rate}/{len(known_eval_embed_true)})')
    logging.info(f'Unknown accuracy: {unknown_svm_acc_rate / len(unknown_eval_embed_true)} ({unknown_svm_acc_rate}/{len(unknown_eval_embed_true)})')
    with open(os.path.join(cfg.output_dir, 'known_svm_confmat.csv'), 'w') as f:
        csv.writer(f).writerows(known_svm_confusion_matrix)
    with open(os.path.join(cfg.output_dir, 'unknown_svm_confmat.csv'), 'w') as f:
        csv.writer(f).writerows(unknown_svm_confusion_matrix)

def load_weights(model, weights_path):
    existing_weights_paths = sorted(glob.glob(weights_path))
    if len(existing_weights_paths) == 0:
        logging.info('Weights is not found.')
        return

    logging.info('Loading weights: ' + existing_weights_paths[-1])
    model.load_state_dict(torch.load(existing_weights_paths[-1]))

def learn(model, loaders, weights_path, leaning_rate, patience):

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

    train_loss /= len(loader) * len(true)
    train_acc  /= len(loader) * len(true)
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

    valid_loss /= len(loader) * len(true)
    valid_acc  /= len(loader) * len(true)
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
    logging.info('Start svm learning')
    svc = svm.SVC(C=1., kernel='rbf', gamma=0.01, shrinking=False, verbose=False)
    classifier = multiclass.OneVsRestClassifier(svc)
    classifier.fit(train_data, train_true)

    eval_pred = classifier.predict(eval_data)
    confusion_matrix = calc_confusion_matrix(eval_pred, eval_true, nclasses)
    return confusion_matrix

if __name__ == '__main__':
    args = get_args()

    # GPU番号
    if args['gpu'] is None:
        gpu_text = console_inputarea('使用するGPU番号（0以上）を入力してください', 'CPUを使用する場合は何も入力せずエンター', numeric_ok=True, lowercase_ok=False, uppercase_ok=False, sign_ok=False)
        if gpu_text is None: exit(0)
        args['gpu'] = gpu_text

    # 時間幅変形の方法
    if args['deform_type'] is None:
        deform_types = ['stretch', 'padding', 'variable']
        selected_no = console_menu('時間幅変形の方法を選択してください', deform_types)
        if selected_no < 0: exit(0)
        args['deform_type'] = deform_types[selected_no]

    # 畳み込み次元数
    if args['model_dims'] is None:
        selected_no = console_menu('CNNの次元数を選択してください', ['1次元（Conv1d）', '2次元（Conv2d）'])
        if selected_no < 0: exit(0)
        args['model_dims'] = selected_no + 1

    # 出力先のパス
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    specific = {
        'deform_type': args['deform_type'],
        'model_dims' : args['model_dims'],
        'code_id'    : args['code_id'],
    }
    output_dir_format = os.path.join(args['dest_dir'], '%(code_id)s/%(deform_type)s-%(model_dims)d/%(datetime)s/')
    args['output_dir']      = output_dir_format % { **specific, 'datetime': now }
    args['output_wild_dir'] = output_dir_format % { **specific, 'datetime': '*' }

    # データセットのパス
    args['dataset_path'] = os.path.join(args['dataset_dir'], 'jvs%(person)03d/VOICEACTRESS100_%(voice)03d_%(deform_type)s.npz')
    if args['fast_dataset_dir'] is not None:
        args['fast_dataset_path'] = os.path.join(args['fast_dataset_dir'], 'jvs%(person)03d/VOICEACTRESS100_%(voice)03d_%(deform_type)s.npz')
    else:
        args['fast_dataset_path'] = args['dataset_path']

    cfg = easydict.EasyDict(args)

    # CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    os.makedirs(cfg.output_dir, exist_ok=True)
    backup_code(['python/*.py'], cfg.output_dir)

    init_logger(os.path.join(cfg.output_dir, 'general.log'))

    cfg.model_type = 'linear'

    main(cfg)
