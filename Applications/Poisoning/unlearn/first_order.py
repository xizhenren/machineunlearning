import os
from os.path import dirname as parent
import json
import argparse

from Applications.Poisoning.configs.config import Config
from Applications.Poisoning.model import get_VGG_CIFAR10
from Applications.Poisoning.poison.injector import LabelflipInjector
from Applications.Poisoning.dataset import Cifar10
from Applications.Poisoning.unlearn.common import evaluate_unlearning
from util import UnlearningResult, reduce_dataset

#new
from Applications.Poisoning.unlearn.common import flatten_model_weights
from Unlearner.Wasserstein import wdp_epsilon_from_params
from util import UnlearningResult

def get_parser():
    parser = argparse.ArgumentParser("first_order", description="Unlearn with first-order method.")
    parser.add_argument("model_folder", type=str, help="base directory to save models and results in")
    parser.add_argument("--config_file", type=str, default='unlearn_config.json',
                        help="config file with parameters for this experiment")
    parser.add_argument("--verbose", "-v", action="store_true", help="enable additional outputs")
    return parser


def run_experiment(model_folder, train_kwargs, poison_kwargs, unlearn_kwargs, reduction=1.0, verbose=False):
    data = Cifar10.load()
    (x_train, y_train), _, _ = data
    y_train_orig = y_train.copy()

    # inject label flips
    injector_path = os.path.join(model_folder, 'injector.pkl')
    if os.path.exists(injector_path):
        injector = LabelflipInjector.from_pickle(injector_path)
    else:
        injector = LabelflipInjector(parent(model_folder), **poison_kwargs)
    x_train, y_train = injector.inject(x_train, y_train)
    data = ((x_train, y_train), data[1], data[2])

    # prepare unlearning data
    (x_train,  y_train), _, _ = data
    x_train, y_train, idx_reduced, delta_idx = reduce_dataset(
        x_train, y_train, reduction=reduction, delta_idx=injector.injected_idx)
    if verbose:
        print(f">> reduction={reduction}, new train size: {x_train.shape[0]}")
    y_train_orig = y_train_orig[idx_reduced]
    data = ((x_train, y_train), data[1], data[2])

    model_init = lambda: get_VGG_CIFAR10(dense_units=train_kwargs['model_size'])
    poisoned_filename = 'poisoned_model.hdf5'
    repaired_filename = 'repaired_model.hdf5'
    first_order_unlearning(model_folder, poisoned_filename, repaired_filename, model_init, data,
                           y_train_orig, injector.injected_idx, unlearn_kwargs, verbose=verbose)
 


def first_order_unlearning(model_folder, poisoned_filename, repaired_filename, model_init, data, y_train_orig, delta_idx,
                            unlearn_kwargs, order=1, verbose=False):
    unlearning_result = UnlearningResult(model_folder)
    poisoned_weights = os.path.join(parent(model_folder), poisoned_filename)
    log_dir = model_folder

    # start unlearning hyperparameter search for the poisoned model
    with open(model_folder.parents[2]/'clean'/'train_results.json', 'r') as f:
        clean_acc = json.load(f)['accuracy']
    repaired_filepath = os.path.join(model_folder, repaired_filename)
    cm_dir = os.path.join(model_folder, 'cm')
    os.makedirs(cm_dir, exist_ok=True)
    unlearn_kwargs['order'] = order
    acc_before, acc_after, diverged, logs, unlearning_duration_s, params = evaluate_unlearning(model_init, poisoned_weights, data, delta_idx, y_train_orig, unlearn_kwargs, clean_acc=clean_acc,
                                                                                       repaired_filepath=repaired_filepath, verbose=verbose, cm_dir=cm_dir, log_dir=log_dir)
    acc_perc_restored = (acc_after - acc_before) / (clean_acc - acc_before)

    #new

    # === WDP 认证：计算 repaired（遗忘后） vs retrain（在 D' 上重训）的 Wasserstein ε ===
    # 1) 取“遗忘后(repaired)”参数向量
    repaired_model = model_init()
    repaired_model.load_weights(repaired_filepath)
    theta_unlearn = flatten_model_weights(repaired_model)

    # 2) 构造 D'（把被投毒索引的标签改回原标签）
    (x_train, y_train), _, _ = data
    y_retrain = y_train.copy()
    if delta_idx is not None and len(delta_idx) > 0:
        y_retrain[delta_idx] = y_train_orig[delta_idx]

    # 3) 得到“重训”模型参数（优先加载已存在的 clean/重训权重；没有就快速重训几轮）
    clean_model_path = os.path.join(parent(model_folder), 'clean_model.hdf5')
    retrained_model = model_init()
    if os.path.exists(clean_model_path):
        retrained_model.load_weights(clean_model_path)
    else:
        # 读取训练配置（与 clean 训练同目录下的 train_config.json）
        train_cfg = Config.from_json(os.path.join(parent(model_folder), 'train_config.json'))
        opt_name = (train_cfg.get('optimizer', 'Adam') or 'Adam').lower()
        lr = float(train_cfg.get('learning_rate', 1e-3))
        bs = int(train_cfg.get('batch_size', 128))
        epochs = int(train_cfg.get('epochs', 5))  # 这里用较小轮次做 baseline

        import tensorflow as tf
        if opt_name == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=lr)

        retrained_model.compile(optimizer=opt,
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])
        retrained_model.fit(x_train, y_retrain,
                            batch_size=bs, epochs=epochs,
                            validation_split=0.0, verbose=0)

    theta_retrain = flatten_model_weights(retrained_model)

    # 4) 计算并写入 WDP 的 ε（同噪声平移 ⇒ W 距离 = 参数 L2 差）
    eps_wdp = wdp_epsilon_from_params(theta_unlearn, theta_retrain, mu=1.0)
    unlearning_result.update({
        'wdp_mu': 2.0,
        'wdp_epsilon': float(eps_wdp),
    })
    # === 结束：后面继续原来的 update/save ===

    #--------------

    unlearning_result.update({
        'acc_clean': clean_acc,
        'acc_before_fix': acc_before,
        'acc_after_fix': acc_after,
        'acc_perc_restored': acc_perc_restored,
        'diverged': diverged,
        'n_gradients': sum(logs),
        'unlearning_duration_s': unlearning_duration_s,
        'num_params': params
    })
    unlearning_result.save()


def main(model_folder, config_file, verbose):
    config_file = os.path.join(model_folder, config_file)
    train_kwargs = Config.from_json(os.path.join(parent(model_folder), 'train_config.json'))
    unlearn_kwargs = Config.from_json(config_file)
    poison_kwargs = Config.from_json(os.path.join(parent(model_folder), 'poison_config.json'))
    run_experiment(model_folder, train_kwargs, poison_kwargs, unlearn_kwargs, verbose=verbose)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
