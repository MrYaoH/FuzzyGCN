import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize

import torch
import torch.nn as nn

import Model
from Model_Train import *
from load_data import adj_transform


def fit_logistic_regression(X, y, data_random_seed=1, repeat=10):
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)

    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool)

    # normalize x
    X = normalize(X, norm='l2')

    # set random state
    rng = np.random.RandomState(data_random_seed)  # this will ensure the dataset will be split exactly the same
                                                   # throughout training

    accuracies = []
    for _ in range(repeat):
        # different random split after each repeat
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)

        # grid search with one-vs-rest classifiers
        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                           n_jobs=5, cv=cv, verbose=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)

        test_acc = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(test_acc)
    return accuracies


def fit_logistic_regression_preset_splits(X, y, train_masks, val_masks, test_mask):
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool)

    # normalize x
    X = normalize(X, norm='l2')

    accuracies = []
    for split_id in range(train_masks.shape[1]):
        # get train/val/test masks
        train_mask, val_mask = train_masks[:, split_id], val_masks[:, split_id]

        # make custom cv
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # grid search with one-vs-rest classifiers
        best_test_acc, best_acc = 0, 0
        for c in 2.0 ** np.arange(-10, 11):
            clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
            clf.fit(X_train, y_train)

            y_pred = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)
            val_acc = metrics.accuracy_score(y_val, y_pred)
            if val_acc > best_acc:
                best_acc = val_acc
                y_pred = clf.predict_proba(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)
                best_test_acc = metrics.accuracy_score(y_test, y_pred)

        accuracies.append(best_test_acc)
    print(np.mean(accuracies))
    return accuracies


def eval_model(args, data, data_split, device, logging):
    n_feat = data.x.shape[1]
    n_class = int(torch.max(data.y)) + 1
    if args.model == 'FGCN':
        net = getattr(Model, args.model)(n_feat, args.hid, n_class, data.x, fz_wd=True)
    else:
        net = getattr(Model, args.model)(n_feat, args.hid, n_class)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=0.)
    criterion = nn.CrossEntropyLoss()

    adj = adj_transform(data).to(device)

    # transform targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
    y = one_hot_encoder.fit_transform(data.y.reshape(-1, 1).cpu().numpy()).astype(np.int)

    _train_acc = []
    _val_acc = []
    _train_loss = []
    _val_loss = []

    input = data.x, torch.from_numpy(y).to(device), adj

    best_acc = 0
    val_acc = 0
    for e in range(args.epochs):
        train_loss, train_acc = train(net, optimizer, criterion, input, data_split, args)
        val_loss, val_acc = val(net, criterion, input, data_split)
        _train_acc.append(train_acc.item())
        _train_loss.append(train_loss.item())
        _val_acc.append(val_acc.item())
        _val_loss.append(val_loss.item())

        logging.debug('Epoch %d: train loss %.3f train acc: %.3f, val loss: %.3f val acc %.3f.' %
                      (e, train_loss, train_acc, val_loss, val_acc))

        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), './result/' + '{}-checkpoint-best-acc.pkl'.format(args.dataset))
    net.load_state_dict(torch.load('./result/' + '{}-checkpoint-best-acc.pkl'.format(args.dataset)))
    test_loss, test_acc = test(net, criterion, input, data_split)

    return test_acc


def dataset_split(y, device, train_ratio=0.1, val_ratio=0., test_ratio=0.9):
    N = y.shape[0]
    class_num = int(torch.max(y)) + 1
    Y = y.detach().cpu()

    train_indices = np.array([])

    data = np.arange(N)
    np.random.shuffle(data)

    if train_ratio >= 1.:
        for c in range(class_num):
            c_index = (Y == c).nonzero().numpy()
            train_indices = np.union1d(train_indices, c_index[: train_ratio])

        val_test = np.setdiff1d(np.arange(N), train_indices)
        val_indices = val_test[: val_ratio]
        test_indices = val_test[val_ratio: val_ratio + test_ratio]

    else:
        num_train = np.ceil(N * train_ratio)
        num_val = np.ceil(N * val_ratio)
        train_indices = data[: num_train]
        val_indices = data[num_train: num_train + num_val]
        test_indices = data[num_train + num_val:]

    return torch.from_numpy(train_indices.astype(dtype=np.int)).to(device), \
        torch.from_numpy(val_indices.astype(dtype=np.int)).to(device), \
        torch.from_numpy(test_indices.astype(dtype=np.int)).to(device)
