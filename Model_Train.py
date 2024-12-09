import torch


def train(net, optimizer, criterion, data, data_split, args):
    net.train()
    optimizer.zero_grad()
    x, y, adj = data
    output, weight_loss = net(x, adj, out_loss=True)
    output = torch.softmax(output[data_split[0].to(torch.long)], 1)
    data_y = torch.argmax(y[data_split[0].to(torch.long)], 1)
    loss = criterion(output, data_y)
    acc = accuracy(output, data_y)
    if len(weight_loss) == 1:
        loss += args.wd * weight_loss[0]
    else:
        loss = loss + args.wd * weight_loss[0] + args.alpha * weight_loss[1]
    loss.backward()
    optimizer.step()
    return loss, acc


def val(net, criterion, data, data_split):
    net.eval()
    x, y, adj = data
    output = torch.softmax(net(x, adj)[data_split[1].to(torch.long)], 1)
    data_y = torch.argmax(y[data_split[1].to(torch.long)], 1)
    loss_val = criterion(output, data_y)
    acc_val = accuracy(output, data_y)
    return loss_val, acc_val


def test(net, criterion, data, data_split):
    net.eval()
    x, y, adj = data
    output = torch.softmax(net(x, adj)[data_split[2].to(torch.long)], 1)
    data_y = torch.argmax(y[data_split[2].to(torch.long)], 1)
    loss_test = criterion(output, data_y)
    acc_test = accuracy(output, data_y)
    return loss_test, acc_test


def accuracy(outputs, labels):
    preds = outputs.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
