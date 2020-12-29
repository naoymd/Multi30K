#-*- coding:utf-8 -*-
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
import random
import shutil
import spacy
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torchtext
import yaml
from addict import Dict
from net import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    torch.backends.cudnn.benchmark = True
spacy = spacy.load("en_core_web_sm")

def parse_args():
    parser = argparse.ArgumentParser(description="train a network for Multi30K Dataset review classification")
    parser.add_argument("config", type=str, help="path of a config file")
    args = parser.parse_args()
    return args

def draw_heatmap(data, row_labels, column_labels, title, save_dir=None, name=None):
    fig, ax = plt.subplots(figsize=(10, 1))
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False, rotation=30)
    ax.set_yticklabels(column_labels, minor=False, rotation=90)
    plt.title(title)
    plt.savefig(os.path.join(save_dir, name + ".png"), bbox_inches="tight")
    plt.close()


def sec2str(sec):
    if sec < 60:
        return "elapsed: {:02d}s".format(int(sec))
    elif sec < 3600:
        min = int(sec / 60)
        sec = int(sec - min * 60)
        return "elapsed: {:02d}m{:02d}s".format(min, sec)
    elif sec < 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        return "elapsed: {:02d}h{:02d}m{:02d}s".format(hr, min, sec)
    elif sec < 365 * 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        dy = int(hr / 24)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        hr = int(hr - dy * 24)
        return "elapsed: {:02d} days, {:02d}h{:02d}m{:02d}s".format(dy, hr, min, sec)


def train(train_iter, val_iter, net, criterion, optimizer, lr_scheduler, TEXT, CONFIG, args):
    print("start train and validation")
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    best_loss = 100
    result_dir = os.path.join("./result", CONFIG.rnn)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    save_dir = os.path.join("./result", CONFIG.rnn, "heatmap/val")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join("./checkpoint")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(CONFIG.epoch_num):
        start = time.time()
        train_loss = train_acc = val_loss = val_acc = 0
        net.train()
        print("epoch", epoch + 1)
        for i, batch_train in enumerate(train_iter):
            story = batch_train.story[0]
            query = batch_train.query[0]
            answer = batch_train.answer[0]
            if story.size(0) != CONFIG.batch_size:
                break

            story = story.to(device)
            query = query[:, :3].to(device)
            answer = answer[:, 0].to(device)

            optimizer.zero_grad()
            if CONFIG.self_attention:
                output, attention_map = net(story, query)
                # print('attention_map')
                # print(attention_map.size())
            else:
                output = net(story, query)
            # if i % 500 == 0:
            #     print('output')
            #     print(output.size())
            #     print(output)
            #     # print(output.max(1))
            loss = criterion(output, answer)
            train_loss += loss.item()
            # print(train_loss)
            train_acc += (output.max(1)[1] == answer).sum().item()
            # print(train_acc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CONFIG.get('clip', 0.5))
            optimizer.step()
            # break
        lr_scheduler.step(loss)
        avg_train_loss = train_loss / len(train_iter.dataset)
        avg_train_acc = train_acc / len(train_iter.dataset)
        # print('loss', avg_train_loss)
        train_time = sec2str(time.time() - start)
        print("train", train_time)
        # break

        start = time.time()
        net.eval()
        with torch.no_grad():
            for i, batch_val in enumerate(val_iter):
                story = batch_val.story[0]
                query = batch_val.query[0]
                answer = batch_val.answer[0]
                if story.size(0) != CONFIG.batch_size:
                    break

                story = story.to(device)
                query = query[:, :3].to(device)
                answer = answer[:, 0].to(device)

                if CONFIG.self_attention:
                    output, attention_map = net(story, query)
                    if (
                        epoch % 1000 == 0 or epoch + 1 == CONFIG.epoch_num
                    ) and i % 5 == 0:
                        for j in range(CONFIG.batch_size):
                            if j % 50 == 0:
                                # heat_map = attention_map[j, :, :].permute(1, 0).cpu().detach().numpy().sum(axis=0, keepdims=True)
                                heat_map = attention_map[j, :, :].cpu().detach().numpy()
                                sentence = [TEXT.vocab.itos[data] for data in story[j, :]]
                                title = [TEXT.vocab.itos[data] for data in query[j, :]]
                                label = TEXT.vocab.itos[answer[j]]
                                name = str(epoch+1) + "_" + str(i) + "_" + str(j)
                                # print('name', name)
                                # print('sentence', sentence)
                                draw_heatmap(heat_map, sentence, label, title, save_dir, name)
                else:
                    output = net(story, query)

                loss = criterion(output, answer)
                val_loss += loss.item()
                val_acc += (output.max(1)[1] == answer).sum().item()
        avg_val_loss = val_loss / len(val_iter.dataset)
        avg_val_acc = val_acc / len(val_iter.dataset)

        if avg_val_loss <= best_loss:
            print("save parameters")
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, "checkpoint.pth"))
            best_loss = avg_val_loss

        val_time = sec2str(time.time() - start)
        print("validation", val_time)
        print(
            "Epoch [{}/{}], train_loss: {loss:.4f}, train_acc: {acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}".format(
                epoch + 1,
                CONFIG.epoch_num,
                loss=avg_train_loss,
                acc=avg_train_acc,
                val_loss=avg_val_loss,
                val_acc=avg_val_acc,
            )
        )
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)
        
        plt.figure()
        plt.plot(train_loss_list, label="train")
        plt.plot(val_loss_list, label="val")
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(result_dir, "loss.png"))
        plt.close()
        plt.figure()
        plt.plot(train_acc_list, label="train")
        plt.plot(val_acc_list, label="val")
        plt.legend()
        plt.savefig(os.path.join(result_dir, "acc.png"))
        plt.close()
        # break


def test(test_iter, net, TEXT, CONFIG):
    print("start test")
    start = time.time()
    save_dir = os.path.join("./result", CONFIG.rnn, "heatmap/test")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join("./checkpoint", "checkpoint.pth")
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval()
    with torch.no_grad():
        total = 0
        test_acc = 0
        for i, batch_test in enumerate(test_iter):
            story = batch_test.story[0]
            query = batch_test.query[0]
            answer = batch_test.answer[0]
            if story.size(0) != CONFIG.batch_size:
                break

            story = story.to(device)
            query = query[:, :3].to(device)
            answer = answer[:, 0].to(device)

            if CONFIG.self_attention:
                output, attention_map = net(story, query)
                if  i % 50 == 0:
                    for j in range(CONFIG.batch_size):
                        if j % 50 == 0:
                            # heat_map = attention_map[j, :, :].permute(1, 0).cpu().detach().numpy().sum(axis=0, keepdims=True)
                            heat_map = attention_map[j, :, :].cpu().detach().numpy()
                            sentence = [TEXT.vocab.itos[data] for data in story[j, :]]
                            title = [TEXT.vocab.itos[data] for data in query[j, :]]
                            label = TEXT.vocab.itos[answer[j]]
                            name = str(i) + "_" + str(j)
                            # print('name', name)
                            # print('sentence', sentence)
                            draw_heatmap(heat_map, sentence, label, title, save_dir, name)
            else:
                output = net(story, query)

            test_acc += (output.max(1)[1] == answer).sum().item()
            total += answer.size(0)
    print("精度: {} %".format(100 * test_acc / total))
    print("test", sec2str(time.time() - start))


def main():
    args = parse_args()
    CONFIG = Dict(yaml.safe_load(open(args.config)))
    pprint.pprint(CONFIG)

    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    torch.cuda.manual_seed_all(CONFIG.seed)
    torch.backends.cudnn.deterministic = True
    
    TEXT = torchtext.data.Field(
        sequential=True,
        tokenize="spacy",
        lower=True,
        fix_length=CONFIG.fix_length,
        batch_first=True,
        include_lengths=True,
    )

    start = time.time()
    print("Loading ...")

    train_dataset, val_dataset, test_dataset = torchtext.datasets.BABI20.splits(
        TEXT, root="./data"
    )
    print("train dataset", len(train_dataset))
    print("val dataset", len(val_dataset))
    print("test dataset", len(test_dataset))
    print("Loading time", sec2str(time.time() - start))

    TEXT.build_vocab(
        train_dataset,
        min_freq=CONFIG.min_freq,
    )

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train_dataset, val_dataset, test_dataset),
        batch_size=CONFIG.batch_size,
        sort_key=lambda x: len(x.story),
        repeat=False,
        shuffle=True,
    )

    print(
        "train_iter {}, val_iter {}, test_iter {}".format(
            len(train_iter.dataset), len(val_iter.dataset), len(test_iter.dataset)
        )
    )
    vocab_size = len(TEXT.vocab)
    print('vocab_size', vocab_size)

    net = Net(vocab_size, CONFIG).to(device)
    
    net = torch.nn.DataParallel(net, device_ids=[0])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=math.sqrt(float(CONFIG.learning_rate))
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=math.sqrt(float(CONFIG.factor)),
        verbose=True,
        min_lr=math.sqrt(float(CONFIG.min_learning_rate)),
    )

    train(train_iter, val_iter, net, criterion, optimizer, lr_scheduler, TEXT, CONFIG, args)
    test(test_iter, net, TEXT, CONFIG)
    print("finished", sec2str(time.time() - start))


if __name__ == "__main__":
    main()