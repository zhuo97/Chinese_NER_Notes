import argparse
import config
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from progressbar import ProgressBar
from model.flat import FLAT
from processor import MSRAProcessor
from vocabulary import Vocabulary
from datasets import FLATDataset
from utils import AverageMeter, load_model, init_logger, logger, load_pickle, save_pickle
from ner_metrics import SeqEntityScore


def load_and_cache_dataset(args, config, processor, vocab, data_type="train"):
    cached_dataset_file = config.cache_dir / "cached_crf-{}_{}_{}_{}".format(
        data_type,
        args.arch,
        args.task_name,
        args.dataset)
    if cached_dataset_file.exists():
        logger.info("Loading feature from dataset file at %s", config.cache_dir)
        dataset = load_pickle(cached_dataset_file)
    else:
        logger.info("Creating feature from dataset file at %s", args.data_dir)
        if data_type == "train":
            samples = processor.get_train_samples()
        elif data_type == "test":
            samples = processor.get_test_samples()

        dataset = FLATDataset(samples, 
                              vocab, 
                              args.label2id, 
                              config.device, 
                              config.overall_max_char_seq_len, 
                              config.overall_max_seq_len)
        logger.info("Saving features into cached file %s", cached_dataset_file)
        save_pickle(dataset, str(cached_dataset_file))
    return dataset


def train(args, config, model, processor, vocab):
    train_dataset = load_and_cache_dataset(args, config, processor, vocab, data_type="train")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=config.learning_rate)

    best_f1 = 0
    for epoch in range(1, 1 + config.epochs):
        pbar = ProgressBar(n_total=len(train_loader), desc="Training")
        train_loss = AverageMeter()
        model.train()
        assert model.training
        for step, batch_samples in enumerate(train_loader):
            batch_lattice, batch_char_seq_len, batch_num_words, batch_pos_s, batch_pos_e, batch_label_ids = batch_samples
            loss, _ = model(batch_lattice, batch_char_seq_len, batch_num_words, batch_pos_s, batch_pos_e, batch_label_ids)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar(step=step, info={"loss": loss.item()})
            train_loss.update(loss.item(), n=1)
        train_log = {"loss": train_loss.avg}
        eval_log, class_info = evaluate(args, config, model, processor, vocab)
        logs = dict(train_log, **eval_log)
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        logger.info(show_info)
        if logs["eval_f1"] > best_f1:
            logger.info(f"\nEpoch {epoch}: eval_f1 improved from {best_f1} to {logs['eval_f1']}")
            logger.info("save model to disk")
            best_f1 = logs["eval_f1"]
            if isinstance(model, nn.DataParallel):
                model_stat_dict = model.module.state_dict()
            else:
                model_stat_dict = model.state_dict()
            state = {"epoch": epoch, "arch": args.arch, "state_dict": model_stat_dict}
            model_path = args.output_dir / "best-model.bin"
            torch.save(state, str(model_path))
            logger.info("Eval Entity Score: ")
            for key, value in class_info.items():
                info = f"Subject: {key} - Acc: {value['precision']} - Recall: {value['recall']} - F1: {value['f1']}"
                logger.info(info)


def evaluate(args, config, model, processor, vocab):
    eval_dataset = load_and_cache_dataset(args, config, processor, vocab, data_type="test")
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=config.batch_size,
        collate_fn=eval_dataset.collate_fn
    )
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, batch_samples in enumerate(eval_dataloader):
            batch_lattice, batch_char_seq_len, batch_num_words, batch_pos_s, batch_pos_e, batch_label_ids = batch_samples
            loss, predicted_labels = model(batch_lattice, batch_char_seq_len, batch_num_words, batch_pos_s, batch_pos_e, batch_label_ids)
            eval_loss.update(loss.item(), n=1)
            batch_label_ids = batch_label_ids.cpu().numpy()
            batch_char_seq_len = batch_char_seq_len.cpu().numpy()
            labels = [[args.id2label[id_] for id_ in label_ids[:batch_char_seq_len[idx]]]
                      for idx, label_ids in enumerate(batch_label_ids)]
            metric.update(pred_paths=predicted_labels, label_paths=labels)
            pbar(step=step)
    eval_info, class_info = metric.result()
    eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    result = {"eval_loss": eval_loss.avg}
    result = dict(result, **eval_info)
    return result, class_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_eval", default=False, action="store_true")
    parser.add_argument("--markup", default="bio", type=str, choices=["bios", "bio"])
    parser.add_argument("--arch", default="flat", type=str)
    parser.add_argument("--task_name", default="ner", type=str)
    parser.add_argument("--dataset", default="resume", type=str)
    args = parser.parse_args()
    
    if args.dataset == "msra":
        args.data_dir = config.msra_data_dir
        args.output_dir = config.msra_output_dir / '{}'.format(args.arch)

        args.id2label = {i: label for i, label in enumerate(config.msra_label2id)}
        args.label2id = config.msra_label2id
        config.num_labels = len(args.id2label)
        
        processor = MSRAProcessor(data_dir=config.msra_data_dir)
        
    elif args.dataset == "resume":
        args.data_dir = config.resume_data_dir
        args.output_dir = config.resume_output_dir / '{}'.format(args.arch)

        args.id2label = {i: label for i, label in enumerate(config.resume_label2id)}
        args.label2id = config.resume_label2id
        config.num_labels = len(args.id2label)
        
        processor = MSRAProcessor(data_dir=config.resume_data_dir)

    if not args.output_dir.exists():
        args.output_dir.mkdir()

    init_logger(log_file=str(args.output_dir / "{}-{}-{}.log".format(args.arch, args.task_name, args.dataset)))

    vocab = Vocabulary(config.pretrain_word_emb_path)
    
    # embedding_size 初始化
    config.char_embedding_size = vocab.pretrain_embeddings_size
    config.word_embedding_size = vocab.pretrain_embeddings_size
    
    model = FLAT(config, vocab)

    model.to(config.device)

    if args.do_train:
        train(args, config, model, processor, vocab)
    if args.do_eval:
        model_path = args.output_dir / "best-model.bin"
        model = load_model(model, model_path=str(model_path))
        eval_log, class_info = evaluate(args, config, model, processor, vocab)
        print(eval_log)
        print("Eval Entity Score: ")
        for key, value in class_info.items():
            info = f"Subject: {key} - Acc: {value['precision']} - Recall: {value['recall']} - F1: {value['f1']}"
            logger.info(info)


if __name__ == "__main__":
    main()
