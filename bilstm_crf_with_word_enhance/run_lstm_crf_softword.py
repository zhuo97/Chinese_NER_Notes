import argparse
import config
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from progressbar import ProgressBar
from model.bilstm_crf_softword import BiLSTM_CRF_SoftWord
from data_processor import CLUENERProcessor, MSRAProcessor
from vocabulary import Vocabulary
from datasets import NERDatasetSoftWord
from utils import AverageMeter, load_model, init_logger, logger
from ner_metrics import SeqEntityScore
from word_enhance import Soft2Idx


def load_and_cache_dataset(args, processor, vocab, data_type="train"):
    cached_dataset_file = args.data_dir / "cached_crf-{}_{}_{}".format(
        data_type,
        args.arch,
        str(args.task_name)
    )
    if cached_dataset_file.exists():
        logger.info("Loading feature from dataset file at %s", args.data_dir)
        dataset = torch.load(cached_dataset_file)
    else:
        logger.info("Creating feature from dataset file at %s", args.data_dir)
        if data_type == "train":
            samples = processor.get_train_samples()
        elif data_type == "test":
            samples = processor.get_test_samples()

        dataset = NERDatasetSoftWord(samples, vocab, args.label2id, args.device)
        logger.info("Saving features into cached file %s", cached_dataset_file)
        torch.save(dataset, str(cached_dataset_file))
    return dataset


def train(args, model, processor, vocab):
    train_dataset = load_and_cache_dataset(args, processor, vocab, data_type="train")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.learning_rate)

    best_f1 = 0
    for epoch in range(1, 1 + args.epochs):
        pbar = ProgressBar(n_total=len(train_loader), desc="Training")
        train_loss = AverageMeter()
        model.train()
        assert model.training
        for step, batch_samples in enumerate(train_loader):
            batch_word_ids, lengths, batch_softword_ids, batch_label_ids = batch_samples
            batch_word_ids = batch_word_ids.to(args.device)
            batch_softword_ids = batch_softword_ids.to(args.device)
            batch_label_ids = batch_label_ids.to(args.device)
            loss = model(batch_word_ids, lengths, batch_softword_ids, batch_label_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pbar(step=step, info={"loss": loss.item()})
            train_loss.update(loss.item(), n=1)
        train_log = {"loss": train_loss.avg}
        eval_log, class_info = evaluate(args, model, processor, vocab)
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


def evaluate(args, model, processor, vocab):
    eval_dataset = load_and_cache_dataset(args, processor, vocab, data_type="test")
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        collate_fn=eval_dataset.collate_fn
    )
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, batch_samples in enumerate(eval_dataloader):
            batch_word_ids, lengths, batch_softword_ids, batch_label_ids = batch_samples
            batch_word_ids = batch_word_ids.to(args.device)
            batch_softword_ids = batch_softword_ids.to(args.device)
            batch_label_ids = batch_label_ids.to(args.device)
            loss = model(batch_word_ids, lengths, batch_softword_ids, batch_label_ids)
            eval_loss.update(loss.item(), n=1)
            predicted_labels = model.predict(batch_word_ids, lengths, batch_softword_ids)
            batch_label_ids = batch_label_ids.cpu().numpy()
            labels = [[args.id2label[id_] for id_ in label_ids[:lengths[idx]]]
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
    parser.add_argument("--arch", default="bilstm_crf_softword", type=str)
    parser.add_argument("--learning_rate", default=0.0015, type=float)
    parser.add_argument("--grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--hidden_size", default=300, type=int)
    parser.add_argument("--task_name", default="ner", type=str)
    parser.add_argument("--dataset", default="msra", type=str)
    args = parser.parse_args()

    if args.dataset == "cluner":
        args.data_dir = config.cluner_data_dir
        args.output_dir = config.cluner_output_dir / '{}'.format(args.arch)
        args.id2label = {i: label for i, label in enumerate(config.cluner_label2id)}
        args.label2id = config.cluner_label2id
    elif args.dataset == "msra":
        args.data_dir = config.msra_data_dir
        args.output_dir = config.msra_output_dir / '{}'.format(args.arch)
        args.id2label = {i: label for i, label in enumerate(config.msra_label2id)}
        args.label2id = config.msra_label2id

    if args.gpu != "":
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")

    if not args.output_dir.exists():
        args.output_dir.mkdir()

    vocab = Vocabulary(config.pretrain_char_emb_path)

    init_logger(log_file=str(args.output_dir / "{}-{}.log".format(args.arch, args.task_name)))

    if args.dataset == "cluner":
        processor = CLUENERProcessor(data_dir=args.data_dir)
    elif args.dataset == "msra":
        processor = MSRAProcessor(data_dir=args.data_dir)

    model = BiLSTM_CRF_SoftWord(
        hidden_size=args.hidden_size,
        label2id=args.label2id,
        device=args.device,
        word_enhance_size=len(Soft2Idx),
        pretrain_embedding_weights=torch.tensor(vocab.pretrain_embeddings, dtype=torch.float32)
    )

    model.to(args.device)
    if args.do_train:
        train(args, model, processor, vocab)
    if args.do_eval:
        model_path = args.output_dir / "best-model.bin"
        model = load_model(model, model_path=str(model_path))
        eval_log, class_info = evaluate(args, model, processor, vocab)
        print(eval_log)
        print("Eval Entity Score: ")
        for key, value in class_info.items():
            info = f"Subject: {key} - Acc: {value['precision']} - Recall: {value['recall']} - F1: {value['f1']}"
            logger.info(info)


if __name__ == "__main__":
    main()
