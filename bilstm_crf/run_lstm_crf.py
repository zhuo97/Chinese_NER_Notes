import argparse
import config
import torch
import torch.nn as nn
from torch import optim
from progressbar import ProgressBar
from bilstm_crf import BiLSTM_CRF
from data_processer import CluenerProcessor
from dataset_loader import DatasetLoader
from utils import AverageMeter, load_model, init_logger, logger
from ner_metrics import SeqEntityScore


def load_and_cache_examples(args, processor, data_type="train"):
    cached_examples_file = args.data_dir / "cached_crf-{}_{}_{}".format(
        data_type,
        args.arch,
        str(args.task_name)
    )
    if cached_examples_file.exists():
        logger.info("Loading feature from dataset file at %s", args.data_dir)
        examples = torch.load(cached_examples_file)
    else:
        logger.info("Creating feature from dataset file at %s", args.data_dir)
        if data_type == "train":
            examples = processor.get_train_examples()
        elif data_type == "dev":
            examples = processor.get_dev_examples()
        logger.info("Saving features into cached file %s", cached_examples_file)
        torch.save(examples, str(cached_examples_file))
    return examples


def train(args, model, processor):
    train_dataset = load_and_cache_examples(args, processor, data_type="train")
    train_loader = DatasetLoader(
        data=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        sort=False,
        vocab=processor.vocab,
        label2id=args.label2id
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.learning_rate)

    best_f1 = 0
    for epoch in range(1, 1 + args.epochs):
        pbar = ProgressBar(n_total=len(train_loader), desc="Training")
        train_loss = AverageMeter()
        model.train()
        assert model.training
        for step, batch in enumerate(train_loader):
            input_ids, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_tags = input_tags.to(args.device)
            loss = model.get_loss(input_ids, input_lens, input_tags)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pbar(step=step, info={"loss": loss.item()})
            train_loss.update(loss.item(), n=1)
        train_log = {"loss": train_loss.avg}
        eval_log, class_info = evaluate(args, model, processor)
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


def evaluate(args, model, processor):
    eval_dataset = load_and_cache_examples(args, processor, data_type="dev")
    eval_dataloader = DatasetLoader(
        data=eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        sort=False,
        vocab=processor.vocab,
        label2id=args.label2id
    )
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            input_ids, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_tags = input_tags.to(args.device)
            loss = model.get_loss(input_ids, input_lens, input_tags)
            eval_loss.update(loss.item(), n=1)
            predicted_tags, _ = model(input_ids, input_lens)
            input_tags = input_tags.cpu().numpy()
            target = [input_[:len_] for input_, len_ in zip(input_tags, input_lens)]
            metric.update(pred_paths=predicted_tags, label_paths=target)
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
    parser.add_argument("--markup", default="bios", type=str, choices=["bios", "bio"])
    parser.add_argument("--arch", default="bilstm_crf", type=str)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--embedding_dim", default=128, type=int)
    parser.add_argument("--hidden_dim", default=384, type=int)
    parser.add_argument("--task_name", default="ner", type=str)
    args = parser.parse_args()
    args.data_dir = config.data_dir
    args.id2label = {i: label for i, label in enumerate(config.label2id)}
    args.label2id = config.label2id
    if args.gpu != "":
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")
    args.output_dir = config.output_dir / '{}'.format(args.arch)
    if not args.output_dir.exists():
        args.output_dir.mkdir()

    init_logger(log_file=str(args.output_dir / "{}-{}.log".format(args.arch, args.task_name)))

    processor = CluenerProcessor(data_dir=config.data_dir)
    processor.get_vocab()
    model = BiLSTM_CRF(
        vocab_size=len(processor.vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        device=args.device,
        tag_dict=args.label2id,
        is_bert=False
    )
    model.to(args.device)
    if args.do_train:
        train(args, model, processor)
    if args.do_eval:
        model_path = args.output_dir / "best-model.bin"
        model = load_model(model, model_path=str(model_path))
        eval_log, class_info = evaluate(args, model, processor)
        print(eval_log)
        print("Eval Entity Score: ")
        for key, value in class_info.items():
            info = f"Subject: {key} - Acc: {value['precision']} - Recall: {value['recall']} - F1: {value['f1']}"
            logger.info(info)

if __name__ == "__main__":
    main()
