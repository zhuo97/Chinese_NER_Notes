import config
import argparse
import torch
import torch.nn as nn
from metrics import SeqEntityScore
from dataset import NERDataset, TestNERDataset
from progressbar import ProgressBar
from model.bert_crf import BertCRF
from data_processor import CLUENERProcessor, MSRAProcessor
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
from utils import init_logger, logger


def load_and_cache_samples(args, processor, data_type):
    cached_dataset_file = args.cached_dir / "cached_crf-{}_{}_{}_{}".format(
        data_type,
        args.arch,
        args.task_name,
        args.dataset
    )
    if cached_dataset_file.exists():
        dataset = torch.load(cached_dataset_file)
    else:
        if data_type == "train":
            samples = processor.get_train_samples()
            dataset = NERDataset(samples, args.label2id, args.bert_model, args.device)
        elif data_type == "dev":
            samples = processor.get_dev_samples()
            dataset = NERDataset(samples, args.label2id, args.bert_model, args.device)
        torch.save(dataset, cached_dataset_file)

    return dataset


def train(args, model, processor):
    train_dataset = load_and_cache_samples(args, processor, "train")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=train_dataset.collate_fn)

    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    liner_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, "lr": args.learning_rate},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": args.learning_rate},

        {"params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, "lr": args.crf_learning_rate},
        {"params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": args.crf_learning_rate},

        {"params": [p for n, p in liner_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, "lr": args.crf_learning_rate},
        {"params": [p for n, p in liner_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": args.crf_learning_rate}
    ]
    # 需要了解 AdamW
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    train_size = len(train_dataset)
    train_step_per_eopch = train_size // args.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(args.epoch_num // 10) * train_step_per_eopch,
                                                num_training_steps=args.epoch_num * train_step_per_eopch)

    # 开始训练
    best_f1 = 0
    for epoch in range(args.epoch_num):
        print(f"Epoch {epoch}/{args.epoch_num}")
        pbar = ProgressBar(n_total=len(train_dataloader), desc="Training")
        train_loss = 0
        model.train()
        assert model.training
        for step, batch_samples in enumerate(train_dataloader):
            batch_word_ids, batch_attn_mask, batch_label_ids = batch_samples
            _, loss = model(batch_word_ids, 
                            batch_attn_mask,
                            batch_label_ids)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.clip_grad)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            train_loss += loss.item()
            pbar(step=step, info={"loss": loss.item()})
            
        train_log = {"loss": float(train_loss) / len(train_dataloader)}
        eval_log, class_info = evaluate(args, model, processor)
        logs = dict(train_log, **eval_log)
        show_info = f"\nEpoch: {epoch} - " + "-".join([f"{key}: {value:.4f} " for key, value in logs.items()])
        logger.info(show_info)
        
        cur_f1 = logs["eval_f1"]
        improved_f1 = cur_f1 - best_f1
        if improved_f1 > 1e-5:
            best_f1 = cur_f1
            model.save_pretrained(args.output_dir)
            logger.info("----- Save best model! -----")
        

def evaluate(args, model, processor):
    eval_dataset = load_and_cache_samples(args, processor, "dev")
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 collate_fn=eval_dataset.collate_fn)
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")

    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for step, batch_samples in enumerate(eval_dataloader):
            batch_word_ids, batch_attn_mask, batch_label_ids = batch_samples
            logits, loss = model(batch_word_ids, 
                                 batch_attn_mask,
                                 batch_label_ids)

            batch_predicted_label_ids = model.crf.decode(logits, batch_attn_mask)
            batch_predicted_labels = [[args.id2label[_id] for _id in predicted_label_ids[1:]] # remove CLS
                                      for predicted_label_ids in batch_predicted_label_ids]
            batch_label_lens = torch.sum(batch_attn_mask, dim=1).tolist()
            batch_label_ids = batch_label_ids.cpu().numpy()
            batch_target_labels = [[args.id2label[_id] for _id in label_ids[1:]] # remove CLS
                                   for label_ids in batch_label_ids]
            batch_target_labels = [target[:len_] 
                                   for target, len_ in zip(batch_target_labels, batch_label_lens)]
            metric.update(pred_paths=batch_predicted_labels, label_paths=batch_target_labels)
            eval_loss += loss.item()
            pbar(step=step, info={"loss": loss.item()})

    eval_info, class_info = metric.result()
    eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    result = {"eval_loss": float(eval_loss) / len(eval_dataloader)}
    result = dict(result, **eval_info)
    return result, class_info


def predict(args, model, processor):
    predict_samples = processor.get_test_samples()
    predict_dataset = TestNERDataset(predict_samples, args.bert_model, args.device)
    predict_dataloader = DataLoader(predict_dataset,
                                    batch_size=args.batch_size,
                                    collate_fn=predict_dataset.collate_fn)
    pbar = ProgressBar(n_total=len(predict_dataloader), desc="Predicting")

    model.eval()
    test_results = []
    with torch.no_grad():
        for step, batch_samples in enumerate(predict_dataloader):
            batch_words, batch_word_ids, batch_attn_mask = batch_samples
            logits = model(batch_word_ids, batch_attn_mask)[0]
            batch_predicted_label_ids = model.crf.decode(logits, batch_attn_mask)
            batch_predicted_labels = [[args.id2label[_id] for _id in predicted_label_ids[1:]] # remove CLS
                                      for predicted_label_ids in batch_predicted_label_ids]
            pbar(step=step)
            
            for words, labels in zip(batch_words, batch_predicted_labels):
                sample_dict = {}
                sample_dict["words"] = words
                sample_dict["labels"] = labels
                test_results.append(sample_dict)

    with open(args.predict_dir, "w") as f:
        for sample in test_results:
            f.write("".join(sample["words"]) + "\t" + ",".join(sample["labels"]) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_eval", default=False, action="store_true")
    parser.add_argument("--do_predict", default=False, action="store_true")
    parser.add_argument("--markup", default="bios", type=str, choices=["bios", "bio"])
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--crf_learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epoch_num", default=50, type=int)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--clip_grad", default=5.0, type=float)
    parser.add_argument("--arch", default="bert_crf", type=str)
    parser.add_argument("--task_name", default="ner", type=str)
    parser.add_argument("--dataset", default="msra", type=str)
    args = parser.parse_args()

    if args.dataset == "cluener":
        args.data_dir = config.cluener_data_dir
        args.predict_dir = config.cluener_predict_dir
        args.log_dir = config.log_dir
        args.output_dir = config.cluener_output_dir
        if not args.output_dir.exists():
            args.output_dir.mkdir()
        args.label2id = config.cluner_label2id
        args.id2label = {_id: _label for _label, _id in args.label2id.items()}

        processor = CLUENERProcessor(data_dir=args.data_dir)
    elif args.dataset == "msra":
        args.data_dir = config.msra_data_dir
        args.predict_dir = config.msra_predict_dir
        args.log_dir = config.log_dir
        args.output_dir = config.msra_output_dir
        if not args.output_dir.exists():
            args.output_dir.mkdir()
        args.label2id = config.msra_label2id
        args.id2label = {_id: _label for _label, _id in args.label2id.items()}

        processor = MSRAProcessor(data_dir=args.data_dir)

    args.bert_model = config.bert_model
    if args.gpu != "":
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")

    args.cached_dir = config.cached_dir
    init_logger(log_file=str(args.log_dir / "{}-{}-{}.log".format(args.arch,
                                                                  args.task_name,
                                                                  args.dataset)))

    if args.do_train:
        model = BertCRF.from_pretrained(config.bert_model, num_labels=len(args.label2id))
        model.to(args.device)

        train(args, model, processor)

    if args.do_eval:
        model = BertCRF.from_pretrained(args.output_dir)
        model.to(args.device)

        evaluate(args, model, processor)

    if args.do_predict:
        model = BertCRF.from_pretrained(args.output_dir)
        model.to(args.device)

        predict(args, model, processor)


if __name__ == "__main__":
    main()