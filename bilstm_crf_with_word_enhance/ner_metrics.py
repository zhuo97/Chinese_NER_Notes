from collections import Counter


def get_entity_bios(seq, id2tag):
    """
    :param seq:
    :param id2tag:
    :return:
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0, 1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for idx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2tag[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1: # "I" 后接 "S"
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = idx
            chunk[2] = idx
            chunk[0] = tag.split("-")[1]
            chunks.append(chunk)
            chunk = [-1, -1, -1]
        if tag.startswith("B-"):
            if chunk[2] != -1: # "I" 后接 "B"
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = idx
            chunk[0] = tag.split("-")[1]
        elif tag.startswith("I-") and chunk[1] != -1:
            _type = tag.split("-")[1]
            if _type == chunk[0]:
                chunk[2] = idx
            if idx == len(seq)-1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk) # "I" 后接 "O"
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id2tag):
    chunks = []
    chunk = [-1, -1, -1,]
    for idx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2tag[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = idx
            chunk[2] = idx
            chunk[0] = tag.split('-')[1]
            if idx == len(seq)-1: # 因为没有 "S"
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = idx

            if idx == len(seq)-1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq, id2tag, markup="bios"):
    assert markup in ["bios", "bio"]
    if markup == "bio":
        return get_entity_bio(seq, id2tag)
    else:
        return get_entity_bios(seq, id2tag)


class SeqEntityScore(object):
    def __init__(self, id2tag, markup="bios"):
        self.id2tag = id2tag
        self.markup = markup
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        precision = 0 if found == 0 else (right / found)
        recall = 0 if origin == 0 else (right / origin)
        f1 = 0. if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def update(self, label_paths, pred_paths):
        """
        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        """
        for label_path, pred_path in zip(label_paths, pred_paths):
            label_entities = get_entities(label_path, self.id2tag, self.markup)
            pre_entities = get_entities(pred_path, self.id2tag, self.markup)
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            precision, recall, f1 = self.compute(origin, found, right)
            class_info[type_] = {"precision": round(precision, 4),
                                 "recall": round(recall, 4),
                                 "f1": round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        precision, recall, f1 = self.compute(origin, found, right)
        return {"precision": precision, "recall": recall, "f1": f1}, class_info