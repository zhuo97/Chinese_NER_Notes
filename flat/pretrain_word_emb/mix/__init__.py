import os
from gensim.models import KeyedVectors
from pathlib import Path

abs_path = Path(__file__).absolute().parent

if os.path.exists(os.path.join(abs_path, "mix_char_word.vec.bin")):
    model = KeyedVectors.load(os.path.join(abs_path, "mix_char_word.vec.bin"))
else:
    model = KeyedVectors.load_word2vec_format(os.path.join(abs_path, "mix_char_word.vec"))
    model.save(os.path.join(abs_path, "mix_char_word.vec.bin"))