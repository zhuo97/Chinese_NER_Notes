char_f = open("./giga_uni/gigaword_chn.all.a2b.uni.ite50.vec", "r")
word_f = open("./ctb50/ctb.50d.vec", "r")

output_f = open("./mix/mix_char_word.vec", "w")

line_no = 0
char_set = set()
for line in char_f:
    line_no += 1
    char_set.add(line.strip().split()[0])
    output_f.write(line.strip() + "\n")

for line in word_f:
    word = line.strip().split()[0]
    if word not in char_set and len(word) > 1:
        line_no += 1
        output_f.write(line.strip() + "\n")

char_f.close()
word_f.close()
output_f.close()

with open("./mix/mix_char_word.vec", "r+") as f:
    old = f.read()
    f.seek(0)
    f.write(str(line_no) + " " + str(50) + "\n")
    f.write(old)
