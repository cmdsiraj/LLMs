import string

# fucntion to assing tags for unkown words
def assign_unk(word):

    punctuations = set(string.punctuation)

    noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
    verb_suffix = ["ate", "ify", "ise", "ize"]
    adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
    adv_suffix = ["ward", "wards", "wise"]

    if any(char.isdigit() for char in word):
        return "--unk_digit--"
    elif any(char.isupper() for char in word):
        return "--unk_upper--"
    elif any(char in punctuations for char in word):
        return "--unk_punct--"
    elif any(word.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"
    elif any(word.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"
    elif any(word.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"
    elif any(word.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"
    else:
        return "--unk--"



def get_word_tag(line, vocab):
    if not line.split():
        word = "--n--"
        tag = "--s--"
    else:
        word, tag = line.split()

        if word not in vocab:
            word = assign_unk(word)
    return word, tag



def preprocessing(vocab, data_fp):
    orig = []
    prep = []

    with open(data_fp, "r") as data_file:
        for cnt, word in enumerate(data_file):

            if not word.split():
                orig.append(word.strip())
                word = "--n--"
                prep.append(word)
                continue

            elif word.strip() not in vocab:
                orig.append(word.strip())
                word = assign_unk(word)
                prep.append(word)
                continue

            else:
                orig.append(word.strip())
                prep.append(word.strip())
    assert(len(orig) == len(open(data_fp, "r").readlines()))
    assert(len(prep) == len(open(data_fp, "r").readlines()))

    return orig, prep


