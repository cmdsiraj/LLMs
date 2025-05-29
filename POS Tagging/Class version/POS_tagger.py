from nltk.corpus import brown
import numpy as np
import pandas as pd
import math
from collections import defaultdict
import string
import nltk

class POS_tagger:

    def __init__(self, universal=True, alpha=0.001):
        if universal:
            self.tagged_sentences = list(brown.tagged_sents(tagset="universal"))
        else:
            self.tagged_sentences = list(brown.tagged_sents())
        
        self.vocab = self.__extract_vocab__(self.tagged_sentences)
        self.emission_counts, self.transition_counts, self.tag_counts = self.__create_dictionaries__(train_l=self.tagged_sentences, vocab=self.vocab)
        self.states = sorted(self.tag_counts.keys())
        self.transition_matrix = self.__create_transition_matrix__(self.transition_counts, alpha=alpha, tag_counts=self.tag_counts)
        self.emission_matrix = self.__create_emission_matrix__(emission_counts=self.emission_counts, alpha=alpha, tag_counts=self.tag_counts, vocab=list(self.vocab))
        


    def __extract_vocab__(self, data):

        freq = defaultdict(int)
        for list in data:
            for word_tag in list:
                freq[word_tag[0]]+=1
            freq["\n"]+=1
        
        vocab_l = [k for k,v in freq.items() if v>1]

        vocab_l.extend(["--unk_upper--", "--unk_digit--", "--unk_punct--", "--unk_noun--", "--unk_verb--", "--unk_adj--", "--unk_adv--", "--unk--", "--n--"])

        vocab = {}

        for i, word in enumerate(sorted(vocab_l)):
            vocab[word] = i
        
        return vocab
    


    def __assign_unk__(self, word):

        punctuations = string.punctuation

        noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
        verb_suffix = ["ate", "ify", "ise", "ize"]
        adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
        adv_suffix = ["ward", "wards", "wise"]

        if any(char.isupper() for char in word):
            return "--unk_upper--"
        elif any(char.isdigit() for char in word):
            return "--unk_digit--"
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
        
        return "--unk--"
    


    def __get_word_tag__(self, line, vocab):

        if line[0] == '\n':
            word = "--n--"
            tag = "--s--"
            return word, tag
        
        else:
            word = line[0]
            tag = line[1]
            if word not in vocab:
                word = self.__assign_unk__(word)
            
            return word, tag
        


    def __create_dictionaries__(self, train_l, vocab):

        emission_count = defaultdict(int)
        transitions_count = defaultdict(int)
        tag_count = defaultdict(int)

        prev_tag = "--s--"

        for line in train_l:
            word, tag = self.__get_word_tag__(("\n", "\n"), vocab)
            emission_count[(tag, word)] += 1
            transitions_count[(prev_tag, tag)] += 1
            tag_count[tag] += 1
            prev_tag = tag
            for word_tag in line:

                word, tag = self.__get_word_tag__(word_tag, vocab)

                emission_count[(tag, word)] += 1

                transitions_count[(prev_tag, tag)] += 1

                tag_count[tag] += 1

                prev_tag = tag
        
        return emission_count, transitions_count, tag_count
    


    def __input_preprocess__(self, input, vocab):
        prep = []

        sentences = input.strip().split("\n")
        words = []
        for sentence in sentences:
            words.extend(nltk.word_tokenize(sentence))
            words.append("\n")

        for word in words:
            if word == "\n":
                word="--n--"
            else:
                if word not in vocab:
                    word = self.assign_unk(word)
                prep.append(word)
        return prep
    


    def __create_transition_matrix__(self, transition_counts, alpha, tag_counts):
    
        all_tags = sorted(tag_counts.keys())
        num_tags = len(all_tags)

        transition_matrix = np.zeros((num_tags, num_tags))

        trans_key = set(transition_counts.keys())

        for i in range(num_tags):
            for j in range(num_tags):
                count = 0

                key = (all_tags[i], all_tags[j])

                if key in trans_key:
                    count = transition_counts[key]
                
                transition_matrix[i, j] = (count+alpha)/(tag_counts[all_tags[i]] + num_tags*alpha)
        
        return transition_matrix
    


    def __create_emission_matrix__(self, emission_counts, alpha, tag_counts, vocab: list):
    
        all_tags = sorted(tag_counts.keys())
        num_tags = len(all_tags)

        num_words = len(vocab)

        emission_matrix = np.zeros((num_tags, num_words))

        emis_keys = set(emission_counts.keys())

        for i in range(num_tags):
            for j in range(num_words):
                count = 0
                
                key = (all_tags[i], vocab[j])

                if key in emis_keys:
                    count = emission_counts[key]
                
                emission_matrix[i, j] = (count+alpha)/(tag_counts[all_tags[i]]+alpha*num_words)
        
        return emission_matrix
    

    def __viterbi_initialize__(self, A, B, vocab, tag_counts, states, corpus):

        num_tags = len(tag_counts.keys())

        best_probs = np.zeros((num_tags, len(corpus)))

        best_paths = np.zeros((num_tags, len(corpus)), dtype=int)

        s_indx = states.index("--s--")

        for i in range(num_tags):

            if A[s_indx, i] == 0:
                best_probs[i,0] = float('-inf')
            else:
                best_probs[i,0] = math.log(A[s_indx, i])+math.log(B[i, vocab[corpus[0]]])
        
        return best_probs, best_paths



    def __viterbi_forward__(self, best_probs, best_paths, A, B, vocab, corpus):
        
        num_tags = best_probs.shape[0]

        for i in range(1, len(corpus)):
            
            for j in range(num_tags):
                best_prob_i = float("-inf")
                best_path_i = None

                for k in range(num_tags):
                    prob = best_probs[k, i-1] + math.log(A[k, j]) + math.log(B[j, vocab[corpus[i]]])

                    if prob > best_prob_i:
                        best_prob_i = prob
                        best_path_i = k
                    
                best_probs[j, i] = best_prob_i
                best_paths[j, i] = best_path_i
        
        return best_probs, best_paths
    

    def __viterbi_backward__(self, best_probs, best_paths, states):

        num_words = best_probs.shape[1]

        z = np.zeros(num_words, dtype=int)

        pred = [None] * num_words

        last_tag_index = np.argmax(best_probs[:,num_words-1])
        z[num_words-1] = last_tag_index
        pred[num_words-1] = states[last_tag_index]

        for i in range(num_words-1, 0, -1):
            z[i-1] = best_paths[z[i], i]
            pred[i-1] = states[z[i-1]]
        
        return pred
    
    def tag(self, input: str):

        input = self.__input_preprocess__(input, self.vocab)

        best_probs, best_paths = self.__viterbi_initialize__(self.transition_matrix, self.emission_matrix, vocab=self.vocab, tag_counts=self.tag_counts, states=self.states, corpus=input)
        best_probs, best_paths = self.__viterbi_forward__(best_probs=best_probs, best_paths=best_paths, A=self.transition_matrix, B=self.emission_matrix, vocab=self.vocab, corpus=input)
        predictions = self.__viterbi_backward__(best_probs=best_probs, best_paths=best_paths, states=self.states)

        word_tag_predictions=[]

        for word, tag in zip(input, predictions):
            word_tag_predictions.append((word, tag))
        
        return word_tag_predictions