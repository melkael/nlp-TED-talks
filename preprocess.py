import collections
import urllib.request
import zipfile
import lxml.etree
import os
import re
import time
import numpy as np
from random import shuffle


def download_and_extract_ted():
    
    # Download the dataset if it's not already there
    if not os.path.isfile('ted_en-20160408.zip'):
        urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")

    with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
        doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
    texts = doc.xpath('//content/text()')
    labels = doc.xpath('//head/keywords/text()')
    del doc
    return texts, labels

# preprocess the texts: lowercase, remove text in parentheses, remove punctuation, tokenize into words (split on whitespace)
#removing text in parentheses

def preprocess_ted(texts):
    input_texts = [re.sub(r'\([^)]*\)', '', input_text) for input_text in texts]
	#lowercase
    input_texts = [input_text.lower() for input_text in input_texts]
	#remove punctuation
    input_texts = [re.sub(r'[^a-z0-9]+', ' ', input_text) for input_text in input_texts]
	#tokenize into words
    input_texts = [input_text.split() for input_text in input_texts]
    return input_texts

def clean_tokens_ted(input_texts):
    all_words = [word for input_text in input_texts for word in input_text]
    print("There are {} tokens in the dataset.".format(len(all_words)))
    all_words_counter = collections.Counter(all_words)

#remove some noise, take away the 100 most common and all words that only appear once
    most_common_50 = [word for word, count in all_words_counter.most_common(100)]
    only_once = [word for word, count in all_words_counter.most_common() if count == 1]
    print("There are {} tokens that appear only once.".format(len(only_once)))

    to_remove = set(only_once + most_common_50)
    print("There are {} unique tokens to remove.".format(len(to_remove)))

    start = time.time()
    input_texts = [[word for word in input_text if word not in to_remove] for input_text in input_texts]
    print("It took {} seconds to remove all unnecessary items.".format(time.time()-start))

    new_all_words = [word for input_text in input_texts for word in input_text]
    print("There are now only {} tokens in the dataset.".format(len(new_all_words)))

    return input_texts

def remove_short_texts(input_texts, labels):
    #remove all inputs that have less than 500 tokens in them
    inputs = zip(input_texts, labels)
    inputs = [text_and_labels for text_and_labels in inputs if len(text_and_labels[0]) > 300]
    print("There are now only {} inputs left.".format(len(inputs)))
    input_texts, labels = zip(*inputs)
    input_texts, labels = list(input_texts), list(labels)
    return input_texts, labels

def pad_texts(input_texts):
	#truncating every text to only the first 500 tokens
    l_max = 1000
    input_texts = [text[:l_max] for text in input_texts]
    input_texts = [(['<zero_pad>'] * (l_max - len(text)) + text) for text in input_texts]
    return input_texts

def preprocess_labels(labels):
	# preprocess the labels: search for occurences of the keywords "technology", "entertainment" or "design" and build labels
    label_lookup = ['ooo', 'Too', 'oEo', 'ooD', 'TEo', 'ToD', 'oED', 'TED']
    for i in range(len(labels)):
        ted_labels = ['o', 'o', 'o']
        keyword_list = labels[i].split(', ')
        if 'technology' in keyword_list:
            ted_labels[0] = 'T'
        if 'entertainment' in keyword_list:
            ted_labels[1] = 'E'
        if 'design' in keyword_list:
            ted_labels[2] = 'D'
        labels[i] = ''.join(ted_labels)
        labels[i] = label_lookup.index(labels[i])
    return labels, label_lookup


def compute_indexes(input_texts):
    # creating the unique vocabulary lookup
    vocab_list = list(set([word for input_text in input_texts for word in input_text]))
    word_to_index = {}
    index_to_word = {}
    for i, word in enumerate(vocab_list):
        word_to_index[word] = i
        index_to_word[i] = word
    input_indices_list = []
    for input_text in input_texts:
        input_indices_list.append([word_to_index[word] for word in input_text])
    
    return word_to_index, index_to_word, input_indices_list


def clean_vocabulary(word_to_index, glove):
    voc_len = len(word_to_index)
    print("vocabulary size: {} words".format(voc_len))
    counter = 0
    not_found_list = []
    embeddings = np.random.uniform(-.1, .1, size=(voc_len, 50))
    for word, index in word_to_index.items():
        if word in glove.vocab:
            counter += 1
            embeddings[index] = glove[word]
        elif word == '<zero_pad>':
            embeddings[index] = np.zeros(50)
        else:
            embeddings[index] = glove['unk']
            not_found_list.append(word)
    print("found {} word vectors, {} of our vocabulary".format(counter, float(counter)/voc_len))
    print("missing words e.g. {}".format(not_found_list[0:50]))
    return embeddings

#keep the class label distribution intact
# combining the tokens and labels for each input, then shuffle them and split into train/test/cv
def generate_datasets(input_indices_list, labels, label_lookup):

    inputs_combined = list(zip(input_indices_list, labels))
    inputs_train, inputs_test, inputs_cv = [], [], []
    for n in range(len(label_lookup)):
        inputs_of_curr_class = [inpu for inpu in inputs_combined if inpu[1] == n]
        l = len(inputs_of_curr_class)
        split1 = round(0.8*l)
        split2 = round(0.9*l)
        inputs_train.extend(inputs_of_curr_class[:split1])
        inputs_cv.extend(inputs_of_curr_class[split1:split2])
        inputs_test.extend(inputs_of_curr_class[split2:])

    shuffle(inputs_train)
    shuffle(inputs_cv)
    shuffle(inputs_test)
    print((len(inputs_train), len(inputs_test), len(inputs_cv)))
    return inputs_train, inputs_test, inputs_cv

# returns minibatched inputs
def get_data_batch(curr_index, batch_size, data):
    curr_batch = data[curr_index:curr_index+batch_size]
    input_batch_list, labels_batch_list = zip(*curr_batch) #unzip the list of input pair tuples (text, label)
        #print([len(text) for text in input_batch_list])
    curr_input_batch = np.array(input_batch_list, dtype=np.int32)
    curr_labels_batch = np.asarray(labels_batch_list)
    return curr_input_batch, curr_labels_batch