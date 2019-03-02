# Deep NLP practicals

These are my answers to the deep NLP practicals from the Oxford course (available on github).

##### The goal is to classify the ted talks by their categories using the text transcripts.

To execute simply launch the ipython notebook.
It uses the preprocess.py file, which contains all the logic to download the Ted Talks and create the word embeddings (Word2Vec). Then the model processes the Ted Talks texts in order to classify them.

I achieved up to 67% accuracy which is pretty decent, although the data is not really well distributed. A seq2seq model would probably achieve better results.
