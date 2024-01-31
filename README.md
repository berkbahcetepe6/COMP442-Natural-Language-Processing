# COMP442-Natural-Language-Processing
# COURSE DESCRIPTION
Natural Language Processing (NLP) is a subfield of artificial intelligence (AI) that is
concerned with modeling computers to understand, generate and interact with natural/human
language. This course will cover fundamental NLP tasks such as language modeling,
parsing, and dialogue, as well as modern-day models to solve these tasks. This is an
introductory, graduate-level course aimed at students willing to contribute to the NLP
literature in the near future. The main programming language will be Python, and we will
heavily depend on popular deep learning and NLP frameworks, e.g., Pytorch, Numpy, and
HF.
# COURSE OBJECTIVES
● To understand the foundational tasks and models for modern NLP,<br>
● To prepare students to conduct high-quality research in NLP,<br>
● To have students develop design and programming skills to solve practical NLP
problems,<br>
● To emphasize the challenges and open problems in NLP
# COURSE LEARNING OUTCOMES
● Understand fundamental NLP tasks such as morphological, syntactic, semantic,
discourse analysis/parsing; language modeling; question answering; natural language
generation; dialogue,<br>
● Understand the full NLP pipeline such as data collection, annotation, fundamental text
data preprocessing, model implementation, development, and test,<br>
● Understand and be able to develop fundamental NLP models such as RNNs,
attention mechanisms, and transformer-based language models,<br>
● Understand and get familiar with more recent NLP techniques such as in-context
learning

# Assignment 1
This assessment comprises two parts: "Fasttext" and "POS Tagging," focusing on word embeddings and their application in NLP tasks. Part 1 involves training word embeddings on a Turkish dataset using Fasttext, employing both Continuous Bag of Words and Skipgram methods. It includes environment setup, data downloading, preprocessing, model training, and analysis through functions like get_nearest_neighbors and get_analogies. Part 2 entails implementing a POS tagger using a BiLSTM network, covering data preprocessing, model implementation, training, and evaluation. Additionally, this part investigates the enhancement of the BiLSTM model's performance through the initialization with Fasttext embeddings. 

# Assignment 2
I engaged in a comprehensive exploration of seq2seq models, focusing on translating Turkish to English using the wmt16 dataset. The assignment involved building and experimenting with different seq2seq architectures, including basic models without attention, single-headed attention models, and multi-headed attention models. Key tasks included training these models using teacher forcing, decoding with beam search, and handling various technical aspects like attention scores and hidden state initialization. The models' performances were evaluated using the BLEU score metric on a test dataset, with the goal of understanding the efficacy of each seq2seq variant in the context of machine translation.

# Assignment 3
I engaged in two distinct parts, focusing on the fine-tuning and application of Transformer models for complex NLP tasks. In Part 1, I trained a Transformer to perform a knowledge-intensive task, initially without success. However, after pretraining on Wikipedia text, the model showed a significant improvement in accessing and utilizing world knowledge not present in the training data. Part 2 was dedicated to multi-task learning, where I used Hugging Face's tools to handle a variety of NLP tasks, such as textual similarity scoring, natural language entailment, and multiple-choice question-answering. This part involved building a flexible multi-task training scheme with a shared Transformer encoder but distinct task heads for each task, processing diverse data formats, and setting up a specialized DataLoader and Trainer. The culmination of this assignment was the successful training and evaluation of the model across different tasks, demonstrating its versatility and effectiveness in multi-task learning scenarios.
