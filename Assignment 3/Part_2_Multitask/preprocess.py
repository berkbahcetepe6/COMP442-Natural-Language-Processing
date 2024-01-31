import transformers
import torch

max_length = 128

def convert_to_stsb_features(example_batch, model_name="roberta-base"):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    sentence1, sentence2 = example_batch['sentence1'], example_batch['sentence2']
    
    # Join the sentences together as specified in the document
    ################
    # Your code here
    ################
    inputs = list(zip(sentence1, sentence2))
    
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, padding='max_length', truncation=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_rte_features(example_batch, model_name="roberta-base"):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    # Join the sentences together as specified in the document
    ################
    sentence1, sentence2 = example_batch['sentence1'], example_batch['sentence2']
    inputs = list(zip(sentence1, sentence2))
    ################
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, padding='max_length', truncation=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_commonsense_qa_features(example_batch, model_name="roberta-base"):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    num_examples = len(example_batch["question"])
    num_choices = len(example_batch["choices"][0]["text"])
    labels2id = {char: i for i, char in enumerate("ABCDE")}
    features = {}
    
    """
    For each question:
        1) Join the same question with every answer key text. For example,
            [('Sammy wanted to go to where the people were.  Where might he go?', 'race track'),
            ('Sammy wanted to go to where the people were.  Where might he go?', 'populated areas'),
            ('Sammy wanted to go to where the people were.  Where might he go?', 'the desert'),
            ('Sammy wanted to go to where the people were.  Where might he go?', 'apartment'),
            ('Sammy wanted to go to where the people were.  Where might he go?', 'roadblock')]
        2) Then encode using batch_encode_plus
        3) Save the encoded data for corresponding keys in lists of lists for each batch
        4) Convert the answerkey to integer mapping (0 to 5) and save under the key 'labels' in features
           If answerKey does not exist, set them all to zeros
    """
    for example_i in range(num_examples):
        
        ################
        question = example_batch["question"][example_i] * num_choices
        choices = example_batch["choices"][example_i]["text"]
        ################
        inputs = [(question, choice) for choice in choices]
        choices_inputs = tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length, padding='max_length', truncation=True,
        )
        ################
        for key, value in choices_inputs.items():
            if key not in features:
                features[key] = []
            features[key].append(value)

        if 'answerKey' in example_batch and example_batch["answerKey"][example_i] in labels2id:
            features.setdefault('labels', []).append(labels2id[example_batch["answerKey"][example_i]])
        else:
            features.setdefault('labels', []).append([0]*num_choices)

    for key, value in features.items():
        features[key] = value.cuda()
        ################
    
    return features
