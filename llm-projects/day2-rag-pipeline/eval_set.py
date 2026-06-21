# 20 question-answer pairs for the Attention Is All You Need paper
# ground_truth = the actual correct answer, used to score faithfulness/relevance

eval_questions = [
    {
        "question": "What optimizer was used to train the model?",
        "ground_truth": "The Adam optimizer was used with β1=0.9, β2=0.98, and ε=10^-9."
    },
    {
        "question": "Who are the authors of this paper?",
        "ground_truth": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin."
    },
    {
        "question": "What is multi-head attention?",
        "ground_truth": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions, using multiple attention heads in parallel."
    },
    {
        "question": "What is the Transformer architecture based on?",
        "ground_truth": "The Transformer is based entirely on attention mechanisms, dispensing with recurrence and convolutions entirely."
    },
    {
        "question": "What is self-attention?",
        "ground_truth": "Self-attention, also called intra-attention, is an attention mechanism relating different positions of a single sequence to compute a representation of that sequence."
    },
    {
        "question": "How many layers does the encoder have?",
        "ground_truth": "The encoder is composed of a stack of N=6 identical layers."
    },
    {
        "question": "How many layers does the decoder have?",
        "ground_truth": "The decoder is also composed of a stack of N=6 identical layers."
    },
    {
        "question": "What is positional encoding used for?",
        "ground_truth": "Since the model has no recurrence or convolution, positional encodings are added to give the model information about the relative or absolute position of tokens in the sequence."
    },
    {
        "question": "What type of positional encoding did they use?",
        "ground_truth": "They used sine and cosine functions of different frequencies for positional encoding."
    },
    {
        "question": "What is the dimensionality of the model (d_model)?",
        "ground_truth": "The model dimension d_model is 512."
    },
    {
        "question": "How many attention heads are used?",
        "ground_truth": "The model uses 8 parallel attention heads (h=8)."
    },
    {
        "question": "What datasets were used for training?",
        "ground_truth": "The model was trained on the WMT 2014 English-German and WMT 2014 English-French datasets."
    },
    {
        "question": "What hardware was used for training?",
        "ground_truth": "The models were trained on one machine with 8 NVIDIA P100 GPUs."
    },
    {
        "question": "What is the purpose of the feed-forward networks in each layer?",
        "ground_truth": "Each layer contains a fully connected feed-forward network applied to each position separately and identically, consisting of two linear transformations with a ReLU activation in between."
    },
    {
        "question": "What regularization techniques were used?",
        "ground_truth": "They used residual dropout and label smoothing during training."
    },
    {
        "question": "What is the BLEU score achieved on English-to-German translation?",
        "ground_truth": "The big Transformer model achieved a BLEU score of 28.4 on the WMT 2014 English-to-German translation task."
    },
    {
        "question": "Why did they choose attention over recurrence?",
        "ground_truth": "Attention mechanisms allow modeling dependencies without regard to their distance in input or output sequences, and enable significantly more parallelization than recurrent models."
    },
    {
        "question": "What is scaled dot-product attention?",
        "ground_truth": "Scaled dot-product attention computes the dot products of the query with all keys, divides each by the square root of the dimension, and applies a softmax to obtain weights on the values."
    },
    {
        "question": "How long did the base model take to train?",
        "ground_truth": "The base models were trained for a total of 100,000 steps, taking 12 hours."
    },
    {
        "question": "What is the vocabulary size used?",
        "ground_truth": "They used a shared source-target vocabulary of about 37000 tokens for English-German."
    },
]

print(f"Total eval questions: {len(eval_questions)}")