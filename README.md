# Language Translation

Creating English-to-Hindi **language translation model** using ***neural machine translation*** with ***seq2seq architecture***.

---

## Background

* **Encoder-Decoder LSTM model** having ***seq2seq*** architecture can be used to solve ***many-to-many sequence problems***, where both inputs and outputs are divided over multiple time-steps.

* The **seq2seq architecture** is a type of *many-to-many sequence modeling*, and is commonly used for a variety of tasks:
  * Text Summarization
  * Chatbot Development
  * Conversational Modeling
  * **Neural Machine Translation**

---

## Dependencies

* TensorFlow  
* Keras  
* NumPy  

---

## Dataset

* Kaggle Dataset: [Hindi-English Parallel Corpus](https://www.kaggle.com/datasets/vaibhavkumar11/hindi-english-parallel-corpus)  
* Download using:
  ```bash
  !kaggle datasets download -d vaibhavkumar11/hindi-english-parallel-corpus
---

## Architecture

* The **Neural Machine Translation** model is based on **Seq2Seq architecture**, which is an **Encoder-Decoder architecture**.
* It consists of two layers of **LSTM networks**:
  * **Encoder** LSTM
    - Input = Sentence in English
    - Output = Hidden and cell state
  * **Decoder** LSTM
    - Input = `<sos>` + translated sentence
    - Output = Translated sentence ending with `<eos>`

---

## Data Preprocessing

* No preprocessing for input English sentences.
* Translated Hindi sentence requires:
  * One copy with `<sos>` token at the start
  * One copy with `<eos>` token at the end

---

## Tokenization

* Tokenize input sentences (English)
  * Convert words to integers
  * Create word-to-index dictionary
  * Count unique input words
  * Determine max input sentence length
* Tokenize output sentences (Hindi)
  * Convert words to integers
  * Create word-to-index dictionary
  * Count unique output words
  * Determine max output sentence length

---

## Padding

* LSTMs expect fixed-length input:
  * **Input (Encoder)**: Pad at beginning (`pre`)
  * **Output (Decoder)**: Pad at end (`post`)

---

## Word Embeddings

* Convert words to vector representations
* Use pretrained **GloVe embeddings**:
  * File: `data/glove.6B.100d.txt`
  * Load into dictionary: `{word: vector}`
  * Create embedding matrix where:
    - Row index = word index
    - Column = embedding dimensions

---

## Create the Model

* Embedding Layer
* Decoder Output:
  * Shape: `(num_samples, max_len_output, vocab_size_output)`
* One-Hot encode decoder output
* Encoder:
  * Input: English sentence
  * Output: Hidden & Cell state
* Decoder:
  * Input: `<sos>` + previous hidden and cell state
  * Output: sequence of predictions
* Final Layer:
  * Dense layer with softmax activation

---

## Summary of the Model

* `input_1`: Encoder input (English)
* `lstm_1`: Encoder LSTM → outputs hidden & cell states
* `input_2`: Decoder input (Hindi with `<sos>`)
* `lstm_2`: Decoder LSTM → consumes hidden & cell states from encoder
* Decoder LSTM output → Dense layer → word predictions

---

## Modifying Model for Prediction

* At prediction time, true decoder inputs aren't known
* Prediction steps:
  1. Encode the input sentence
  2. Start decoder with `<sos>`
  3. Predict first word
  4. Use predicted word as next decoder input
  5. Repeat until `<eos>` is generated

---

## Create the Prediction Model

* Encoder remains the same
* Decoder modified to take 1 word at a time
* Inputs:
  * Previous word
  * Hidden state
  * Cell state
* Outputs:
  * Predicted word
  * Updated hidden and cell states

---

## Summary of the Prediction Model

* `input_5`: One-word decoder input of shape `(None, 1)`
* `input_3`, `input_4`: Hidden and cell states
* Output of LSTM → Dense → prediction

---

## Make Predictions

* Create reverse token dictionaries:
  * `{idx: word}` for input and output
* Process:
  1. Tokenize & pad English sentence
  2. Encode using encoder model
  3. Initialize with `<sos>`
  4. Loop:
    * Predict next word
    * Append to output list
    * Stop if `<eos>` or max length reached
  5. Convert integer outputs to words
  6. Return final sentence

---

## Test

* Randomly select English sentence from dataset
* Tokenize & pad
* Pass through prediction pipeline
* Display input → predicted translation

---

## Improvements

* Current training: 5 epochs (due to hardware)
  - Can increase for better accuracy
  - Watch out for overfitting
* Ways to improve:
  - Add more training data
  - Use dropout
  - Tune architecture & parameters

---

## Conclusion

* **Neural Machine Translation** is a complex but powerful NLP application.
* It uses:
  - **Seq2Seq architecture**
  - **Encoder-Decoder** model with **LSTMs**
* Encoder encodes source language
* Decoder generates target language word by word

---

## Advantages and Disadvantages

**Advantages**:
* Good at mapping sequences
* Learns sentence structures

**Disadvantages**:
* **Vanilla seq2seq lacks context**
* Not suitable for advanced conversational AI
* For better performance in chatbots:
  - Use **Attention mechanisms**
  - Or use **Transformers**

---
