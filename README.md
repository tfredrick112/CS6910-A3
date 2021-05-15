# CS6910-A3

## Authors: Dhruvjyoti Bagadthey EE17B156, D Tony Fredrick EE17B154


1. Install the required libraries in your environment using this command:

`
pip install -r requirements.txt
`

2. **IMPORTANT: To Check code correctness use the notebooks Seq2Seq_NoWANDB_BestModel.ipynb and Attention_without_WANDB.ipynb ONLY**.

3. To train a Seq2Seq model for Dakshina Dataset transliteration from English to Hindi, use the notebook: **Seq2Seq_With_WANDB.ipynb**.
  
4. To train a Seq2Seq model with Attention for Dakshina Dataset transliteration from English to Hindi, use the notebook: **Attention_with_WANDB.ipynb**.

Note: Wherever you need to log to wandb, please remember to change the name of the entity and project in the corresponding line of code.

### Link to the project report:

https://wandb.ai/ee17b154tony/dl_assignment_3/reports/CS6910-Assignment-3-Report--Vmlldzo2NzU5MzI

### General Framework:

ALl our notebooks have been created in Google Colab with a GPU backend. We have used TensorFlow and Keras for defining, training and testing our model.

### Vanilla Seq2Seq model:
### Attention Seq2Seq model:

`
extract_data_info(data_dict)
`
: Returns important information about the data like input characters, target characters,maximum sequence lengths etc

`
make_one_hot_arrays(data_dict, max_encoder_seq_length, max_decoder_seq_length, num_input_tokens, num_target_tokens)
`
: This function takes the training/validation/test dictionary as input and produces the one-hot encoded versions of the respective data.

`
class AttentionLayer(Layer)
`
: This class implements Bahdanau attention and creates a layer called attention that can be integrated with keras very easily(https://arxiv.org/pdf/1409.0473.pdf). A major part of the code has been borrowed from https://github.com/thushv89/attention_keras

`
define_model(num_cells, cell_type, input_embedding_size, dropout_fraction, beam_size)
`
: Defines a single layer encoder, single layer decoder Seq2Seq model with Bahdanau attention. 

num_cells: Number of cells in the encoder and decoder layers

cell_type: choice of cell type: Simple RNN, LSTM, GRU

num_encoder_layers: Number of layers in the encoder

num_decoder_layers: Number of layers in the decoder

input_embedding_size: Dimenions of the vector to represent each character

dropout_fraction: fraction of neurons to drop out

The best model obtained was:

![image](https://user-images.githubusercontent.com/62587866/118363816-c8594880-b5b3-11eb-9a53-861e4167b793.png)

`
prepare_inference_model_lstm_1(model, num_cells)
`
: Takes in a model that has the cell_type = 'LSTM' and converts into an inference model. ie it reorders the connections of a model defined by the above function and trained using teacher forcing. returns the dismantled encoder and the decoder

`
transliterate_word_lstm_1(input_words, encoder_model, decoder_model)
`
: Decodes the given input sequence in batches using the encoder and the decoder models returned by prepare_inference_model_lstm_1

`
prepare_inference_model_rnngru_1(model, num_cells)
`
: Takes in a model that has the cell_type = 'RNN' or 'GRU' and converts into an inference model. ie it reorders the connections of a model defined by the above function and trained using teacher forcing. returns the dismantled encoder and the decoder

`
transliterate_word_rnngru_1(input_words, encoder_model, decoder_model)
`
: Decodes the given input sequence in batches using the encoder and the decoder models returned by prepare_inference_model_rnngru_1

`
train_with_wandb()
`
: Trains, validates the model on the data and logs the accuracies and losses into wandb. The characterwise validation accuracy with teacher forcing is logged per epoch. The inference validation accuracy without teacher forcing is logged after the complete training phase.

`
train(num_cells, cell_type, input_embedding_size, dropout_fraction, batch_size,epochs)
`
: Trains, validates the model on the data. The characterwise validation accuracy with teacher forcing is plotted per epoch. The inference validation accuracy without teacher forcing is printed after the complete training phase.

`
transliterate_word_rnngru_attn_out(input_word,encoder_model,decoder_model)
`
: Decodes the given input word, one character at a time. Returns the attention maps for that word. The encoder-decoder models must be RNN/GRU

The Attention matrices are:

![image](https://user-images.githubusercontent.com/62587866/118363981-4fa6bc00-b5b4-11eb-8d85-a877ee582dfd.png)


### Attention Seq2Seq model other utilities:
This part contains the unused transliterate_word_lstm_attn_out function and a code that takes in the attention maps,the input and target words and produces a .gif for connectivity visualisation.

The gifs are present in a zip file. 

`
transliterate_word_lstm_attn_out(input_word,encoder_model,decoder_model)
`
: Decodes the given input word, one character at a time. Returns the attention maps for that word. The encoder-decoder models must be LSTM



