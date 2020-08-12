# generativerconverstionalproject
This project is based upon a conversational chatbot .At first i have used DATASET from a danycahtbot through kaggle,where it has data collected from various chats on danychatbot between human and robot ,so i have used a file called robot.txt from kaggale ,it has around 2000 replies by a chatbot which, i have used as my data.
Now for first part i have used message_prep.py where i have just converted the message of robot.txt into a group of two lines.
now for second part i have used preprocessing.py in which i have used two list , input_doc and target_doc , where i have line in pairs append to them such first line gets through input_docs and second line  in pairs goes adds <start> and <end> then append  to target_doc.now i hav created two sets input_token and target_token whhere i append toen from input_doc and target_doc and used some regrex there.now added a feature dictionary for input_token,a features dictionary for 
 target_token,a reverse features dictionary for input_token (where the keys and values are swapped),a reverse features dictionary for target_token.
  p, it’s time to vectorize the data! We’re going to need vectors to input into our encoder and decoder, as well as a vector of target data we can use to train the decoder.

Because each matrix is almost all zeros, i used numpy.zeros() from the NumPy library to build them out.

We defined a NumPy matrix of zeros called encoder_input_data with two arguments:

the shape of the matrix — in my case the number of documents (or sentences) by the maximum token sequence length (the longest sentence we want to see) by the number of unique tokens (or words)
the data type we want — in my case NumPy’s float32, which can speed up the processing a bit.


At this point i   filled out the 1s in each vectors
the vectors have timesteps — i used these to track where in a given document (sentence) i am in.

To build out a three-dimensional NumPy matrix of one-hot vectors, we can assign a value of 1 for a given word at a given timestep in a given line:

matrix_name[line, timestep, features_dict[token]] = 1.
Keras will fit — or train — the seq2seq model using these matrices of one-hot vectors:

the encoder input data
the decoder input data
the decoder target data

The reason i have used two decoder data is  to do with a technique known as teacher forcing that most seq2seq models employ during training. Here’s the idea: i have a  input token from the previous timestep to help train the model for the current timestep’s target token
now for training_model.py
Deep learning models in Keras are built in layers, where each layer is a step in the model.

the encoder requires two layer types from Keras:

An input layer, which defines a matrix to hold all the one-hot vectors that is feeded to the model.
An LSTM layer, with some output dimensionality.
We can import these layers as well as the model we need like so:

from keras.layers import Input, LSTM
from keras.models import Model
Next, i set up the input layer, which requires some number of dimensions that i have used In this casee the code is written to handle varying batch sizes, so  don’t need to specify that dimension.

# the shape specifies the input matrix sizes
encoder_inputs = Input(shape=(None, num_encoder_tokens))
For the LSTM layer,  to select the dimensionality (the size of the LSTM’s hidden states, which helps determine how closely the model molds itself to the training data — something we can play around with) and whether to return the state (in this case we do):

encoder_lstm = LSTM(100, return_state=True)
#  dimensionality of 256
# so any LSTM output matrix will have 
# shape [batch_size, 256]
 the only thing that require  from the encoder is its final states. We can get these by linking  LSTM layer with  input layer:

encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_outputs isn’t really important for us, so we can just discard it. However, the states, we’ll save in a list:

encoder_states = [state_hidden, state_cell]

Decoder Training Setup
The decoder looks a lot like the encoder (phew!), with an input layer and an LSTM layer that we use together:

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
# This time we care about full return sequences
However, with our decoder, i passed in the state data from the encoder, along with the decoder inputs. This time, the output is benefical  instead of the states:

# The two states will be discarded for now
decoder_outputs, decoder_state_hidden, decoder_state_cell = 
    decoder_lstm(decoder_inputs, initial_state=encoder_states)
We also need to run the output through a final activation layer, using the Softmax function, that will give us the probability distribution — where all probabilities sum to one — for each token. The final layer also transforms our LSTM output from a dimensionality of whatever we gave it (in our case, 10) to the number of unique words within the hidden layer’s vocabulary (i.e., the number of unique target tokens, which is definitely more than 10!).

decoder_dense = Dense(num_decoder_tokens, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)
Keras’s implementation could work with several layer types, but Dense is the least complex, so we’ll go with that. We also need to modify our import statement to include it before running the code:

from keras.layers import Input, LSTM, Dense

now for chat.py
chatbot i have taken   decode_sequence() from test_mode.py and rename  to .generate_response(). As a reminder, this is where response generation and selection take place:

The encoder model encodes the user input
The encoder model generates an embedding (the last hidden state values)
The embedding is passed from the encoder to the decoder
The decoder generates an output matrix of possible words and their probabilities
I used NumPy to help us choose the most probable word (according to the model)
the chosen word gets translated back from a NumPy matrix into human language and added to the output sentence
now in chat.py , i have used some rule based model also, i have used regrex to check the intent of user_response if the intent of user_response match with the regrex pattern it will follow the rule based patern , else it will genearte random responses using deep learning training.
