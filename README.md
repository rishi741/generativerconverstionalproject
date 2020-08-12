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
