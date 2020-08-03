# generativerconverstionalproject
converstionalchatbot
this chatbot includes data set from kaggle rdany chatbot,a robot.txt file of conversation of a bot with human.we have used around 117kb data for this chatbotnow i have inculded 5 .py file as message_prep.py,preprocessing.py,training_mode.py,test_model.py and chat.py
First i have use message_prep.py to open the robot.txt file that i have used for data set required for the project,in message_prep.py i have open the file than remove the '\n' from the file and than converted lines into pairs of two.
Now for this chatbot i have used seq2seq model
seq2seq networks have two parts:
An encoder that accepts language (or audio or video) input. The output matrix of the encoder is discarded, but its state is preserved as a vector.

A decoder that takes the encoder’s final state (or memory) as its initial state. We use a technique called “teacher forcing” to train the decoder to predict the following text (characters or words) in a target sequence given the previous text.

