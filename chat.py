import numpy as np
import re
from test_model import encoder_model, decoder_model, num_decoder_tokens, num_encoder_tokens, input_features_dict, target_features_dict, reverse_target_features_dict, max_decoder_seq_length, max_encoder_seq_length

class chatbot:
    negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")
    def __init__(self):
        self.intent_chatbot = {'describe_system_intent': r'.*\s*your system.*','area_intent': r'.*area.square.*(\d+)'}
    def start_chat(self):
        user_response = input("Hi,what's your name\n")
        if user_response in self.negative_responses:
            print("cool!..Bye")
            return
        reply = input("hey,{} i am rdany(r means robot) chatbot,tell about yourself\n".format(user_response))
        self.match(reply)
    def make_exit(self,reply):
        if exit in self.exit_commands:
            if exit in reply:
                return True
            return False
    def match(self,reply):
        for key,value in self.intent_chatbot:
            intent = key
            regrex_pattern = value
            found_match = re.match(regrex_pattern,reply)
            if found_match and intent == 'describe_system_intent':
                return self.describe_system_intent()
            elif found_match and intent == "area_intent":
                return self.area_intent(found_match.groups()[0])
        self.chat(reply)
    def string_to_matrix(self,reply):
        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
        user_input_matrix = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        for timestep, token in enumerate(tokens):
            # add an if clause to handle user input:
            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1.
        return user_input_matrix
    def generate_response(self,reply):
        test_input = self.string_to_matrix(reply)
        # Encode the input as state vectors.
        states_value = encoder_model.predict(test_input)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first token of target sequence with the start token.
        target_seq[0, 0, target_features_dict['<START>']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        decoded_sentence = ''

        stop_condition = False
        while not stop_condition:
            # Run the decoder model to get possible
            # output tokens (with probabilities) & states
            output_tokens, hidden_state, cell_state = decoder_model.predict(
                [target_seq] + states_value)

            # Choose token with highest probability
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_features_dict[sampled_token_index]
            decoded_sentence += " " + sampled_token

            # Exit condition: either hit max length
            # or find stop token.
            if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [hidden_state, cell_state]
            # remove <START> and <END> tokens
            # from decoded_sentence
            decoded_sentence = decoded_sentence.replace("<START>","").replace("<END>")
        return decoded_sentence
    def chat(self,reply):
        if self.make_exit(reply):
            print("cool !...have a nice day ..bye")
            return
        reply = input(self.generate_response(reply))
        return self.match(reply)

    def describe_system_intent(self):
        reply = input("I am based rule-generative based  chatbot model...isn't that cool !\n")
        self.match(reply)
    def area_intent(self,number):
        area = number**2
        reply = input("area of square is {}\n".format(area))
        self.match(reply)
robot = chatbot()
robot.start_chat()






