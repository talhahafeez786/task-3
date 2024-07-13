import tkinter as tk
import nltk
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Sample corpus
corpus = """
Hello! How can I help you today? 
I am here to answer your questions. 
You can ask me anything. 
I will do my best to assist you. 
Thank you for using our service.
"""

# Tokenization
sent_tokens = nltk.sent_tokenize(corpus)
word_tokens = nltk.word_tokenize(corpus)

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you."
    else:
        robo_response = sent_tokens[idx]
    sent_tokens.remove(user_response)
    return robo_response

class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("Chatbot")

        self.chat_log = tk.Text(master, state='disabled', width=50, height=20)
        self.chat_log.grid(row=0, column=0, columnspan=2)

        self.user_input = tk.Entry(master, width=50)
        self.user_input.grid(row=1, column=0)

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1)

        self.quit_button = tk.Button(master, text="Quit", command=master.quit)
        self.quit_button.grid(row=2, column=0, columnspan=2)

    def send_message(self):
        user_message = self.user_input.get()
        self.user_input.delete(0, tk.END)

        if user_message.lower() == 'bye':
            self.chat_log.config(state='normal')
            self.chat_log.insert(tk.END, "Chatbot: Bye! Take care..\n")
            self.chat_log.config(state='disabled')
            return
        
        self.chat_log.config(state='normal')
        self.chat_log.insert(tk.END, "You: " + user_message + "\n")
        
        bot_response = response(user_message)
        self.chat_log.insert(tk.END, "Chatbot: " + bot_response + "\n")
        self.chat_log.config(state='disabled')
        self.chat_log.see(tk.END)  # Scroll to the end

def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(stop_words='english')  # Removed the tokenizer
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you."
    else:
        robo_response = sent_tokens[idx]
    sent_tokens.remove(user_response)
    return robo_response

if __name__ == "__main__":
    root = tk.Tk()
    chatbot_gui = ChatbotGUI(root)
    root.mainloop()
