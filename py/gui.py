import tkinter as tk
from tkinter import scrolledtext
from PIL import ImageTk, Image

def gui (on_input_change):
    root = tk.Tk()
    root.title("Sentiment Image Chatbot")

    root.geometry("600x600")
    root.resizable(False, False)
    chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20, state=tk.DISABLED, font=("Arial", 12))
    chat_history.pack(pady=10)
    input_frame = tk.Frame(root)
    input_frame.pack(pady=10)

    input_text = tk.Entry(input_frame, width=50, font=("Arial", 12))
    input_text.pack(side=tk.LEFT, padx=10)

    def on_input_change_local():
        input_message = input_text.get()
        if input_message:
            chat_history.config(state=tk.NORMAL)
            chat_history.insert(tk.END, f"You: {input_message}\n")
            chat_history.yview(tk.END)

            on_input_change(input_message, chat_history, input_text)

    send_button = tk.Button(input_frame, text="Send", command=on_input_change_local, font=("Arial", 12), bg="lightblue")
    send_button.pack(side=tk.LEFT)

    root.mainloop()
