import tkinter as tk
from tkinter import *
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

class DigitRecognizer:
    def __init__(self, this):
        self.this = this
        self.canvas = Canvas(this, width=600, height=600, bg='white')
        self.canvas.pack()

        self.button_predict = Button(this, text="Predict", command=self.predict)
        self.button_predict.pack()
        self.button_clear = Button(this, text="Clear", command=self.clear)
        self.button_clear.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.model = tf.keras.models.load_model('model.h5')
        self.image = Image.new("L", (600, 600), 255)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 12), (event.y - 12)
        x2, y2 = (event.x + 12), (event.y + 12)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=12)
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, 600, 600), fill=255)

    def predict(self):
        image_resized = self.image.resize((60, 60), Image.LANCZOS)
        image_inverted = ImageOps.invert(image_resized)
        image_np = np.array(image_inverted).reshape(1, 60, 60, 1).astype('float32') / 255.0
        prediction = self.model.predict(image_np)
        digit = np.argmax(prediction)
        print(f"Predicted digit: {digit}")

if __name__ == "__main__":
    root = Tk()
    root.title("Digit Guesser")
    recognizer = DigitRecognizer(root)
    root.mainloop()