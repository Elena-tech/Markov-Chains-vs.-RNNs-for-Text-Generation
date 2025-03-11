# **📝 Solutions: AI Text Remix Challenge!**  

---

## **🎯 Challenge 1: Markov Rhyming Poet! 🎶**  
### **🔹 Solution: Using a Rhyming Dictionary**  

We use the **`pronouncing`** library to find rhyming words and modify the Markov chain to prefer them! 🔥  

```python
import random
import pronouncing

def build_markov_chain(text, order=1):
    """Build a Markov chain with rhyming capabilities 🎶"""
    words = text.split()
    markov_chain = {}

    for i in range(len(words) - order):
        key = tuple(words[i:i+order])
        next_word = words[i+order]
        if key not in markov_chain:
            markov_chain[key] = []
        markov_chain[key].append(next_word)

    return markov_chain

def generate_rhyming_markov_text(chain, length=10, seed_word=None):
    """Generate rhyming text using Markov chains + rhyming words 🎤"""
    key = random.choice(list(chain.keys())) if not seed_word else seed_word
    generated_words = list(key)

    for _ in range(length):
        if key in chain:
            possible_words = chain[key]
            rhyming_choices = [w for w in possible_words if pronouncing.rhymes(w)]
            next_word = random.choice(rhyming_choices) if rhyming_choices else random.choice(possible_words)
            generated_words.append(next_word)
            key = tuple(generated_words[-len(key):])
        else:
            break

    return ' '.join(generated_words)

# 📜 Example text with poetic vibes:
sacred_text = """
The moon shines bright, casting light on the night.
Stars flicker, burning fire, never tire.
The wind hums a tune, while dreams begin soon.
"""

# 🏗️ Build & Generate!
markov_chain = build_markov_chain(sacred_text, order=2)
rhyming_text = generate_rhyming_markov_text(markov_chain, length=10)

print("🎶 Rhyming Markov Generated Text:")
print(rhyming_text + " ✨")
```

🔹 **What This Does:**  
✔ Uses Markov Chains to generate random words.  
✔ Prefers words that **rhyme** with the previous word! 🎤  

---

## **🎯 Challenge 2: Train Your Own Mini-RNN! 🧠**  
### **🔹 Solution: Training a Simple LSTM**  

Here’s how you **train an LSTM model** on **a small dataset** to generate poetic text! 🤖📜  

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 📜 Custom text dataset
poetry_lines = [
    "The sun sets beyond the trees",
    "A river flows with gentle ease",
    "Shadows dance beneath the moon",
    "The night will fade away too soon"
]

# 🌟 Tokenize the dataset
tokenizer = Tokenizer()
tokenizer.fit_on_texts(poetry_lines)
total_words = len(tokenizer.word_index) + 1

# 📊 Convert text into training sequences
input_sequences = []
for line in poetry_lines:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

# 🚀 Pad sequences for consistency
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# 🎯 Define features & labels
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# 🔥 Build an LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 8, input_length=max_sequence_length-1),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

# ⚡ Compile & Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=1)

# 🏆 Generate new poetic lines!
def generate_poetry(seed_text, next_words=10):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        seed_text += " " + tokenizer.index_word[predicted[0]]
    return seed_text

# 📝 Example Run
print("\n📜 AI-Generated Poetry:")
print(generate_poetry("The sun", next_words=5))
```

🔹 **What This Does:**  
✔ Trains a **small LSTM model** on poetry.  
✔ Generates text **in a similar style** to the dataset.  
✔ Uses **word embeddings + LSTM layers** to predict the next word!  

---

## **🎯 Challenge 3: Hybrid AI Poet! 🤯**  
### **🔹 Solution: Mixing Markov & RNN!**  

Here’s how to **combine Markov Chains & RNNs** to create a hybrid AI poet! 🏆✨  

```python
from transformers import pipeline
import random

# 🚀 Load AI wizard (GPT-2)
rnn_generator = pipeline("text-generation", model="gpt2")

# 🧙‍♂️ Markov Chain Generator
def generate_markov_seed(chain, length=5):
    """Generate a Markov chain text snippet to feed into GPT-2"""
    key = random.choice(list(chain.keys()))
    generated_words = list(key)

    for _ in range(length):
        if key in chain:
            next_word = random.choice(chain[key])
            generated_words.append(next_word)
            key = tuple(generated_words[-len(key):])
        else:
            break

    return ' '.join(generated_words)

# 🏗️ Build Markov Chain
sacred_text = """
The ocean whispers, calling deep.
The night is young, but stars still weep.
A path unknown, where secrets keep.
The sky is dark, but dreams won’t sleep.
"""
markov_chain = build_markov_chain(sacred_text, order=2)

# 🔮 Generate a Markov seed & complete with RNN magic!
markov_seed = generate_markov_seed(markov_chain, length=7)
rnn_output = rnn_generator(markov_seed, max_length=50, num_return_sequences=1)

print("💡 Hybrid AI Poet Generated Text:")
print(rnn_output[0]["generated_text"] + " ✨")
```

🔹 **What This Does:**  
✔ Uses **Markov Chains** to create a **starter sentence**.  
✔ Passes that into **GPT-2 (RNN)** to **continue the story**! 📜💡  
✔ Creates a **hybrid AI poet** that mixes **randomness & deep learning!** 🤯  

---

## **🏆 Recap: What Did We Do?**
| **Challenge** | **Solution Summary** |
|--------------|----------------------|
| 📝 **Markov Rhyming Poet** | Used **rhyming words** to make Markov’s text sound more poetic 🎶 |
| 🔥 **Train Your Own RNN** | Built an **LSTM-based generator** using Keras 🤖 |
| 🌀 **Hybrid AI Poet** | Combined **Markov randomness + RNN intelligence** for a unique generator 🔥 |

---

## **🚀 Bonus: What’s Next?**
🎭 Try using **different datasets** (e.g., song lyrics, fairy tales).  
💡 Modify **Markov’s order size** to see how it affects randomness.  
🔗 Experiment with **different RNN architectures** (GRU, Transformer models).  
