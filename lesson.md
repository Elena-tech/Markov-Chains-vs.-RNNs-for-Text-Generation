# **🧙‍♂️ Battle of the AI Wizards: Markov vs. RNN! 🤖**
Welcome, young apprentice! Today, you will **witness** a battle between two mighty **text-generating wizards**:

🔮 **Markov the Predictor** – A wizard who follows strict rules and only remembers the last few words!  
🌀 **RNN the Learner** – A sorcerer who learns from experience and generates text with deep wisdom!  

Who will win this magical duel? Let’s find out! 🚀✨

---

## **🔢 Step 1: Markov’s Spell of Randomness! 🎲**
**Markov the Predictor** can only see **a few words ahead**, like a fortune teller who only remembers yesterday’s visions! 🏮🔮

Let’s build his **spellbook of words** and see what kind of **prophecy** he conjures! 📝✨

```python
import random

def build_markov_chain(text, order=1):
    """Create a Markov chain dictionary from input text 📖✨"""
    words = text.split()
    markov_chain = {}

    for i in range(len(words) - order):
        key = tuple(words[i:i+order])
        next_word = words[i+order]
        if key not in markov_chain:
            markov_chain[key] = []
        markov_chain[key].append(next_word)

    return markov_chain

def generate_markov_text(chain, length=20, seed_word=None):
    """Generate a mystical prophecy using Markov magic! 🔮"""
    key = random.choice(list(chain.keys())) if not seed_word else seed_word
    generated_words = list(key)

    for _ in range(length):
        if key in chain:
            next_word = random.choice(chain[key])
            generated_words.append(next_word)
            key = tuple(generated_words[-len(key):])
        else:
            break

    return ' '.join(generated_words)

# 📜 Behold, the sacred text of our AI prophecy:
sacred_text = """
The moon 🌙 whispers to the sea 🌊, while the stars 🌟 dance in the sky.
The wizard speaks in riddles, and the trees 🌲 sing the songs of time.
Magic flows like a river, and shadows follow the wind. 🍃✨
"""

# 🧙‍♂️ Markov’s Spellbook
markov_chain = build_markov_chain(sacred_text, order=2)
generated_prophecy = generate_markov_text(markov_chain, length=15)

print("🔮 **Markov's Prophecy:**")
print(generated_prophecy + " ✨")
```

---

## **🤖 Step 2: RNN’s Power of Deep Learning! 🧠⚡**
Unlike Markov, **RNN the Learner** remembers **everything** he reads and generates text that **flows like poetry**! 📜✨

Let’s summon his **deep learning magic** and witness his wisdom! ⚡📖

```python
from transformers import pipeline

# 🚀 Load the AI wizard (GPT-2 model)
generator = pipeline("text-generation", model="gpt2")

# 🌙 Invoke RNN magic
prompt = "The enchanted forest hides secrets"
rnn_generated_text = generator(prompt, max_length=50, num_return_sequences=1)

print("\n🌀 **RNN's Prophecy:**")
print(rnn_generated_text[0]["generated_text"] + " 🌌")
```

---

## **🤔 Who Wins the Battle? YOU Decide! ⚔️✨**
### **Questions for the Grand Wizard Council (YOU!):**
1️⃣ Which text sounds more **magical and poetic**? ✨📜  
2️⃣ Which AI wizard would you trust to **write a fantasy novel**? 📖🔥  
3️⃣ What happens if we **change Markov’s memory size** (increase `order`)? 🧠🔧  
4️⃣ Which method do you think **AI assistants like Siri use**? 🤖💬  

---

### **🔮 Summary of the Battle**
| Feature 🔥 | Markov the Predictor 🧙‍♂️ | RNN the Learner 🌀 |
|------------|-------------------------|---------------------|
| **Memory** | Remembers only a few words 🧠 | Remembers everything 📚 |
| **Coherence** | Often random & weird 🤪 | More logical & structured 🏛️ |
| **Training** | No learning required 🚀 | Needs lots of data 📊 |
| **Speed** | Super fast ⚡ | Slower but smarter 🏆 |

Who will you choose as the **ultimate AI wizard**?  
The **quick but simple Markov**, or the **deeply wise RNN**? 🧙‍♂️⚡

---

### **🎭 Bonus Challenge: Make It Your Own!**
🔥 Try running this with **your own custom text** (like your favorite book excerpt)!  
🔥 Experiment with **different `order` values** for Markov!  
🔥 Try **different AI models** instead of GPT-2!  

Let the **AI wizard battle** begin! 🏆✨  
🚀 **Run your code and share your results!** 🚀
