from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Check if MPS is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load base model
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map=device,
    torch_dtype=torch.float16 if device == "mps" else torch.float32
)

# Load PEFT adapter
model = PeftModel.from_pretrained(model, "barissglc/tinyllama-tarot-v1")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

def generate_response(prompt, max_new_tokens=200, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def create_tarot_prompt(card1, card2, card3, question):
    return f"""You are a professional tarot reader. Provide a detailed tarot reading following this EXACT structure. Do not skip any sections.

### Cards:
1. {card1}
2. {card2}
3. {card3}

### Question:
{question}

### Required Format:
1. Individual Card Meanings

① {card1}
Core Meaning: [Provide the core meaning of this card]
Work-related Meaning: [How this card relates to work/career]

② {card2}
Core Meaning: [Provide the core meaning of this card]
Work-related Meaning: [How this card relates to work/career]

③ {card3}
Core Meaning: [Provide the core meaning of this card]
Work-related Meaning: [How this card relates to work/career]

2. Cards' Relationship & Dynamics

Energy Flow: [Describe how the energy flows between these cards]
Overall Theme: [What is the main theme of this combination]

3. Comprehensive Interpretation

Positive Signals:
✅ [First positive aspect]
✅ [Second positive aspect]

Risk Warnings:
⚠️ [First risk or challenge]
⚠️ [Second risk or challenge]

Key Questions:
- [First important question to consider]
- [Second important question to consider]

4. Practical Advice

Action Steps:
1. [First action step]
2. [Second action step]
3. [Third action step]

Things to Avoid:
❌ [First thing to avoid]
❌ [Second thing to avoid]

### Response:
"""

def get_user_input():
    print("\n=== Tarot Reading Input ===")
    print("Please enter your three tarot cards:")
    card1 = input("First card: ").strip().title()  # Convert to Title Case
    card2 = input("Second card: ").strip().title()
    card3 = input("Third card: ").strip().title()
    print("\nWhat would you like to know about these cards?")
    question = input("Your question: ").strip()
    return card1, card2, card3, question

# Get user input and generate reading
card1, card2, card3, question = get_user_input()
prompt = create_tarot_prompt(card1, card2, card3, question)
print("\nGenerating your tarot reading...\n")
response = generate_response(prompt, max_new_tokens=10000)  # Increased token limit for more detailed response
print("\n=== Your Tarot Reading ===\n")
print(response)