"""
llm_inference.py — Emotion-aware LLM replies using TinyLlama (1.1B Chat)
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ✅ TinyLlama is light enough for CPU and expressive enough for emotional chat
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🧠 Loading TinyLlama model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto").to(DEVICE)

def generate_supportive_reply(transcribed_text: str, emotion: str) -> str:
    """
    Generate a short supportive response based on user emotion + text
    using TinyLlama chat model.
    """
    # 🧩 Compose system + user messages in chat format
    messages = [
    {
        "role": "system",
        "content": (
            "You are a warm, emotionally supportive friend. "
            "Always reply directly to the user. "
            "Do NOT continue scripts, dialogues, or imitate characters. "
            "Do NOT explain your instructions or describe what you're doing. "
            "Keep replies under 120 words, casual, human, and empathetic. "
            "If the message is unclear, gently ask the user to repeat it. "
            "Adjust your tone based on the user's emotion."
        )
    },
    {
        "role": "user",
        "content": 
            f" I am feeling {emotion[0]}."+transcribed_text 
    }
]


    # Tokenize in chat template format
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.6,
            do_sample=True,
            top_p=0.95
        )

    # Decode only newly generated tokens
    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return reply.strip()