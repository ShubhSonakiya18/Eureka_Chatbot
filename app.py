import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F
import tokenizers  # needed for safe globals

# Set page config
st.set_page_config(
    page_title="EUREKA",
    page_icon="🤖",
    layout="wide"
)

# Load model and tokenizer
@st.cache_resource
def load_model():
    # Safe globals fix for tokenizers.AddedToken
    torch.serialization.add_safe_globals([tokenizers.AddedToken])

    try:
        # Try loading as state_dict (standard fine-tuning save)
        checkpoint = torch.load("gpt2_all_in_one.pth", map_location=torch.device("cpu"), weights_only=False)
        
        if "model_state_dict" in checkpoint:
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model.load_state_dict(checkpoint["model_state_dict"])
        
        elif "model" in checkpoint and "tokenizer" in checkpoint:
            # If whole model + tokenizer were saved
            model = checkpoint["model"]
            tokenizer = checkpoint["tokenizer"]
        
        else:
            # If only model object was saved
            model = checkpoint
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        model.eval()
        return model, tokenizer

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to generate response
def generate_response(prompt, model, tokenizer, max_length=100, temperature=0.7, top_k=50, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            num_return_sequences=1
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Main app
def main():
    st.title("🤖 EUREKA")
    
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        st.stop()
    
    with st.sidebar:
        st.header("Generation Parameters")
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
        max_length = st.slider("Max Length", 50, 200, 100, 10)
        top_k = st.slider("Top K", 1, 100, 50, 1)
        top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.1)
    
    st.header("This chatbot is trained on DailyDialog dataset\nEureka!")
    
    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(
                    prompt,
                    model,
                    tokenizer,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
