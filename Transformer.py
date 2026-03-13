import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoTokenizer, AutoModel
from bertviz import head_view
import torch

# Define the models we want to visualize
MODELS = {
    "BERT (Bidirectional)": "bert-base-uncased",
    "GPT-2 (Autoregressive)": "gpt2"
}

@st.cache_resource
def load_model(model_name):
    """Loads the tokenizer and model, caching them so they don't reload on every UI interaction."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # output_attentions=True is required to get the weights for visualization
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    return tokenizer, model

# def get_attention_html(model_type, text):
#     """Generates the HTML for the attention visualization."""
#     tokenizer, model = load_model(MODELS[model_type])
    
#     inputs = tokenizer.encode(text, return_tensors='pt')
#     outputs = model(inputs)
    
#     # Extract attention and tokens
#     attention = outputs[-1] 
#     tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    
#     # Generate the HTML using BertViz
#     html_code = head_view(attention, tokens, html_action='return')
#     return html_code.data

def get_attention_html(model_type, text):
    """Generates the HTML for the attention visualization."""
    tokenizer, model = load_model(MODELS[model_type])
    
    # 1. Use the tokenizer directly to get a dictionary of all required inputs (input_ids, attention_mask, etc.)
    inputs = tokenizer(text, return_tensors='pt')
    
    # 2. Unpack the inputs into the model
    outputs = model(**inputs)
    
    # 3. Explicitly extract the attentions using the named attribute
    attention = outputs.attentions 
    
    # 4. Extract tokens using the specific input_ids
    input_ids = inputs['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # 5. Generate the HTML using BertViz
    html_code = head_view(attention, tokens, html_action='return')
    return html_code.data

# --- Streamlit UI ---

st.title("Transformer Attention Visualizer")
st.write("Interact with the self-attention mechanisms of different transformer architectures.")

# Sidebar for controls
st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox("Choose a Model Architecture:", list(MODELS.keys()))

st.sidebar.markdown("""
**How to use:**
1. Select a model type.
2. Enter a short sentence.
3. Hover over the words in the visualization to see how attention is distributed across different heads and layers.
""")

# Main input area
user_text = st.text_input("Enter text to analyze:", "The quick brown fox jumps over the lazy dog.")

if st.button("Visualize Attention"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner(f"Loading {selected_model} and processing..."):
            try:
                # Generate the interactive HTML visualization
                viz_html = get_attention_html(selected_model, user_text)
                
                # Render the HTML in the Streamlit app
                st.subheader(f"Attention Map: {selected_model}")
                components.html(viz_html, height=800, scrolling=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")

