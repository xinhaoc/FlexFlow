#!/usr/bin/env python

import os
import requests
from transformers import AutoModelForCausalLM

# Change working dir to folder storing this script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def convert_hf_model(model, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for name, params in model.named_parameters():
        name = (
            name.replace(".", "_")
            .replace("self_attn", "attention")
            .replace("q_proj", "wq")
            .replace("k_proj", "wk")
            .replace("v_proj", "wv")
            .replace("o_proj", "wo")
            .replace("mlp", "feed_forward")
            .replace("gate_proj", "w1")
            .replace("down_proj", "w2")
            .replace("up_proj", "w3")
            .replace("input_layernorm", "attention_norm")
            .replace("post_attention_layernorm", "ffn_norm")
            .replace("embed_tokens", "tok_embeddings")
            .replace("lm_head", "output")
            .replace("model_", "")
        )
        params.detach().cpu().numpy().tofile(f"{dst_folder}/{name}")

# Download and convert big model weights
model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
dst_folder="../weights/llama_7B_weights"
convert_hf_model(model, dst_folder)

# Download and convert small model weights
model = AutoModelForCausalLM.from_pretrained("JackFram/llama-160m")
dst_folder="../weights/llama_160M_weights"
convert_hf_model(model, dst_folder)

# Download tokenizer
os.makedirs("../tokenizer", exist_ok=True)
tokenizer_filepath = '../tokenizer/tokenizer.model'
url = 'https://huggingface.co/JackFram/llama-160m/resolve/main/tokenizer.model'
r = requests.get(url)
open(tokenizer_filepath , 'wb').write(r.content)