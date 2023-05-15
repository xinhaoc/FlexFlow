from transformers import OPTConfig, OPTForCausalLM, GPT2Tokenizer

model_id = "facebook/opt-6.7b"
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
model = OPTForCausalLM.from_pretrained(model_id)

prompts = [
            "Today is a beautiful day and I want",
        ]


for name, param in model.named_parameters():
            name = name.replace('.', '_').replace('model_decoder_', '')
            name = name.replace('self_attn_q_proj_weight', 'attention_wq_weight')
            name = name.replace('self_attn_k_proj_weight', 'attention_wk_weight')
            name = name.replace('self_attn_v_proj_weight', 'attention_wv_weight')
            name = name.replace('self_attn_out_proj_weight', 'attention_wo_weight')
            
            name = name.replace('self_attn_q_proj_bias', 'attention_wq_bias')
            name = name.replace('self_attn_k_proj_bias', 'attention_wk_bias')
            name = name.replace('self_attn_v_proj_bias', 'attention_wv_bias')
            name = name.replace('self_attn_out_proj_bias', 'attention_wo_bias')
            param.detach().cpu().numpy().tofile('/home/ubuntu/FlexFlow/examples/cpp/inference/opt/weights_6B/' + name)

# for prompt in prompts:
#     input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids
#     print(input_ids)
#     generated_ids = model.generate(input_ids, max_length=30)
#     generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#     print(generated_ids)
#     print(generated_string)