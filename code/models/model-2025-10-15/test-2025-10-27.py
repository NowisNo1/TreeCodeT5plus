from transformers import CodeGenForCausalLM

model_path = "../CodeGen"
model = CodeGenForCausalLM.from_pretrained(model_path)
trainable_params = model.parameters()

print(sum(p.numel() for p in trainable_params))
