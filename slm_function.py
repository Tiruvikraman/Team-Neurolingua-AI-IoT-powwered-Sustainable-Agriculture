
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "phi3_finetuned_final",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
def response(input,length):
    my_question =input
    inputs = "Please answer to this question: " + my_question
    inputs = tokenizer(inputs, return_tensors="pt")
    outputs = model.generate(**inputs,max_new_tokens=length)
    answer = tokenizer.decode(outputs[0])
    return answer.replace('<pad>','').replace('</s>','')
