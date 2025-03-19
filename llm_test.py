import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        # Each line is a separate JSON object
        data = [json.loads(line) for line in f]
    return data

# Prepare the input for LLM
def prepare_input(data):
    inputs = []
    for item in data:
        # Build input in role: user format
        prompt = [{
            "role": "user",
            "content": f"{item['instruction']}\nOnly output the answer from the following options: Yes, No.\nInput: {item['input']}"
        }]
        inputs.append(prompt)
    return inputs

# Load model and tokenizer
def load_model():
    model_name = "/home/hang/projects/TEA-GLM/llm-model/vicuna-7b-v1.5"  # Replace with actual Llama3.1 8B model name
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    return model, tokenizer

# Generate responses
def generate_responses(model, tokenizer, inputs):
    responses = []
    for input_text in inputs:
        # Tokenize input
        inputs_tokenized = tokenizer.apply_chat_template(input_text, tokenize=False, add_generation_prompt=True)
        inputs_tokenized = tokenizer([inputs_tokenized], return_tensors="pt").to("cuda")
        
        # Generate output
        outputs = model.generate(
            **inputs_tokenized,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs_tokenized.input_ids, outputs)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(input_text)
        print(response)
        # Decode output and extract only the answer (Yes/No)
        # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the last word (Yes/No) from the response
        responses.append(response)
    return responses

# Main function
def main():
    # Load dataset
    dataset = load_dataset("test_dataset/8-shot/bace/bace_0.json")
    
    # Prepare inputs
    inputs = prepare_input(dataset)
    
    # Load model
    model, tokenizer = load_model()
    
    # Generate responses
    responses = generate_responses(model, tokenizer, inputs)
    
    # Print results
    for i, (input_text, response) in enumerate(zip(inputs, responses)):
        print(f"Sample {i+1}:")
        print("Input:", input_text)
        print("Response:", response)
        print("-" * 50)

if __name__ == "__main__":
    main()
