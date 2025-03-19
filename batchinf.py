from vllm import LLM, SamplingParams
import json
import os
from pathlib import Path
import argparse

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        # Each line is a separate JSON object
        data = [json.loads(line) for line in f]
    # 构建prompt数组
    prompts = [
        f"{d['instruction']}\n Your output must match one of the options exactly: 'Yes' or 'No'.\nOutput only the chosen option.\n{d['input']}"
        for d in data
    ]
    return prompts, data

def ensure_directory(directory):
    """确保输出目录存在"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def process_dataset(input_path, output_path):
    """处理数据集中的JSON文件并保存结果"""
    # 初始化vLLM
    llm = LLM(model="/data2/zeyu/zeyu1/llmmodel/Llama3.1-70b-Instruct")
    sampling_params = SamplingParams(temperature=0.8, max_tokens=100)

    # 遍历输入目录
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith('.json'):
                input_file = os.path.join(root, file)
                
                # 构建对应的输出路径
                rel_path = os.path.relpath(root, input_path)
                output_dir = os.path.join(output_path, rel_path)
                ensure_directory(output_dir)
                output_file = os.path.join(output_dir, file)

                # 读取JSON文件并构建prompts
                prompts, original_data = load_dataset(input_file)
                
                # 批量推理
                outputs = llm.generate(prompts, sampling_params)
                
                # 保存结果
                results = []
                for i, output in enumerate(outputs):
                    result = {
                        "instruction": original_data[i]["instruction"],
                        "input": original_data[i]["input"],
                        "expected_output": original_data[i]["output"],
                        "model_output": output.outputs[0].text.strip()
                    }
                    results.append(result)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                print(f"Processed: {input_file}")

if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description='Process dataset with AI model')
    parser.add_argument('--input', type=str, default="test_dataset",
                        help='Input directory path')
    parser.add_argument('--output', type=str, default="output",
                        help='Output directory path')
    args = parser.parse_args()

    shots = [1, 2, 5, 8]
    datasets = ['bace', 'bbbp', 'cyp450', 'hiv', 'muv', 'tox21', 'toxcast']
    for dataset in datasets:
        for shot in shots:
            input_path = f"{args.input}/{shot}-shot/{dataset}"
            output_path = f"{args.output}/{shot}-shot/{dataset}"
            process_dataset(input_path, output_path)