from openai import OpenAI
import json
import os
from pathlib import Path
import time
import argparse

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        # Each line is a separate JSON object
        data = [json.loads(line) for line in f]
    return data

def ensure_directory(directory):
    """确保输出目录存在"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_api_response(client, prompt, max_retries=3, retry_delay=5):
    """获取API响应，失败时自动重试"""
    for attempt in range(max_retries):
        try:
            print(f"prompt: {prompt}")
            completion = client.chat.completions.create(
                extra_body={},
                model="meta-llama/llama-3.3-70b-instruct:free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # if completion and completion.choices and len(completion.choices) > 0 and completion.choices[0].message:
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:  # 如果不是最后一次尝试
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            continue
    
    raise Exception(f"Failed to get valid response after {max_retries} attempts")

def process_dataset(input_path, output_path):
    """处理数据集中的JSON文件并保存结果"""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )

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

                # 读取JSON文件
                data = load_dataset(input_file)
                for d in data:
                    while True:  # 持续尝试直到成功
                        try:
                            # 构建提示信息
                            prompt = f"{d['instruction']}\n Your output must match one of the options exactly: 'Yes' or 'No'.\nOutput only the chosen option.\n{d['input']}"

                            # 获取响应（包含重试机制）
                            response = get_api_response(client, prompt)
                            print(f"Response: {response}")

                            # 保存结果
                            result = {
                                "instruction": d["instruction"],
                                "input": d["input"],
                                "expected_output": d["output"],
                                "model_output": response
                            }

                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(result, f, ensure_ascii=False, indent=2)

                            print(f"Processed: {input_file}")
                            break  # 成功后退出循环

                        except Exception as e:
                            print(f"Error occurred: {str(e)}")
                            print("Retrying after 5 seconds...")
                            time.sleep(5)  # 等待5秒后重试

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