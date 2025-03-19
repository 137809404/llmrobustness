import pandas as pd

def read_parquet(file_path):
    """
    读取 parquet 文件并返回 DataFrame
    """
    try:
        # 使用 pandas 读取 parquet 文件
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return None

def main():
    # 替换为你的 parquet 文件路径
    file_path = "/home/hang/projects/MolecularGPT/test_process/bace/0/test_bace_0.parquet"
    
    # 读取 parquet 文件
    df = read_parquet(file_path)
    
    if df is not None:
        # 打印文件内容
        print("Parquet file contents:")
        print(df)
        
        # 打印文件的基本信息
        print("\nFile info:")
        print(f"Number of rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        
        # 打印前 5 行数据
        print("\nFirst 5 rows:")
        print(df.head())

if __name__ == "__main__":
    main() 