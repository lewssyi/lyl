import os
import re

def clean_file(file_path):
    # 尝试用不同的编码读取文件
    encodings = ['utf-8', 'gbk', 'utf-8-sig', 'cp936', 'latin-1']
    content = None
    used_encoding = None
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
            used_encoding = enc
            break
        except (UnicodeDecodeError, LookupError):
            continue
            
    if content is None:
        print(f"❌ 无法读取文件（未知编码）: {file_path}")
        return

    # 清洗掉类似于 [cite: 64] 的干扰标签
    cleaned_content = re.sub(r'\+\]', '', content)
    
    # 如果内容有变化，用读取时相同的编码写回，防止乱码
    if content != cleaned_content:
        with open(file_path, 'w', encoding=used_encoding) as f:
            f.write(cleaned_content)
        print(f"✨ 成功清洗 ({used_encoding}): {file_path}")

# 自动扫描并清洗
files_to_clean = ['run_main.py']
src_dir = 'src'

if os.path.exists(src_dir):
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                files_to_clean.append(os.path.join(root, file))

for file_path in files_to_clean:
    if os.path.exists(file_path):
        clean_file(file_path)

print("🎉 所有残留系统标签一键清洗完毕！请放心执行 python run_main.py")