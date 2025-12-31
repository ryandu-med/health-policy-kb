import os
import glob
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document

# ==========================================
# ✅ 智能路径配置 (自动找当前目录下的 data 文件夹)
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
META_FILE = os.path.join(DATA_DIR, "kb_data.pkl")


def read_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for i, page in enumerate(reader.pages):
            if i > 80: break  # 防止文件太长
            text += page.extract_text() or ""
        return chunk_text(text)
    except:
        return []


def read_word(file_path):
    try:
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return chunk_text(text)
    except:
        return []


def chunk_text(text, size=600, overlap=100):
    if not text: return []
    text = text.replace('\n', ' ').replace('  ', ' ')
    return [text[i:i + size] for i in range(0, len(text), size - overlap) if len(text[i:i + size]) > 50]


def main():
    print(f"=== 🚀 启动：正在扫描 data 文件夹 ===")
    print(f"📂 资料路径: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误：没找到 data 文件夹！请确认你在 Health_KB 下创建了 data 文件夹并放入了文件。")
        return

    print("📥 加载 AI 模型...")
    encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    texts = []
    metadata = []

    # 扫描所有 docx 和 pdf
    files = glob.glob(os.path.join(DATA_DIR, "*.*"))
    valid_files = [f for f in files if f.lower().endswith(('.docx', '.pdf'))]

    print(f"\n📂 发现 {len(valid_files)} 个有效文件，开始处理...")

    for i, f in enumerate(valid_files):
        name = os.path.basename(f)
        print(f"   [{i + 1}/{len(valid_files)}] 读取: {name}")

        chunks = []
        if f.lower().endswith('.docx'):
            chunks = read_word(f)
            type_str = 'word'
        elif f.lower().endswith('.pdf'):
            chunks = read_pdf(f)
            type_str = 'pdf'

        for chunk in chunks:
            texts.append(f"【文件】{name}\n内容：{chunk}")
            metadata.append({"source": name, "type": type_str, "content": chunk})

    if texts:
        print(f"\n💾 正在为 {len(texts)} 条片段生成索引...")
        embeddings = encoder.encode(texts, normalize_embeddings=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))

        # 保存到 data 文件夹里
        faiss.write_index(index, INDEX_FILE)
        with open(META_FILE, "wb") as f:
            pickle.dump({'texts': texts, 'metadata': metadata}, f)

        print(f"\n✅ 索引已保存到: {DATA_DIR}")
        print("🎉🎉🎉 成功！数据处理完成。请运行 app.py。")
    else:
        print("❌ 错误：data 文件夹里没有找到有效的 Word 或 PDF 文件！")


if __name__ == "__main__":
    main()