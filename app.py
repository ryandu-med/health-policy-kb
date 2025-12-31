import os
import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# ================= 配置区域 =================
# 👇👇👇 请在这里填入你的 Key 👇👇👇
DEEPSEEK_API_KEY = "sk-c83c82aa94e245c390cf242e93d6585a"
# 👆👆👆👆👆👆👆👆👆👆👆👆👆👆
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# ✅ 智能路径配置 (自动找当前目录下的 data 文件夹)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
META_FILE = os.path.join(DATA_DIR, "kb_data.pkl")

st.set_page_config(layout="wide", page_title="健康融入所有政策知识库平台", page_icon="🏥")

# ================= 样式优化 =================
st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; color: #333; }
    h1 { color: #2c3e50; text-align: center; font-family: "Microsoft YaHei"; padding: 20px 0; border-bottom: 2px solid #e9ecef; margin-bottom: 30px; }

    /* 搜索框和按钮 */
    .stTextInput input { border: 1px solid #ced4da; border-radius: 6px; height: 48px; }
    .stButton button { height: 48px; border-radius: 6px; font-weight: bold; background-color: #007bff; color: white; }
    .stButton button:hover { background-color: #0056b3; }

    /* 原文卡片样式 */
    .source-card { background-color: white; padding: 15px; border-radius: 8px; border-left: 5px solid #007bff; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .source-title { font-weight: bold; color: #007bff; font-size: 0.9em; margin-bottom: 8px; }
    .source-content { font-size: 0.95em; line-height: 1.6; color: #495057; }
</style>
""", unsafe_allow_html=True)


# ================= 核心逻辑 =================
@st.cache_resource
def load_resources():
    # 加载嵌入模型
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


class KnowledgeBase:
    def __init__(self):
        self.encoder = load_resources()
        self.index = None
        self.texts = []
        self.metadata = []

    def load(self):
        # 检查文件是否存在
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "rb") as f:
                data = pickle.load(f)
                self.texts = data['texts']
                self.metadata = data['metadata']
            return True
        return False

    def search(self, query, top_k=5):
        """
        语义搜索功能
        top_k: 返回最相关的几条
        """
        if not self.index: return []

        # 1. 语义分析：将问题转化为向量
        vec = self.encoder.encode([query], normalize_embeddings=True)

        # 2. 向量检索：在知识库中匹配
        dists, idxs = self.index.search(np.array(vec).astype('float32'), top_k)

        results = []
        for i, idx in enumerate(idxs[0]):
            # 阈值过滤：距离小于 1.5 才算相关 (可根据实际情况微调)
            if idx != -1 and idx < len(self.texts) and dists[0][i] < 1.5:
                results.append({"content": self.texts[idx], "meta": self.metadata[idx]})
        return results


def ask_ai(sys_msg, user_msg):
    """调用 DeepSeek API"""
    if "sk-" not in DEEPSEEK_API_KEY: return "🚨 请配置 API Key"
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    try:
        return client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            stream=True, temperature=0.4  # 温度稍微调低，让分析更严谨
        )
    except Exception as e:
        return f"错误: {e}"


# ================= 界面构建 =================
if 'kb' not in st.session_state:
    st.session_state.kb = KnowledgeBase()
    st.session_state.kb.load()
if "messages" not in st.session_state: st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/health-book.png", width=70)
    st.markdown("### 🏥 知识库状态")
    if st.session_state.kb.index:
        st.success(f"🟢 已挂载 {len(st.session_state.kb.texts)} 条政策片段")
    else:
        st.error(f"🔴 数据丢失\n请运行 generate_data.py")
    st.divider()
    st.info("💡 **提示**：\n\n**功能1**：精准查找文件原文。\n**功能2**：AI 深度评估政策效果。")

# Main Title
st.title("🏥 健康融入所有政策知识库平台")

# 搜索区
c1, c2, c3 = st.columns([1, 5, 1])
with c1:
    if st.button("🗑️ 清空"): st.session_state.messages = []; st.rerun()
with c2:
    q = st.text_input("搜索", placeholder="请输入您想查询的政策关键词或分析需求...", key="q",
                      label_visibility="collapsed")
with c3:
    search = st.button("🔍 查询")

# 功能选择模式
mode = st.radio("请选择功能模式：",
                ["功能1：政策检索与资料学习", "功能2：政策分析与政策评估"],
                horizontal=True)

st.divider()

# 处理逻辑
if (search or q) and q:
    # 防止页面刷新导致输入丢失，强制存入历史
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != q:
        st.session_state.messages = [{"role": "user", "content": q}]  # 每次新搜清空旧的，保持界面清爽（可选）

        # 显示用户提问
        st.subheader(f"📝 提问：{q}")

        with st.spinner("正在进行语义分析与知识库检索..."):
            # 统一先进行检索
            search_results = st.session_state.kb.search(q, top_k=4)

        # ================= 功能 1：政策检索与资料学习 =================
        if mode == "功能1：政策检索与资料学习":
            if not search_results:
                st.error("对不起，目前知识库中无相关信息。")
            else:
                st.success(f"✅ 检索到 {len(search_results)} 条相关原文资料：")

                # 直接展示原文卡片
                for res in search_results:
                    source_name = res['meta']['source']
                    content = res['meta']['content']
                    # 使用 HTML/CSS 渲染好看的卡片
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">📄 来源文件：{source_name}</div>
                        <div class="source-content">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # 后面可以加一个小小的 AI 总结，辅助阅读（可选）
                st.markdown("---")
                st.caption("🤖 AI 辅助阅读：以上是为您找到的最相关原文，请查阅。")

        # ================= 功能 2：政策分析与政策评估 =================
        elif mode == "功能2：政策分析与政策评估":
            if not search_results:
                st.warning("⚠️ 知识库中未找到直接相关文件，AI 将基于通用知识进行分析，但结果可能缺乏实证依据。")
                context = "（知识库中无具体资料，请基于你的专业知识回答）"
            else:
                st.info(f"📚 已基于 {len(search_results)} 份相关政策文件进行分析...")
                # 拼接资料
                context = "\n\n".join([f"【资料{i + 1}】{r['content']}" for i, r in enumerate(search_results)])

            # 构建高阶 Prompt
            system_prompt = """
            你是一位资深的“健康融入所有政策（HiAP）”评估专家。
            你的任务是根据用户的提问，结合提供的参考资料，进行深度的政策分析与评估。

            要求：
            1. **语义分析**：首先理解用户提问的核心诉求。
            2. **结合资料**：必须优先引用提供的资料中的数据、条款或目标。
            3. **评估工具**：请运用专业的政策评估框架（如RE-AIM模型、逻辑框架法、SWOT分析等），在回答中体现你使用了评估视角。
            4. **实施效果**：重点分析政策的实施路径、预期效果及潜在挑战。
            5. **输出格式**：输出为结构清晰的段落文字，逻辑严密，语言专业。
            """

            user_prompt = f"用户需求：{q}\n\n参考资料：\n{context}\n\n请开始撰写评估分析报告："

            # 流式输出
            st.markdown("### 📊 政策分析与评估报告")
            stream = ask_ai(system_prompt, user_prompt)
            st.write_stream(stream)