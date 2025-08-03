# 파일명: app.py
import os
import pandas as pd
import re
from collections import defaultdict, Counter
import streamlit as st
import plotly.express as px
import time

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="반도체 정비노트 검색&추천", layout="wide")
st.title("🔧 반도체 정비노트 검색 & 성공률 추천")

# OpenAI API Key
api_key = st.text_input("🔑 OpenAI API Key를 입력하세요", type="password")
if not api_key:
    st.warning("API 키를 입력해야 진행 가능합니다.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# 엑셀 업로드
uploaded_file = st.file_uploader("📁 엑셀 파일 업로드 (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.info("엑셀 파일을 업로드하면 분석이 시작됩니다.")
    st.stop()

df = pd.read_excel(uploaded_file)
df = df.dropna(subset=['정비노트'])
st.success(f"✅ 업로드 완료: 총 {len(df)} 행")

# -----------------------------
# 텍스트 기반 성공률 계산
# -----------------------------
lines = []
for note in df["정비노트"].astype(str):
    for line in note.split('\n'):
        text = re.sub(r'^\d{2}월\d{2}일 \d{2}:\d{2} ', '', line).strip()
        if text and not text.startswith("LOT m951990 처리 중 abnormal 현상 감지"):
            lines.append(text)

cause_pattern = re.compile(r'LOT 진행 중 (.+) 발생')
first_action_pattern = re.compile(r'1차 조치: (.+) → 여전히 이상 발생')
second_action_pattern = re.compile(r'정비 시작\. (.+) 진행')
third_action_pattern = re.compile(r'추가 조치: (.+)')

cause_aliases = {
    "wafer not 발생": "wafer not",
    "wafer not 감지됨": "wafer not",
    "wafer not 발생 확인": "wafer not",
}

def normalize_cause(cause):
    for alias, norm in cause_aliases.items():
        if alias in cause:
            return norm
    return cause

cause_action_counts = defaultdict(lambda: defaultdict(Counter))
cause = None

for line in lines:
    if (m := cause_pattern.search(line)):
        cause = normalize_cause(m.group(1).strip())
        continue
    if cause is None:
        continue

    if (m1 := first_action_pattern.search(line)):
        cause_action_counts[cause][m1.group(1).strip()]['first'] += 1
    elif (m2 := second_action_pattern.search(line)):
        cause_action_counts[cause][m2.group(1).strip()]['second'] += 1
    elif (m3 := third_action_pattern.search(line)):
        cause_action_counts[cause][m3.group(1).strip()]['third'] += 1

rows = []
for cause, actions in cause_action_counts.items():
    for action, counts in actions.items():
        f, s, t = counts.get('first', 0), counts.get('second', 0), counts.get('third', 0)
        total, success = f + s + t, s + t
        rate = round(success / total * 100, 2) if total > 0 else 0
        rows.append({"대표원인": cause, "조치": action, "총횟수": total, "실패횟수": f, "성공횟수": success, "성공률(%)": rate})

df_success = pd.DataFrame(rows)

# -----------------------------
# 벡터 DB 구성 (Chroma 사용)
# -----------------------------
documents = [Document(page_content=str(row['정비노트']), metadata={'row': idx}) for idx, row in df.iterrows()]
split_docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vectordb = Chroma.from_documents(documents=split_docs, embedding=embedding_model)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(search_kwargs={'k': 20}), return_source_documents=True)

# -----------------------------
# UI
# -----------------------------
tab1, tab2 = st.tabs(["🔍 정비 검색 & 추천", "📈 통계 자료"])

with tab1:
    example_keywords = ["wafer not", "plasma ignition failure", "pumpdown 시간 지연", "slot valve 동작 불량", "RF auto match 불량"]
    st.markdown(f"<p style='font-size:18px;'>💡 예시 키워드: {' | '.join(example_keywords)}</p>", unsafe_allow_html=True)
    st.markdown("<h3>검색할 정비 이슈를 입력하세요</h3>", unsafe_allow_html=True)
    query = st.text_input("", placeholder="예: plasma ignition failure", key="query")

    if query.strip():
        bar = st.progress(0)
        for p in range(0, 101, 10): time.sleep(0.05); bar.progress(p)
        bar.empty()
        st.success(f"'{query}' 검색 완료")

        output = qa_chain({"query": query})
        docs = output["source_documents"]

        recommended = []
        for doc in docs:
            note = doc.page_content
            row_idx = doc.metadata['row']
            for _, row in df_success.iterrows():
                if row["조치"] in note:
                    recommended.append({
                        "조치": row["조치"],
                        "성공률": row["성공률(%)"],
                        "장비ID": df.loc[row_idx, '장비ID'] if '장비ID' in df.columns else 'N/A',
                        "모델": df.loc[row_idx, '모델'] if '모델' in df.columns else 'N/A',
                        "정비종류": df.loc[row_idx, '정비종류'] if '정비종류' in df.columns else 'N/A',
                        "정비자": df.loc[row_idx, '정비자'] if '정비자' in df.columns else 'N/A',
                        "정비노트": note
                    })

        if not recommended:
            st.warning("❗ 유사 사례가 없습니다.")
        else:
            dedup = {}
            for r in recommended:
                act = r["조치"]
                if act not in dedup or dedup[act]["성공률"] < r["성공률"]:
                    dedup[act] = r
            top3 = sorted(dedup.values(), key=lambda x: x["성공률"], reverse=True)[:3]

            st.subheader("✅ 성공률 상위 3개 조치")
            for idx, r in enumerate(top3, 1):
                st.markdown(f"**{idx}) {r['조치']}** - 성공률 {r['성공률']}%")

            st.markdown("### 🧾 대표 사례")
            note_html = top3[0]["정비노트"].replace("\n", "<br>")
            st.markdown(f"""
<div style="border:1px solid #ccc; padding:15px; background:#f9f9f9;">
<b>조치명:</b> {top3[0]['조치']}<br>
<b>장비:</b> {top3[0]['장비ID']} / {top3[0]['모델']}<br>
<b>정비종류:</b> {top3[0]['정비종류']}<br>
<b>정비자:</b> {top3[0]['정비자']}<br><br>
<b>정비노트:</b><br>{note_html}
</div>
""", unsafe_allow_html=True)

with tab2:
    st.subheader("📊 통계 분석")

    top5_equip = df['모델'].value_counts().head(5)
    fig1 = px.bar(x=top5_equip.index, y=top5_equip.values, text=top5_equip.values, color=top5_equip.values)
    st.plotly_chart(fig1, use_container_width=True)

    top5_cause = df_success.groupby('대표원인')['총횟수'].sum().nlargest(5)
    fig2 = px.bar(x=top5_cause.values, y=top5_cause.index, orientation='h', text=top5_cause.values)
    st.plotly_chart(fig2, use_container_width=True)

    df_success['row'] = df_success.index
    df_joined = df.merge(df_success, left_index=True, right_on='row', how='left')
    eng_stats = df_joined.groupby("정비자")["성공률(%)"].mean().dropna().round(1).sort_values(ascending=False).head(5)
    fig3 = px.bar(x=eng_stats.index, y=eng_stats.values, text=eng_stats.values, color=eng_stats.values)
    st.plotly_chart(fig3, use_container_width=True)
