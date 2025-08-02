# ✅ 변경 내용 적용된 전체 app.py 코드

import os
import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import time

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # ✅ FAISS로 교체
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="반도체 정비노트 검색&추천", layout="wide")
st.title("반도체 정비노트 검색 & 성공률 추천")

# OpenAI API Key 입력
api_key = st.text_input("OpenAI API Key를 입력하세요", type="password")
if not api_key:
    st.warning("API 키를 입력해야 진행 가능합니다.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# 엑셀 업로드
uploaded_file = st.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.info("엑셀 파일을 업로드하면 분석이 시작됩니다.")
    st.stop()

# 데이터 로드 및 성공률 계산
df = pd.read_excel(uploaded_file)
df = df.dropna(subset=['정비노트'])
st.success(f"업로드 완료: 총 {len(df)} 행")

all_texts = [str(note).strip() for note in df['정비노트']]
lines = []
for note in all_texts:
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
    cause_match = cause_pattern.search(line)
    if cause_match:
        cause = normalize_cause(cause_match.group(1).strip())
        continue
    if cause is None:
        continue

    m1 = first_action_pattern.search(line)
    m2 = second_action_pattern.search(line)
    m3 = third_action_pattern.search(line)

    if m1:
        action = m1.group(1).strip()
        cause_action_counts[cause][action]['first'] += 1
    elif m2:
        action = m2.group(1).strip()
        cause_action_counts[cause][action]['second'] += 1
    elif m3:
        action = m3.group(1).strip()
        cause_action_counts[cause][action]['third'] += 1

rows = []
for cause, actions in cause_action_counts.items():
    for action, counts in actions.items():
        first = counts.get('first', 0)
        second = counts.get('second', 0)
        third = counts.get('third', 0)
        total = first + second + third
        success = second + third
        rate = round(success / total * 100, 2) if total > 0 else 0
        rows.append({"대표원인": cause, "조치": action, "총횟수": total, "실패횟수": first, "성공횟수": success, "성공률(%)": rate})

df_success = pd.DataFrame(rows)

# LangChain RAG

# 문서 및 벡터 DB 구성
documents = [Document(page_content=str(row['정비노트']), metadata={'row': idx}) for idx, row in df.iterrows()]
split_docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

if os.path.exists("faiss_index"):
    vectordb = FAISS.load_local("faiss_index", embedding_model)
else:
    vectordb = FAISS.from_documents(split_docs, embedding=embedding_model)
    vectordb.save_local("faiss_index")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(search_kwargs={'k': 20}), return_source_documents=True)

# 탭 구성
tab1, tab2 = st.tabs(["정비 검색 & 추천", "통계 자료"])

with tab1:
    example_keywords = ["wafer not", "plasma ignition failure", "pumpdown 시간 지연", "slot valve 동작 불량", "RF auto match 불량"]
    st.markdown(f"예시 키워드: {' | '.join(example_keywords)}")
    st.subheader("검색할 정비 이슈를 입력하세요")
    query = st.text_input("", key="search_query", placeholder="정비 이슈를 입력 후 Enter")
    st.markdown("Enter 키를 누르면 검색이 시작됩니다.")

    if query.strip():
        progress = st.empty()
        bar = progress.progress(0)
        for p in range(0, 101, 10):
            time.sleep(0.1)
            bar.progress(p)
        progress.empty()

        st.success(f"'{query}' 검색 완료")
        output = qa_chain({"query": query})
        docs = output['source_documents']

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
            st.warning("검색된 사례가 없습니다.")
        else:
            dedup = {}
            for r in recommended:
                act = r["조치"]
                if act not in dedup or dedup[act]["성공률"] < r["성공률"]:
                    dedup[act] = r
            top3 = sorted(dedup.values(), key=lambda x: x["성공률"], reverse=True)[:3]

            st.subheader("성공률 상위 3개 조치")
            for idx, r in enumerate(top3, 1):
                st.markdown(f"{idx}) {r['조치']} - {r['성공률']}%")

            top1 = top3[0]
            note_html = top1['정비노트'].replace("\n", "<br>")
            st.markdown("### 최근 실제 수행 사례")
            st.markdown(f"""
<div style="border:2px solid #B0C4DE; padding:15px; background-color:#F3F7FF;">
<b>조치명:</b> {top1['조치']}<br>
<b>장비:</b> {top1['장비ID']} / {top1['모델']}<br>
<b>정비종류:</b> {top1['정비종류']}<br>
<b>정비자:</b> {top1['정비자']}<br><br>
<b>정비노트:</b><br>{note_html}
</div>
""", unsafe_allow_html=True)

            for idx, r in enumerate(top3[1:], 2):
                with st.expander(f"Top {idx} 조치"):
                    note_html = r['정비노트'].replace("\n", "<br>")
                    st.markdown(f"""
<div style="border:2px solid #B0C4DE; padding:15px; background-color:#F3F7FF;">
<b>조치명:</b> {r['조치']}<br>
<b>장비:</b> {r['장비ID']} / {r['모델']}<br>
<b>정비종류:</b> {r['정비종류']}<br>
<b>정비자:</b> {r['정비자']}<br><br>
<b>정비노트:</b><br>{note_html}
</div>
""", unsafe_allow_html=True)

with tab2:
    st.subheader("정비 통계 자료")

    st.subheader("가장 많이 고장난 장비 TOP5")
    top5_equip = df['모델'].value_counts().head(5)
    max_count = top5_equip.values.max()
    fig1 = px.bar(
        x=top5_equip.index,
        y=top5_equip.values,
        text=top5_equip.values,
        color=top5_equip.values,
        color_continuous_scale='Blues'
    )
    fig1.update_traces(textposition='outside')
    fig1.update_layout(title="장비별 고장 빈도 TOP5", xaxis_title="장비 모델", yaxis_title="고장 횟수", height=400, yaxis=dict(range=[0, max_count + 5]))
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("가장 많이 발생한 원인 TOP5")
    top5_cause = df_success.groupby('대표원인')['총횟수'].sum().nlargest(5)
    fig2 = px.bar(
        x=top5_cause.values,
        y=top5_cause.index,
        orientation='h',
        text=top5_cause.values,
        color=top5_cause.values,
        color_continuous_scale='OrRd'
    )
    fig2.update_traces(textposition='outside')
    fig2.update_layout(title="가장 많이 발생한 원인 TOP5", xaxis_title="발생 횟수", yaxis_title="대표 원인", height=400)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("정비자별 평균 조치 성공률 TOP5")
    df_success['row'] = df_success.index
    df_joined = df.merge(df_success, left_index=True, right_on='row', how='left')
    eng_stats = df_joined.groupby("정비자")["성공률(%)"].mean().dropna().round(1).sort_values(ascending=False).head(5)
    max_success = eng_stats.values.max()
    fig3 = px.bar(
        x=eng_stats.index,
        y=eng_stats.values,
        text=eng_stats.values,
        color=eng_stats.values,
        color_continuous_scale="Greens",
        labels={"x": "정비자", "y": "평균 성공률"}
    )
    fig3.update_traces(textposition='outside')
    fig3.update_layout(title="정비자별 평균 성공률 (TOP5)", xaxis_title="정비자", yaxis_title="성공률 (%)", height=400, yaxis=dict(range=[0, max_success + 10]))
    st.plotly_chart(fig3, use_container_width=True)
