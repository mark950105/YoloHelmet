import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. 設定網頁標題與介紹
st.set_page_config(page_title="安全帽配戴即時偵測系統", page_icon="⛑️", layout="centered")
st.title("⛑️ 公共場域行為偵測：安全帽配戴辨識")
st.markdown("本系統由 AI 模型驅動，上傳騎士照片即可自動偵測是否配戴安全帽。")

# 2. 載入 YOLO 模型 (快取機制可以防止網頁每次重整都重新載入，加快速度)
@st.cache_resource
def load_model():
    # 網頁伺服器會讀取與此程式碼放在同一個資料夾的 best.pt
    return YOLO("best.pt")

try:
    model = load_model()
    st.success("AI 模型載入成功！")
except Exception as e:
    st.error(f"模型載入失敗，請確認資料夾內是否有 best.pt 檔案。錯誤訊息: {e}")

# 3. 提供使用者兩種上傳圖片的方式：拍照或從相簿上傳
st.subheader("📸 步驟一：提供測試影像")
source_option = st.radio("請選擇影像來源：", ("上傳照片檔案", "開啟手機/電腦相機拍照"))

uploaded_file = None
if source_option == "上傳照片檔案":
    uploaded_file = st.file_uploader("請選擇一張 JPG 或 PNG 格式的照片", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("請對準鏡頭拍攝畫面")

# 4. 執行模型辨識並顯示結果
if uploaded_file is not None:
    st.subheader("🔍 步驟二：AI 辨識結果")
    
    # 將上傳的檔案轉換為 Pillow 影像格式
    input_image = Image.open(uploaded_file)
    
    # 在網頁上秀出使用者的原圖
    st.image(input_image, caption="原始影像", use_container_width=True)
    
    with st.spinner("AI 正在辨識中，請稍候..."):
        # 呼叫 YOLO 進行預測
        results = model.predict(input_image, conf=0.25) # conf 為門檻值，可調整
        
        # 取得繪製好標籤框的結果影像 (BGR 格式轉 RGB)
        res_plotted = results[0].plot()
        output_image = Image.fromarray(res_plotted[..., ::-1])
        
        # 顯示辨識後的結果
        st.image(output_image, caption="AI 偵測結果（未配戴安全帽將被框選標註）", use_container_width=True)
        st.success("辨識完成！")