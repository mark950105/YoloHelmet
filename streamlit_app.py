import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# 1. 設定網頁標題
st.set_page_config(page_title="安全帽配戴即時偵測系統", page_icon="⛑️", layout="centered")
st.title("⛑️ 公共場域行為偵測：安全帽配戴辨識")

# 2. 強制檢查目前雲端資料夾底下的所有檔案
st.sidebar.header("📁 雲端伺服器檔案檢查")
current_files = os.listdir(".")
st.sidebar.write("目前目錄下的檔案清單：", current_files)

# 強制抓取當前目錄下的 best.pt 絕對路徑
model_path = os.path.abspath("best.pt")

if "best.pt" in current_files:
    file_size = os.path.getsize(model_path) / (1024 * 1024) # 轉成 MB
    st.sidebar.success(f"確認找到 best.pt！檔案大小: {file_size:.2f} MB")
else:
    st.sidebar.error("❌ 警告：伺服器目錄找不到 best.pt！請檢查 GitHub 是否上傳成功。")

# 3. 載入 YOLO 模型 (移除快取機制，強制每次都重新讀取實體檔案)
try:
    # 強制傳入絕對路徑，若檔案有問題會直接報錯，不會偷偷下載官方模型
    model = YOLO(model_path)
    
    # 印出這顆模型到底能認出哪些東西（自訂安全帽模型應該只會認出 helmet 等）
    st.sidebar.write("🎯 目前模型內建的標籤類別：", model.names)
    st.success("AI 實體模型載入成功！")
except Exception as e:
    st.error(f"💥 模型載入發生嚴重錯誤！錯誤訊息: {e}")

# 4. 側邊欄設定門檻值
conf_threshold = st.sidebar.slider("調整 AI 信心度門檻 (Confidence)", 0.05, 1.0, 0.25, 0.05)

# 5. 提供影像來源
st.subheader("📸 步驟一：提供測試影像")
source_option = st.radio("請選擇影像來源：", ("上傳照片檔案", "開啟手機/電腦相機拍照"))

uploaded_file = None
if source_option == "上傳照片檔案":
    uploaded_file = st.file_uploader("請選擇一張 JPG 或 PNG 格式的照片", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("請對準鏡頭拍攝畫面")

# 6. 執行模型辨識
if uploaded_file is not None:
    st.subheader("🔍 步驟二：AI 辨識結果")
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="原始影像", use_container_width=True)
    
    with st.spinner("AI 正在辨識中..."):
        results = model(input_image, conf=conf_threshold)
        res_plotted = results[0].plot()
        output_image = Image.fromarray(res_plotted)
        st.image(output_image, caption="YOLO 官方原生偵測結果", use_container_width=True)
        
        st.write("📋 伺服器後台偵測到的原始數據如下：")
        st.json(results[0].summary())
