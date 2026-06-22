import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. 設定網頁標題與介紹
st.set_page_config(page_title="安全帽配戴即時偵測系統", page_icon="⛑️", layout="centered")
st.title("⛑️ 公共場域行為偵測：安全帽配戴辨識")
st.markdown("本系統由 AI 模型驅動，上傳騎士照片即可自動偵測是否配戴安全帽。")

# 2. 載入 YOLO 模型
@st.cache_resource
def load_model():
    # 這裡載入你的模型
    return YOLO("best.pt")

try:
    model = load_model()
    st.success("AI 模型載入成功！")
except Exception as e:
    st.error(f"模型載入失敗，請確認資料夾內是否有 best.pt 檔案。錯誤訊息: {e}")

# 側邊欄設定門檻值
st.sidebar.header("⚙️ 模型的參數設定")
conf_threshold = st.sidebar.slider("調整 AI 信心度門檻 (Confidence)", 0.05, 1.0, 0.25, 0.05)

# 3. 提供使用者兩種上傳圖片的方式
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
    
    # 讀取圖片
    input_image = Image.open(uploaded_file).convert("RGB")
    
    # 顯示原圖
    st.image(input_image, caption="原始影像", use_container_width=True)
    
    with st.spinner("AI 正在辨識中，請稍候..."):
        # 用最標準的方式讓 YOLO 預測並自己繪圖
        results = model(input_image, conf=conf_threshold)
        
        # 取得 YOLO 原生畫好框的 BGR 圖片矩陣，並轉回 RGB
        res_plotted = results[0].plot()
        output_image = Image.fromarray(res_plotted)
        
        # 顯示最終結果
        st.image(output_image, caption="YOLO 官方原生偵測結果", use_container_width=True)
        
        # 在下方印出模型真正抓到的類別和標籤，用來當場檢查
        st.write("📋 伺服器後台偵測到的原始數據如下：")
        st.json(results[0].summary())
