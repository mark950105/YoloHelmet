import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw
import io

# 1. 設定網頁標題
st.set_page_config(page_title="安全帽配戴即時偵測系統", page_icon="⛑️", layout="centered")
st.title("⛑️ 公共場域行為偵測：安全帽配戴辨識")
st.markdown("本系統由 Roboflow API 驅動，上傳照片即可自動偵測是否配戴安全帽。")

# 2. 初始化 Roboflow API (請填入你自己的 API Key 與專案資訊)
# 註：如果這部分對非資工背景太難，可以在 Roboflow 的 "Deploy" 頁面找到這些程式碼參數
API_KEY = "你的_ROBOFLOW_API_KEY"       # 替換成你的 Private API Key
PROJECT_ID = "你的_專案名稱"            # 替換成你的專案 URL 名稱 (例如 helmet-detection-xxxx)
VERSION = 1                             # 替換成你訓練的模型版本號碼

@st.cache_resource
def init_roboflow():
    rf = Roboflow(api_key=API_KEY)
    project = rf.project(PROJECT_ID)
    return project.model(VERSION)

try:
    model = init_roboflow()
    st.success("Roboflow 雲端模型連線成功！")
except Exception as e:
    st.error("模型連線失敗，請確認 API Key 與專案設定。")

# 3. 提供影像來源
st.subheader("📸 步驟一：提供測試影像")
source_option = st.radio("請選擇影像來源：", ("上傳照片檔案", "開啟手機/電腦相機拍照"))

uploaded_file = None
if source_option == "上傳照片檔案":
    uploaded_file = st.file_uploader("請選擇一張 JPG 或 PNG 格式的照片", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("請對準鏡頭拍攝畫面")

# 4. 執行模型辨識
if uploaded_file is not None:
    st.subheader("🔍 步驟二：AI 辨識結果")
    
    # 讀取並顯示原圖
    image = Image.open(uploaded_file)
    st.image(image, caption="原始影像", use_container_width=True)
    
    # 將圖片轉換成 bytes 準備傳給 API
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    with st.spinner("Roboflow 雲端辨識中..."):
        try:
            # 呼叫 Roboflow API 進行預測並直接取得繪製好框線的圖片
            # 這樣我們就不需要寫複雜的畫框程式碼
            result_img_bytes = model.predict_image_bytes(img_bytes, hosted=True)
            
            # 轉換回 Pillow 格式並秀在網頁上
            output_image = Image.open(io.BytesIO(result_img_bytes))
            st.image(output_image, caption="AI 偵測結果", use_container_width=True)
            st.success("辨識完成！")
        except Exception as e:
            st.error(f"辨識過程中發生錯誤: {e}")
