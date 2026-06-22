import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 1. 設定網頁標題與介紹
st.set_page_config(page_title="安全帽配戴即時偵測系統", page_icon="⛑️", layout="centered")
st.title("⛑️ 公共場域行為偵測：安全帽配戴辨識")
st.markdown("本系統由 AI 模型驅動，上傳騎士照片即可自動偵測是否配戴安全帽。")

# 2. 載入 YOLO 模型
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
    st.success("AI 模型載入成功！")
except Exception as e:
    st.error(f"模型載入失敗，請確認資料夾內是否有 best.pt 檔案。錯誤訊息: {e}")

# 側邊欄參數設定
st.sidebar.header("⚙️ 模型的參數設定")
conf_threshold = st.sidebar.slider("調整 AI 信心度門檻 (Confidence)", 0.05, 1.0, 0.20, 0.05)

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
    
    # 讀取原始圖片
    input_image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner("AI 正在辨識中，請稍候..."):
        # 呼叫 YOLO 進行預測
        results = model.predict(input_image, conf=conf_threshold)
        result = results[0]
        
        # 檢查有沒有偵測到任何物件
        if len(result.boxes) == 0:
            st.image(input_image, caption="原始影像（未偵測到物件）", use_container_width=True)
            st.warning(f"目前信心度設為 {conf_threshold}，AI 沒有偵測到任何物件。請嘗試把信心度調低！")
        else:
            st.info(f"🎉 成功偵測到 {len(result.boxes)} 個物件！正在手動繪製標籤框...")
            
            # 建立一個可以在上面畫畫的物件
            draw_image = input_image.copy()
            draw = ImageDraw.Draw(draw_image)
            
            # 取得模型的所有類別名稱 (例如 {0: 'helmet', 1: 'no-helmet'})
            names = model.names
            
            # 逐一將偵測到的框抓出來畫圖
            for box in result.boxes:
                # 取得框的左上角與右下角座標 [xmin, ymin, xmax, ymax]
                xyxy = box.xyxy[0].tolist()
                xmin, ymin, xmax, ymax = map(int, xyxy)
                
                # 取得信心度與類別 ID
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label_name = names.get(cls_id, f"Class {cls_id}")
                
                # 設定顏色：通常未配戴安全帽用紅色，戴安全帽用綠色 (根據類別名稱包含 no 自行判斷)
                if "no" in label_name.lower():
                    box_color = "#FF0000"  # 紅色
                else:
                    box_color = "#00FF00"  # 綠色
                
                # 1. 畫出矩形外框 (線條寬度設為 5，看得比較清楚)
                draw.rectangle([xmin, ymin, xmax, ymax], outline=box_color, width=5)
                
                # 2. 畫出標籤文字背景
                text_content = f"{label_name} {conf:.2f}"
                draw.rectangle([xmin, ymin - 25, xmin + (len(text_content) * 12), ymin], fill=box_color)
                
                # 3. 寫上文字
                draw.text((xmin + 5, ymin - 22), text_content, fill="#FFFFFF")
            
            # 顯示最終繪製完畢的圖片
            st.image(draw_image, caption="AI 偵測結果（手動繪製框線版）", use_container_width=True)
            st.success("網頁影像渲染完成！")
