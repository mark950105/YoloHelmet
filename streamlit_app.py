import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# 1. 設定網頁標題與介紹 (這裡已修正為正確的 set_page_config)
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
    
    # 讀取原始圖片
    input_image = Image.open(uploaded_file).convert("RGB")
    img_width, img_height = input_image.size
    
    with st.spinner("AI 正在辨識中，請稍候..."):
        # 呼叫 YOLO 進行預測
        results = model.predict(input_image, conf=conf_threshold)
        result = results[0]
        
        if len(result.boxes) == 0:
            st.image(input_image, caption="原始影像（未偵測到物件）", use_container_width=True)
            st.warning(f"目前信心度設為 {conf_threshold}，AI 沒有偵測到任何物件。")
        else:
            st.info(f"🎉 成功偵測到 {len(result.boxes)} 個物件！正在進行座標修正與繪圖...")
            
            # 複製圖片準備手動畫框
            draw_image = input_image.copy()
            draw = ImageDraw.Draw(draw_image)
            
            # 取得模型的所有類別名稱
            names = model.names
            
            for box in result.boxes:
                # 取得信心度與類別 ID
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # 對應自訂標籤名稱
                label_name = names.get(cls_id, f"Class {cls_id}")
                if label_name == "Class 0" or cls_id == 0:
                    display_label = "Helmet (有戴安全帽)"
                    box_color = "#00FF00"  # 綠色
                elif label_name == "Class 1" or cls_id == 1:
                    display_label = "No Helmet (未戴安全帽)"
                    box_color = "#FF0000"  # 紅色
                else:
                    display_label = f"{label_name}"
                    box_color = "#00FFFF"
                
                # 座標處理
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                
                # 自動判斷並校正 YOLO 格式造成的巨大變形框
                if (x2 - x1) > (img_width * 0.8) and (y2 - y1) > (img_height * 0.8):
                    if hasattr(box, 'xywh') and len(box.xywh) > 0:
                        xc, yc, w, h = box.xywh[0].tolist()
                        if xc <= 1.0:
                            x1 = int((xc - w/2) * img_width)
                            y1 = int((yc - h/2) * img_height)
                            x2 = int((xc + w/2) * img_width)
                            y2 = int((yc + h/2) * img_height)
                        else:
                            x1, y1, x2, y2 = map(int, [xc - w/2, yc - h/2, xc + w/2, yc + h/2])
                else:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # 邊界保護
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_width, x2), min(img_height, y2)
                
                # 畫出精準外框
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=4)
                
                # 寫上標籤與信心度
                text_content = f"{display_label} {conf:.2f}"
                draw.rectangle([x1, y1 - 22, x1 + (len(text_content) * 8) + 10, y1], fill=box_color)
                draw.text((x1 + 5, y1 - 20), text_content, fill="#FFFFFF")
            
            # 顯示結果
            st.image(draw_image, caption="AI 偵測結果", use_container_width=True)
            st.success("安全帽辨識影像渲染成功！")
