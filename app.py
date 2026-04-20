import streamlit as st
import pandas as pd
import numpy as np
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import tempfile
import os
import re
from datetime import datetime

# --- 1. THE AI ENGINE SETUP ---
@st.cache_resource
def load_ocr_model():
    # Load the PP-OCRv4 model (High accuracy for tables)
    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

ocr = load_ocr_model()

# --- 2. THE SOUTH AFRICAN BANK PARSER ---
class SABankParser:
    def __init__(self, raw_text_list):
        self.lines = raw_text_list
        self.data = []

    def clean_amount(self, text):
        """Standardizes SA currency formats like 1 200,50 or 1,200.50"""
        clean = re.sub(r'[^\d,\.-]', '', text).replace(' ', '')
        if ',' in clean and '.' in clean: # 1,200.50
            clean = clean.replace(',', '')
        elif ',' in clean: # 1200,50
            clean = clean.replace(',', '.')
        return clean

    def parse(self):
        """
        Looks for patterns: [Date] [Description] [Amount]
        SA Banks usually use: DD MMM, DD/MM/YYYY, or YYYY-MM-DD
        """
        date_pattern = r'(\d{2} \w{3}|\d{2}/\d{2}/\d{2,4}|\d{4}-\d{2}-\d{2})'
        
        current_row = {}
        for line in self.lines:
            # Check if line starts with a date (New Transaction)
            date_match = re.search(date_pattern, line)
            
            if date_match:
                # Save previous row if it exists
                if current_row: self.data.append(current_row)
                
                # Start new row
                current_row = {
                    "Date": date_match.group(1),
                    "Description": line.replace(date_match.group(1), "").strip(),
                    "Amount": ""
                }
            elif current_row:
                # If no date, but we have a current row, this is likely 
                # a continuation of a description or the amount
                amount_match = re.findall(r'([-]?\d+[\s,.]\d{2})', line)
                if amount_match:
                    current_row["Amount"] = self.clean_amount(amount_match[-1])
                else:
                    current_row["Description"] += f" {line}"

        if current_row: self.data.append(current_row)
        return pd.DataFrame(self.data)

# --- 3. THE USER INTERFACE ---
st.set_page_config(page_title="SA Bank to Xero", page_icon="🇿🇦", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-text { color: #64748B; margin-bottom: 2rem; }
    div.stButton > button { background-color: #1E3A8A; color: white; border-radius: 8px; width: 100%; border: none; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">SA Bank Parser</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Convert scanned PDFs to Xero-ready CSVs using local AI.</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload Statement(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 Process Statements"):
        all_dfs = []
        progress_bar = st.progress(0)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # 1. Convert PDF to Image (Handles scanned & digital)
            images = convert_from_path(tmp_path, dpi=300)
            raw_lines = []
            
            for i, img in enumerate(images):
                st.write(f"Scanning Page {i+1} of {uploaded_file.name}...")
                img_np = np.array(img)
                result = ocr.ocr(img_np, cls=True)
                
                if result[0]:
                    for line in result[0]:
                        raw_lines.append(line[1][0])
            
            # 2. Parse Text
            parser = SABankParser(raw_lines)
            df = parser.parse()
            all_dfs.append(df)
            
            os.remove(tmp_path)
            progress_bar.progress((idx + 1) / len(uploaded_files))

        # 3. Combine and Export
        if all_dfs:
            final_df = pd.concat(all_dfs).reset_index(drop=True)
            
            # Xero Format Mapping
            xero_df = pd.DataFrame({
                "*Date": final_df["Date"],
                "*Amount": final_df["Amount"],
                "Payee": "",
                "Description": final_df["Description"],
                "Reference": ""
            })

            st.success("Successfully Processed!")
            st.dataframe(xero_df, use_container_width=True)
            
            csv = xero_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Xero CSV",
                data=csv,
                file_name=f"Xero_Import_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
