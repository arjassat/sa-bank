import streamlit as st
import pandas as pd
import numpy as np
import easyocr
from pdf2image import convert_from_path
import tempfile
import os
import re
from datetime import datetime

# --- 1. THE AI ENGINE SETUP ---
@st.cache_resource
def load_ocr_model():
    # Load EasyOCR (Free, No API key, works on newer Python)
    return easyocr.Reader(['en'])

reader = load_ocr_model()

# --- 2. THE SOUTH AFRICAN BANK PARSER ---
class SABankParser:
    def __init__(self, raw_text_list):
        self.lines = raw_text_list
        self.data = []

    def clean_amount(self, text):
        """Standardizes SA currency formats like 1 200,50 or 1,200.50"""
        # Remove currency symbols and handle spaces
        clean = re.sub(r'[^\d,\.-]', '', text).replace(' ', '')
        if ',' in clean and '.' in clean: 
            clean = clean.replace(',', '')
        elif ',' in clean: 
            clean = clean.replace(',', '.')
        return clean

    def parse(self):
        # Pattern for common SA date formats
        date_pattern = r'(\d{2} \w{3}|\d{2}/\d{2}/\d{2,4}|\d{4}-\d{2}-\d{2})'
        
        current_row = {}
        for line in self.lines:
            date_match = re.search(date_pattern, line)
            
            if date_match:
                if current_row: self.data.append(current_row)
                current_row = {
                    "Date": date_match.group(1),
                    "Description": line.replace(date_match.group(1), "").strip(),
                    "Amount": ""
                }
            elif current_row:
                # Look for numbers that look like prices
                amount_match = re.findall(r'([-]?\d+[\s,.]\d{2})', line)
                if amount_match:
                    current_row["Amount"] = self.clean_amount(amount_match[-1])
                else:
                    current_row["Description"] += f" {line}"

        if current_row: self.data.append(current_row)
        return pd.DataFrame(self.data)

# --- 3. THE USER INTERFACE ---
st.set_page_config(page_title="SA Bank to Xero", page_icon="🇿🇦")

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; }
    div.stButton > button { background-color: #1E3A8A; color: white; border-radius: 8px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">SA Bank Parser</div>', unsafe_allow_html=True)
st.info("Upload scanned or digital PDFs. Supported: FNB, ABSA, Standard Bank, Nedbank, HBZ.")

uploaded_files = st.file_uploader("Upload Statement(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 Process Statements"):
        all_dfs = []
        
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # Convert PDF to Image
            images = convert_from_path(tmp_path, dpi=200)
            raw_lines = []
            
            for i, img in enumerate(images):
                st.write(f"Reading {uploaded_file.name} - Page {i+1}...")
                img_np = np.array(img)
                # RUN AI
                result = reader.readtext(img_np, detail=0) 
                raw_lines.extend(result)
            
            # Parse
            parser = SABankParser(raw_lines)
            df = parser.parse()
            all_dfs.append(df)
            os.remove(tmp_path)

        if all_dfs:
            final_df = pd.concat(all_dfs).reset_index(drop=True)
            
            # Formatting for Xero
            xero_df = pd.DataFrame({
                "*Date": final_df["Date"],
                "*Amount": final_df["Amount"],
                "Payee": "",
                "Description": final_df["Description"],
                "Reference": ""
            })

            st.success("Extraction Complete!")
            st.dataframe(xero_df, use_container_width=True)
            
            csv = xero_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Xero CSV",
                data=csv,
                file_name=f"Xero_Import_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
