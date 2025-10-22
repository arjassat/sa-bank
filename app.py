import streamlit as st
import pandas as pd
# pdfplumber is no longer required
import re
from io import BytesIO
import os

# --- AI INTEGRATION SETUP ---
from google import genai
from google.genai import types 
from google.genai.errors import APIError

@st.cache_resource(show_spinner=False)
def get_gemini_client():
    """Initializes the Gemini client using the key from Streamlit Secrets."""
    if "GEMINI_API_KEY" not in st.secrets:
        return None
    
    api_key = st.secrets["GEMINI_API_KEY"]
    os.environ["GEMINI_API_KEY"] = api_key
    
    try:
        return genai.Client()
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None

client = get_gemini_client()
# --- END AI INTEGRATION SETUP ---


# --- 2. HELPER FUNCTIONS ---

def clean_value(value):
    """
    Cleans numeric values by handling SA (comma for decimal, space/dot for thousands) format.
    Kept as a safety net for the AI's JSON output.
    """
    if not isinstance(value, str): 
        # Handle direct float/int from AI's JSON output
        if isinstance(value, (int, float)):
            return float(value)
        return None
        
    value = str(value).strip().replace('\n', '').replace('\r', '') 
    
    # 1. Remove currency symbols and merge spaces between digits
    value = re.sub(r'[R$]', '', value, flags=re.IGNORECASE)
    value = re.sub(r'(\d)\s+(\d)', r'\1\2', value) 

    # 2. Handle South African formatting (1 000,00 or 1.000,00)
    if ',' in value:
        value = value.replace('.', '').replace(' ', '')
    else: # Handle standard dot-as-decimal format if no comma is present
        value = value.replace(' ', '')

    # 3. Replace the South African decimal comma with a standard dot
    value = value.replace(',', '.')
    
    # 4. Clean up formatting indicators (Dr/Cr)
    value = value.replace('Cr', '').replace('Dr', '-').strip()
    
    # 5. Final aggressive cleanup to remove non-numeric/non-dot/non-sign characters
    value = re.sub(r'[^\d\.\-]+', '', value) 
        
    try:
        if re.match(r'^-?\d*\.?\d+$', value):
            return float(value)
        return None
    except:
        return None

def clean_description_for_xero(description):
    """Cleans up transaction descriptions for easy Xero reconciliation."""
    if not isinstance(description, str): return ""
        
    description = description.strip()
    
    # Remove common reference/date patterns left over by extraction
    description = re.sub(r'\s*\d{6}\s+\d{4}\s+\d{2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', '', description, flags=re.IGNORECASE)
    description = re.sub(r'(?:Ref\s*|Reference\s*|No\s*|Nr\s*|ID\s*):\s*[\w\d\-]+', '', description, flags=re.IGNORECASE)
    description = re.sub(r'Serial:\d+/\d+', '', description)

    # Remove common transaction type prefixes
    description = re.sub(r'(?:POS Purchase|ATM Withdrawal|Immediate Payment|Internet Pmt To|Teller Transfer Debit|Direct Credit|EFT|IB Payment)\s*', '', description, flags=re.IGNORECASE)
    
    description = re.sub(r'\s{2,}', ' ', description).strip(' -').strip()
    
    return description

# --- 4. CORE EXTRACTION LOGIC (GEMINI ONLY) ---

def gemini_extract_from_pdf(pdf_file_path: BytesIO, file_name: str) -> pd.DataFrame:
    """
    PRIMARY METHOD: Uses Gemini's vision capability for extraction based on strict JSON rules.
    This function's prompt is designed to separate main transactions and fees.
    """
    st.info("🔄 **Initiating Gemini AI Extraction...** (This may take a moment)")
    if not client:
        st.error("Gemini client is inactive. Cannot run AI extraction.")
        return pd.DataFrame()

    try:
        pdf_part = types.Part.from_bytes(
            data=pdf_file_path.getvalue(),
            mime_type='application/pdf'
        )
        
        prompt = f"""
            You are a South African accounting assistant. Your task is to extract all **transaction lines** from the provided bank statement PDF ({file_name}). 
            The output **must be a single JSON list** where each item is a transaction.
            
            **Required Columns (MUST be included):**
            1.  'Date': The transaction date (in any format, e.g., '2024/03/16', '16 Mar', '03 16').
            2.  'Description': A clean, single-line description of the transaction, including any fee description if provided on its own line.
            3.  'Amount': The Rand amount for that line item. Debits must be negative (e.g., -100.00), Credits must be positive (e.g., 500.00). **CRITICAL: If a transaction amount and a fee amount are listed separately, list them as two separate line items.**

            **Example JSON Output (Mandatory format):**
            [
              {{
                "Date": "16 Mar 2024",
                "Description": "IMMEDIATE PAYMENT 145089014 FAWZIA BAYAT",
                "Amount": -2500.00
              }},
              {{
                "Date": "16 Mar 2024",
                "Description": "FEE IMMEDIATE PAYMENT",
                "Amount": -4.30
              }},
              {{
                "Date": "17 Mar 2024",
                "Description": "CREDIT TRANSFER SALARY DEPOSIT",
                "Amount": 15000.00
              }}
            ]
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, pdf_part],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )

        import json
        
        json_text = response.text.strip().strip('```json').strip('```')
        
        data = json.loads(json_text)
        
        if isinstance(data, list) and data:
            st.success(f"Gemini AI Extraction successful! Extracted {len(data)} transactions.")
            return pd.DataFrame(data)
        else:
            st.error("Gemini extracted a result, but it was empty or not a valid list of transactions.")
            return pd.DataFrame()
            
    except APIError as e:
        st.error(f"Gemini API Error: {e}. Check your API key or contact support.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Gemini AI Extraction failed due to an unexpected error. Error: {e}")
        return pd.DataFrame()


def parse_pdf_data(pdf_file_path, file_name):
    """
    Modified core function: Now uses only Gemini AI for extraction.
    This enforces the desired output format of separated fees.
    """
    
    pdf_file_path.seek(0) # Ensure the file pointer is at the beginning
    
    # --- PHASE 1: GEMINI EXTRACTION (Now the sole method) ---
    df_gemini = gemini_extract_from_pdf(pdf_file_path, file_name)
    
    if not df_gemini.empty:
        # Standardization and cleaning steps
        required_cols = ['Date', 'Description', 'Amount']
        if not all(col in df_gemini.columns for col in required_cols):
             st.error("AI output is missing required columns (Date, Description, Amount).")
             return pd.DataFrame(), "FAILED"

        df_gemini['Date'] = df_gemini['Date'].astype(str)
        df_gemini['Description'] = df_gemini['Description'].astype(str)
        
        # Apply the fixed clean_value to the AI's amount column for safety
        df_gemini['Amount'] = df_gemini['Amount'].apply(lambda x: clean_value(x)) 
        df_gemini.dropna(subset=['Amount'], inplace=True)
        
        if not df_gemini.empty:
            return df_gemini[['Date', 'Description', 'Amount']], "AI_ONLY"

    st.error(f"Gemini AI extraction failed for {file_name}. No data extracted.")
    return pd.DataFrame(), "FAILED"


# --- 5. STREAMLIT APP LOGIC ---

if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []

st.set_page_config(page_title="🇿🇦 Free SA Bank Statement to CSV Converter (AI Only)", layout="wide")

st.title("🇿🇦 SA Bank Statement PDF to CSV Converter (AI-Only)")
st.markdown("""
    ### Now using **Gemini AI exclusively** for robust extraction and separate fee line items.
    
    This version skips the PDF table parsing to ensure **all transactions and bank fees are captured on separate lines** for easier Xero/accounting reconciliation.
    ---
""")

if client:
    st.sidebar.success("Gemini AI Engine: **Active** ✅ (Exclusive Use)")
else:
    st.sidebar.warning("Gemini AI Engine: **Inactive** 🛑. Please set **GEMINI_API_KEY** in Streamlit Secrets.")


uploaded_files = st.file_uploader(
    "Upload your bank statement PDF files (Multiple files supported)",
    type=["pdf"],
    accept_multiple_files=True,
    key="unique_pdf_uploader_fixed" 
)

# --- PROCESSING STARTS HERE ---
if uploaded_files:
    st.subheader("Processing Files...")
    
    all_df = []
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        st.markdown(f"**Processing:** `{file_name}`")
        
        pdf_data = BytesIO(uploaded_file.read())
        
        # CORE CALL: Now exclusively uses the AI logic
        df_transactions, source = parse_pdf_data(pdf_data, file_name)

        if not df_transactions.empty and 'Amount' in df_transactions.columns:
            
            # Apply final cleaning and formatting
            df_transactions['Description'] = df_transactions['Description'].apply(clean_description_for_xero)
            
            df_final = df_transactions.rename(columns={
                'Date': 'Date',
                'Description': 'Description',
                'Amount': 'Amount'
            })
            
            try:
                # Attempt to parse date in SA format (day/month/year)
                df_final['Date'] = pd.to_datetime(df_final['Date'], errors='coerce', dayfirst=True).dt.strftime('%d/%m/%Y')
            except Exception as e:
                st.warning(f"Could not standardize dates for {file_name}. Dates remain in raw format. Error: {e}")
            
            # Final structure: Date, Description, Amount
            df_xero = pd.DataFrame({
                'Date': df_final['Date'].fillna(''),
                'Description': df_final['Description'].astype(str),
                'Amount': df_final['Amount'].round(2), 
            })
            
            # Ensure the order is exactly Date, Description, Amount
            df_xero = df_xero[['Date', 'Description', 'Amount']]
            
            df_xero.dropna(subset=['Date', 'Amount'], inplace=True)
            
            all_df.append(df_xero)
            
            st.success(f"Successfully extracted {len(df_xero)} transactions from {file_name} using **Gemini AI**.")

    
    # --- 6. COMBINE AND DOWNLOAD ---

    if all_df:
        final_combined_df = pd.concat(all_df, ignore_index=True)
        
        st.markdown("---")
        st.subheader("✅ All Transactions Combined and Ready for Download")
        
        st.dataframe(final_combined_df)
        
        # Convert DataFrame to CSV for download
        csv_output = final_combined_df.to_csv(index=False, sep=',', encoding='utf-8')
        st.download_button(
            label="⬇️ Download Bank Statement CSV File",
            data=csv_output,
            file_name="SA_Bank_Statements_AI_Export.csv",
            mime="text/csv"
        )
