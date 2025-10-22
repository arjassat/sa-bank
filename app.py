import streamlit as st
import pandas as pd
import pdfplumber
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


# --- 1. CONFIGURATION AND BANK SPECIFIC RULES ---

# Strong list-based rules for detection (used by detect_bank_format)
BANK_RULES_LIST = [
    ("FNB", ["fnb.co.za"]),
    ("FNB", ["first national bank", "account"]),
    
    ("ABSA", ["absa", "investment account"]),
    ("ABSA", ["absa", "bank limited", "account"]),
    
    ("STANDARD", ["standard bank", "private banking"]),
    ("STANDARD", ["standard bank", "current account"]),
    
    ("HBZ", ["hbz bank limited", "account"]),
    
    ("NEDBANK", ["nedbank limited", "account"]),
    ("CAPITEC", ["capitec bank", "account"]),
]

# Map for parsing logic once the bank is detected (used by parse_pdf_data)
BANK_RULES_MAP = {
    "FNB": {
        "columns": ["Date", "Description", "Amount", "Balance"],
        "table_settings": {
            "vertical_strategy": "text", 
            "horizontal_strategy": "text",
            "explicit_vertical_lines": [30, 80, 480, 540],
            "snap_y_tolerance": 5,
            "min_words_vertical": 2
        },
        "standardize_func": "standardize_fnb"
    },
    "STANDARD": {
        "columns": ["Details", "Service Fee", "Debits", "Credits", "Date", "Balance"],
        "table_settings": {
            "vertical_strategy": "lines", 
            "horizontal_strategy": "text",
            "explicit_vertical_lines": [30, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
            "snap_y_tolerance": 3
        },
        "standardize_func": "standardize_standard"
    },
    "ABSA": {
        "columns": ["date", "transaction amount R", "description", "balance R"],
        "table_settings": {
            "vertical_strategy": "text",
            "explicit_vertical_lines": [30, 80, 400, 550],
            "snap_y_tolerance": 5
        },
        "standardize_func": "standardize_absa"
    },
    "HBZ": {
        "columns": ["Date", "Particulars", "Debit", "Credit", "Reference"],
        "table_settings": {
            "vertical_strategy": "lines",
            "snap_y_tolerance": 3
        },
        "standardize_func": "standardize_hbz"
    },
    # Default/Generic fallback rule for unknown banks
    "GENERIC": {
        "columns": ['Date', 'Description', 'Amount', 'Balance'], 
        "table_settings": {"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 5},
        "standardize_func": "generic_fallback"
    }
}


# --- 2. HELPER FUNCTIONS ---

def detect_bank_format(text_content: str) -> str:
    """Identifies the bank statement format using robust, case-insensitive keyword matching."""
    lower_text = re.sub(r'\s+', ' ', text_content.lower())

    for bank, keywords in BANK_RULES_LIST:
        if all(keyword in lower_text for keyword in keywords):
            return bank
            
    return "GENERIC"

def clean_value(value):
    """
    Cleans numeric values by handling SA (comma for decimal, space/dot for thousands) format.
    """
    if not isinstance(value, str): 
        return None
        
    value = str(value).strip().replace('\n', '').replace('\r', '') 
    
    # 1. Remove currency symbols and merge spaces between digits
    value = re.sub(r'[R$]', '', value, flags=re.IGNORECASE)
    value = re.sub(r'(\d)\s+(\d)', r'\1\2', value) 

    # 2. Remove thousands separators (dot or space)
    if re.search(r'\d\.\d{3}(?:,\d{2})?', value):
        value = value.replace('.', '')
        
    # 3. Replace the South African decimal comma with a standard dot
    value = value.replace(',', '.')
    
    # 4. Clean up formatting indicators (Dr/Cr)
    value = value.replace('Cr', '').replace('Dr', '-').strip()
    
    # 5. Final aggressive cleanup to remove non-numeric/non-dot/non-sign characters
    value = re.sub(r'[^\d\.\-]+', '', value) 
        
    try:
        # Check if it looks like a number before converting
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

# --- 3. STANDARDIZATION FUNCTIONS (FEE FILTERING REMOVED) ---

def standardize_fnb(df):
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    def calculate_fnb_amount(row):
        amount = clean_value(row.get('amount'))
        if amount is None: return None
        is_debit = ('dr' in str(row.get('balance', '')).lower() or 
                    '-' in str(row.get('amount', '')))
        return -abs(amount) if is_debit else abs(amount)

    df['Amount'] = df.apply(calculate_fnb_amount, axis=1)
    df['Date'] = df['date'].astype(str)
    df['Description'] = df['description'].astype(str)
    
    return df[['Date', 'Description', 'Amount']]

def standardize_standard(df):
    """
    FIX: Subtracts the Service Fee from Debits/Credits, but keeps the resulting base transaction line.
    No fee filtering is applied here.
    """
    
    # 1. Column Header Check/Remapping
    if 'Debits' not in df.columns and len(df.columns) == 6:
        st.warning("Standard Bank column headers not detected. Applying fixed column names...")
        df.columns = ["Details", "Service Fee", "Debits", "Credits", "Date", "Balance"]
    
    required_cols = ["Details", "Debits", "Credits", "Date", "Service Fee"] 
    if not all(col in df.columns for col in required_cols):
        raise KeyError(f"Standard Bank parsing failed: Expected columns {required_cols} are not in the extracted DataFrame.")
        
    # 2. Clean values
    df['Debits'] = df['Debits'].apply(clean_value)
    df['Credits'] = df['Credits'].apply(clean_value)
    df['Service Fee'] = df['Service Fee'].apply(clean_value)
    
    # 3. FEE ADJUSTMENT: Subtract the fee from the Debit/Credit amount on lines where it is co-located
    # This is the core fix to change 804.30 to 800.00.
    fee_magnitude = abs(df['Service Fee'].fillna(0))
    
    # Adjust Debits:
    debit_fee_mask = (df['Service Fee'].notna() & df['Debits'].notna())
    # New Debits = Old Debits - Fee amount (ensures base transaction amount remains)
    df.loc[debit_fee_mask, 'Debits'] = df.loc[debit_fee_mask, 'Debits'] - fee_magnitude.loc[debit_fee_mask]

    # Adjust Credits (less common):
    credit_fee_mask = (df['Service Fee'].notna() & df['Credits'].notna())
    df.loc[credit_fee_mask, 'Credits'] = df.loc[credit_fee_mask, 'Credits'] - fee_magnitude.loc[credit_fee_mask]
        
    # 4. Recalculate Amount after fee removal
    df['Amount'] = df['Credits'].fillna(0) - df['Debits'].fillna(0)
    
    # 5. Final Standardization
    df['Date'] = df['Date'].astype(str)
    df['Description'] = df['Details'].astype(str)
    
    # Final cleanup to remove rows where the amount is effectively zero
    df = df[df['Amount'].abs() > 0.01] 
    
    return df[['Date', 'Description', 'Amount']]

def standardize_absa(df):
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    df['Amount'] = df['transaction_amount_r'].apply(clean_value)
    df['Date'] = df['date'].astype(str)
    df['Description'] = df['description'].astype(str)
    
    return df[['Date', 'Description', 'Amount']]
    
def standardize_hbz(df):
    df['Debit'] = df['Debit'].apply(clean_value)
    df['Credit'] = df['Credit'].apply(clean_value)
    
    df['Amount'] = df['Credit'].fillna(0) - df['Debit'].fillna(0)
    df['Date'] = df['Date'].astype(str)
    df['Description'] = df['Particulars'].astype(str)
    
    return df[['Date', 'Description', 'Amount']]

def generic_fallback(df):
    st.warning("Could not apply specific standardization. CSV will contain raw extracted columns.")
    return df 

# --- 4. CORE EXTRACTION LOGIC (GEMINI PROMPT REVERTED TO NEUTRAL) ---

def gemini_extract_from_pdf(pdf_file_path: BytesIO, file_name: str) -> pd.DataFrame:
    """
    FALLBACK: Uses Gemini's vision capability for extraction.
    """
    st.info("🔄 **Initiating Gemini OCR/Extraction Fallback...** (This may take a moment)")
    if not client:
        st.error("Gemini client is inactive. Cannot run OCR fallback.")
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
            st.success(f"Gemini Fallback successful! Extracted {len(data)} transactions.")
            return pd.DataFrame(data)
        else:
            st.error("Gemini extracted a result, but it was empty or not a valid list of transactions.")
            return pd.DataFrame()
            
    except APIError as e:
        st.error(f"Gemini API Error: {e}. Check your API key or contact support.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Gemini Fallback failed due to an unexpected error. Error: {e}")
        return pd.DataFrame()


def parse_pdf_data(pdf_file_path, file_name):
    """Core function to extract tables, detect bank, and standardize the data."""
    all_transactions = pd.DataFrame()
    bank_name = "GENERIC"
    
    try:
        # 1. First attempt: Use pdfplumber for detection and extraction
        with pdfplumber.open(pdf_file_path) as pdf:
            
            full_text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
            bank_name = detect_bank_format(full_text)
            
            rules = BANK_RULES_MAP.get(bank_name, BANK_RULES_MAP["GENERIC"])

            if bank_name == "GENERIC":
                st.warning("⚠️ Could not reliably identify the bank statement format.")
            else:
                st.info(f"✅ Detected Bank: **{bank_name}**. Applying custom parsing rules.")
            
            
            # 2. Extract Data Page by Page using pdfplumber's rules
            for page in pdf.pages:
                tables = page.extract_tables(rules["table_settings"])
                
                for table in tables:
                    if table and len(table) > 1:
                        # Skip header rows
                        start_index = 0
                        for i, row in enumerate(table):
                            if any("BAL" in str(cell).upper() for cell in row if cell) or \
                               any("AMOUNT" in str(cell).upper() for cell in row if cell):
                                start_index = i + 1
                                break
                        data_rows = table[start_index:]
                        
                        if not data_rows: continue
                        
                        df = pd.DataFrame(data_rows)
                        all_transactions = pd.concat([all_transactions, df], ignore_index=True)
                            
        # 3. Standardization & Final Formatting (Only if pdfplumber found data)
        if not all_transactions.empty:
            all_transactions.dropna(thresh=2, inplace=True)
            
            standardize_func_name = rules["standardize_func"]
            if standardize_func_name == "standardize_fnb":
                df_final = standardize_fnb(all_transactions.copy())
            elif standardize_func_name == "standardize_standard":
                df_final = standardize_standard(all_transactions.copy())
            elif standardize_func_name == "standardize_absa":
                df_final = standardize_absa(all_transactions.copy())
            elif standardize_func_name == "standardize_hbz":
                df_final = standardize_hbz(all_transactions.copy())
            else: 
                df_final = generic_fallback(all_transactions.copy())
                
            df_final.dropna(subset=['Amount'], inplace=True)
            
            if not df_final.empty:
                st.success("pdfplumber extracted transactions successfully.")
                return df_final, bank_name
            else:
                st.warning("pdfplumber extracted data but standardization failed. Falling back to Gemini.")

    except Exception as e:
        st.warning(f"pdfplumber failed for {file_name}. Reason: '{e}'. Trying Gemini fallback...")
        
    # --- PHASE 2: GEMINI FALLBACK (If pdfplumber fails or returns no clean data) ---
    
    pdf_file_path.seek(0)
    
    df_gemini = gemini_extract_from_pdf(pdf_file_path, file_name)
    
    if not df_gemini.empty:
        # Standardize the Gemini-extracted DataFrame to ensure consistent column names
        df_gemini['Date'] = df_gemini['Date'].astype(str)
        df_gemini['Description'] = df_gemini['Description'].astype(str)
        
        # Apply the fixed clean_value to the AI's amount column for safety
        df_gemini['Amount'] = df_gemini['Amount'].apply(lambda x: clean_value(str(x))) 
        df_gemini.dropna(subset=['Amount'], inplace=True)
        
        if not df_gemini.empty:
            return df_gemini[['Date', 'Description', 'Amount']], "AI_EXTRACTED"
        
    st.error(f"Both pdfplumber and Gemini extraction failed for {file_name}. No data extracted.")
    return pd.DataFrame(), "FAILED"

# --- 5. STREAMLIT APP LOGIC ---

if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []

st.set_page_config(page_title="🇿🇦 Free SA Bank Statement to CSV Converter", layout="wide")

st.title("🇿🇦 SA Bank Statement PDF to CSV Converter")
st.markdown("""
    ### Built with Gemini AI for accountants: Free, robust tool for South African bank statement conversion.
    
    **Supported Banks (with custom rules): ABSA, FNB, Standard Bank, HBZ.** Uses **Gemini AI** as a powerful **fallback** for **scanned or difficult PDFs**.
    ---
""")

if client:
    st.sidebar.success("Gemini AI Fallback/OCR: **Active** ✅")
else:
    st.sidebar.warning("Gemini AI Fallback/OCR: **Inactive** 🛑. Please set **GEMINI_API_KEY** in Streamlit Secrets.")


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
        
        # CORE CALL: attempts pdfplumber first, then falls back to Gemini
        df_transactions, bank_name = parse_pdf_data(pdf_data, file_name)

        if not df_transactions.empty and 'Amount' in df_transactions.columns:
            
            # Apply final cleaning and formatting
            df_transactions['Description'] = df_transactions['Description'].apply(clean_description_for_xero)
            
            df_final = df_transactions.rename(columns={
                'Date': 'Date',
                'Description': 'Description',
                'Amount': 'Amount'
            })
            
            # *** IMPORTANT: GLOBAL FEE FILTERING IS REMOVED HERE ***

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
            
            st.success(f"Successfully extracted {len(df_xero)} transactions from {file_name}")

    
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
            file_name="SA_Bank_Statements_Export.csv",
            mime="text/csv"
        )
