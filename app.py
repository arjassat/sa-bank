import streamlit as st
import pandas as pd
import pdfplumber
import re
from io import BytesIO

# --- AI INTEGRATION SETUP ---
import os
# NOTE: Ensure 'google-genai' is installed (pip install google-genai)

from google import genai
from google.genai import types 

@st.cache_resource(show_spinner=False)
def get_gemini_client():
    """Initializes the Gemini client using the key from Streamlit Secrets."""
    # 1. Check for the key in Streamlit secrets
    if "GEMINI_API_KEY" not in st.secrets:
        return None
    
    # 2. Retrieve the key and set the environment variable
    api_key = st.secrets["GEMINI_API_KEY"]
    os.environ["GEMINI_API_KEY"] = api_key
    
    # 3. Initialize the client
    try:
        # The client will automatically pick up the GEMINI_API_KEY environment variable
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
    """
    Identifies the bank statement format using robust, case-insensitive keyword matching.
    """
    lower_text = re.sub(r'\s+', ' ', text_content.lower())

    for bank, keywords in BANK_RULES_LIST:
        if all(keyword in lower_text for keyword in keywords):
            return bank
            
    return "GENERIC"

def ai_analyze_statement(text_content: str, file_name: str) -> tuple[str, str]:
    """
    Uses Gemini to analyze the statement text and provide a structured hint.
    (UNCOMMENTED AND LIVE API CALL)
    """
    
    if not client:
        return "GENERIC", "AI analysis skipped: Gemini client not initialized. Please set **GEMINI_API_KEY** in Streamlit Secrets."

    prompt = f"""
Analyze the text from a South African bank statement '{file_name}'.
Identify the full name of the bank and suggest the best two to three keywords for a detection rule for the Python code.
Output your analysis in a concise JSON format:
{{
   "identified_bank_name": "Full bank name (e.g., Capitec Bank, Nedbank)",
   "developer_notes": "Brief note on table structure (e.g., Single amount column with +/- sign, or Separate Debit/Credit columns).",
   "suggested_keywords": ["keyword1", "keyword2"]
}}
TEXT CONTENT START: {text_content[:8000]} TEXT CONTENT END
"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        # Parse the JSON response
        import json
        ai_data = json.loads(response.text)
        
        bank = ai_data.get('identified_bank_name', 'Unknown Bank')
        notes = ai_data.get('developer_notes', 'No notes.')
        keywords = ", ".join(ai_data.get('suggested_keywords', []))
        
        return "GENERIC", f"AI Suggestion: **{bank}**. Notes: {notes}. Keywords: {keywords}"
        
    except Exception as e:
        return "GENERIC", f"AI analysis failed: {e}. Check API key and configuration."
        
    return "GENERIC", f"AI analysis is currently **MOCKED**. Please configure the **Gemini API** for this feature to work."


def clean_value(value):
    """Cleans numeric values by handling SA (comma for decimal, space/dot for thousands) format."""
    if not isinstance(value, str):
        return None
        
    value = str(value).replace('\n', '').replace('\r', '') 
    value = re.sub(r'(\d)\s+(\d)', r'\1\2', value) 
    
    value = value.replace(' ', '').replace('.', '') 
    value = value.replace(',', '.') 
    
    value = value.replace('Cr', '').replace('Dr', '-').strip() 
    
    value = re.sub(r'[^\d\.\-]+', '', value) 
        
    try:
        if re.match(r'^-?\d*\.?\d+$', value):
            return float(value)
        return None
    except:
        return None

def clean_description_for_xero(description):
    """Cleans up transaction descriptions for easy Xero reconciliation."""
    if not isinstance(description, str):
        return ""
        
    description = description.strip()
    
    description = re.sub(r'\s*\d{6}\s+\d{4}\s+\d{2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', '', description, flags=re.IGNORECASE)
    description = re.sub(r'(?:Ref\s*|Reference\s*|No\s*|Nr\s*|ID\s*):\s*[\w\d\-]+', '', description, flags=re.IGNORECASE)
    description = re.sub(r'Serial:\d+/\d+', '', description)

    description = re.sub(r'(?:POS Purchase|ATM Withdrawal|Immediate Payment|Internet Pmt To|Teller Transfer Debit|Direct Credit|EFT|IB Payment)\s*', '', description, flags=re.IGNORECASE)
    
    description = re.sub(r'\s{2,}', ' ', description).strip(' -').strip()
    
    return description

# --- 3. STANDARDIZATION FUNCTIONS ---

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
    df['Debits'] = df['Debits'].apply(clean_value)
    df['Credits'] = df['Credits'].apply(clean_value)
    
    df['Amount'] = df['Credits'].fillna(0) - df['Debits'].fillna(0)
    
    df['Date'] = df['Date'].astype(str)
    df['Description'] = df['Details'].astype(str)
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


# --- 4. CORE EXTRACTION LOGIC ---

def parse_pdf_data(pdf_file_path, file_name):
    """Core function to extract tables, detect bank, and standardize the data."""
    all_transactions = pd.DataFrame()
    bank_name = "GENERIC"

    try:
        with pdfplumber.open(pdf_file_path) as pdf:
            
            # 1. Detect Bank 
            full_text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
            bank_name = detect_bank_format(full_text)
            
            if bank_name == "GENERIC":
                    st.warning("⚠️ Could not reliably identify the bank statement format. Attempting AI analysis...")
                    
                    # --- AI FALLBACK INTEGRATION POINT ---
                    _ , ai_message = ai_analyze_statement(full_text, file_name)
                    st.markdown(f"**AI Report for Developer:** {ai_message}")
                    # We stick with "GENERIC" for parsing, but provide the AI suggestion
                    
                    st.info("Using **Generic Fallback** rules for table extraction.")
            else:
                st.info(f"✅ Detected Bank: **{bank_name}**. Applying custom parsing rules.")
            
            rules = BANK_RULES_MAP[bank_name]
            
            # 2. Extract Data Page by Page
            for page in pdf.pages:
                # Use page.extract_tables with the specific settings
                tables = page.extract_tables(rules["table_settings"])
                
                for table in tables:
                    if table and len(table) > 1:
                        # Skip header rows
                        start_index = 0
                        for i, row in enumerate(table):
                            if any("BALANCE" in str(cell).upper() for cell in row if cell):
                                start_index = i + 1
                                break
                        data_rows = table[start_index:]
                        
                        if not data_rows: continue
                        
                        try:
                            # Use detected column names
                            if len(data_rows[0]) == len(rules["columns"]):
                                df = pd.DataFrame(data_rows, columns=rules["columns"])
                            else:
                                df = pd.DataFrame(data_rows)
                        except:
                            df = pd.DataFrame(data_rows)

                        all_transactions = pd.concat([all_transactions, df], ignore_index=True)
                            
            # 3. Standardization & Final Formatting
            if not all_transactions.empty:
                all_transactions.dropna(thresh=2, inplace=True)
                
                # Apply the correct standardization function
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
                    # GENERIC FALLBACK
                    return generic_fallback(all_transactions.copy()), bank_name
                
                # Final filtering and column selection for the standardized result
                df_final.dropna(subset=['Amount'], inplace=True)
                
                return df_final, bank_name

    except Exception as e:
        st.error(f"An error occurred during PDF processing for {file_name}. Error: {e}")
        st.info("This often happens when `pdfplumber` cannot find a text layer (i.e., it's a scanned image) or the PDF is malformed.")
        return pd.DataFrame(), bank_name
    
    return pd.DataFrame(), bank_name

# --- 5. STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(page_title="🇿🇦 Free SA Bank Statement to Xero CSV Converter", layout="wide")

    st.title("🇿🇦 SA Bank Statement PDF to Xero CSV Converter")
    st.markdown("""
        ### Built by Gemini AI for accountants: Free, easy-to-use tool for South African bank statement conversion.
        
        This app includes **AI integration** (requires your Gemini API key setup) to help identify and provide developer hints for unknown bank statements!
        
        **Supported Banks (with custom rules): ABSA, FNB, Standard Bank, HBZ.**
        ---
    """)
    
    # Check if AI client is initialized and inform the user
    if client:
        st.sidebar.success("Gemini AI Analysis: **Active** ✅")
    else:
        st.sidebar.warning("Gemini AI Analysis: **Inactive** 🛑. Please set **GEMINI_API_KEY** in Streamlit Secrets.")


    # --- CRITICAL FIX: Added a unique 'key' argument to fix the StreamlitDuplicateElementId error ---
    uploaded_files = st.file_uploader(
        "Upload your bank statement PDF files (Multiple files supported)",
        type=["pdf"],
        accept_multiple_files=True,
        key="unique_pdf_uploader" 
    )

    if uploaded_files:
        st.subheader("Processing Files...")
        
        all_df = []
        
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            st.markdown(f"**Processing:** `{file_name}`")
            
            pdf_data = BytesIO(uploaded_file.read())
            
            # Pass file_name to the parsing function
            df_transactions, bank_name = parse_pdf_data(pdf_data, file_name)

            if not df_transactions.empty and 'Amount' in df_transactions.columns:
                
                df_transactions['Description'] = df_transactions['Description'].apply(clean_description_for_xero)
                
                df_final = df_transactions.rename(columns={
                    'Date': 'Date',
                    'Description': 'Description',
                    'Amount': 'Amount'
                })
                
                try:
                    df_final['Date'] = pd.to_datetime(df_final['Date'], errors='coerce', dayfirst=True).dt.strftime('%d/%m/%Y')
                except:
                    st.warning(f"Could not standardize dates for {file_name}.")
                
                # Final Xero structure: Date, Amount, Payee, Description, Reference
                df_xero = pd.DataFrame({
                    'Date': df_final['Date'].fillna(''),
                    'Amount': df_final['Amount'].round(2), 
                    'Payee': '', 
                    'Description': df_final['Description'].astype(str),
                    'Reference': file_name.split('.')[0] 
                })
                
                df_xero.dropna(subset=['Date', 'Amount'], inplace=True)
                
                all_df.append(df_xero)
                
                st.success(f"Successfully extracted {len(df_xero)} transactions from {file_name}")

        
        # --- 6. COMBINE AND DOWNLOAD ---

        if all_df:
            final_combined_df = pd.concat(all_df, ignore_index=True)
            
            st.markdown("---")
            st.subheader("✅ All Transactions Combined and Ready for Download")
            
            st.dataframe(final_combined_df)
            
            csv_output = final_combined_df.to_csv(index=False, sep=',', encoding='utf-8')
            st.download_button(
                label="⬇️ Download Xero Ready CSV File",
                data=csv_output,
                file_name="SA_Bank_Statements_Xero_Export.csv",
                mime="text/csv"
            )

if __name__ == '__main__':
    main()
    main()
