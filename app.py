import streamlit as st
import pandas as pd
import pdfplumber
import re
from io import BytesIO

# --- 1. CONFIGURATION AND BANK SPECIFIC RULES ---

# Strong list-based rules for detection (used by detect_bank_format)
# Keywords are converted to lowercase for robust, case-insensitive matching.
BANK_RULES_LIST = [
    # FNB (Very strong identifier: fnb.co.za or the full bank name)
    ("FNB", ["fnb.co.za"]),
    ("FNB", ["first national bank", "account"]),
    
    # ABSA (Investment Account is a strong, unique indicator found in the files)
    ("ABSA", ["absa", "investment account"]),
    ("ABSA", ["absa", "bank limited", "account"]),
    
    # Standard Bank (Covers Private Banking and Business Current)
    ("STANDARD", ["standard bank", "private banking"]),
    ("STANDARD", ["standard bank", "current account"]),
    
    # HBZ (Unique bank name)
    ("HBZ", ["hbz bank limited", "account"]),
    
    # Nedbank and Capitec (General identifiers for future proofing)
    ("NEDBANK", ["nedbank limited", "account"]),
    ("CAPITEC", ["capitec bank", "account"]),
]

# Map for parsing logic once the bank is detected (used by parse_pdf_data)
BANK_RULES_MAP = {
    # FNB: Single 'Amount' column, +/- sign implied by 'Cr' or 'Dr' in Balance.
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
    # Standard Bank: Debits and Credits are separate columns.
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
    # ABSA: Single 'transaction amount R' column.
    "ABSA": {
        "columns": ["date", "transaction amount R", "description", "balance R"],
        "table_settings": {
            "vertical_strategy": "text",
            "explicit_vertical_lines": [30, 80, 400, 550],
            "snap_y_tolerance": 5
        },
        "standardize_func": "standardize_absa"
    },
    # HBZ Bank: Debit/Credit columns separate.
    "HBZ": {
        "columns": ["Date", "Particulars", "Debit", "Credit", "Reference"],
        "table_settings": {
            "vertical_strategy": "lines",
            "snap_y_tolerance": 3
        },
        "standardize_func": "standardize_hbz"
    },
    # Default/Generic fallback rule (for Nedbank/Capitec/Unknown)
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
    Returns the bank name (e.g., "ABSA", "FNB") or "GENERIC" if no bank is identified.
    """
    # Convert text to lowercase and normalize multiple spaces/newlines to single space
    lower_text = re.sub(r'\s+', ' ', text_content.lower())

    for bank, keywords in BANK_RULES_LIST:
        # Check if ALL required keywords are present in the text
        if all(keyword in lower_text for keyword in keywords):
            return bank
            
    return "GENERIC"

def clean_value(value):
    """Cleans numeric values by handling SA (comma for decimal, space/dot for thousands) format."""
    if not isinstance(value, str):
        return None
        
    # Standardize and clean up broken numbers (e.g., '2\n\n500,00' -> '2500,00')
    value = str(value).replace('\n', '').replace('\r', '') 
    value = re.sub(r'(\d)\s+(\d)', r'\1\2', value) 
    
    # Remove thousand separators (. or space)
    value = value.replace(' ', '').replace('.', '') 
    
    # Convert comma decimal to dot decimal (must be done after removing thousand dots)
    value = value.replace(',', '.') 
    
    # Handle FNB's Cr/Dr and general cleaning
    value = value.replace('Cr', '').replace('Dr', '-').strip() 
    
    # Final cleanup to keep only allowed characters
    value = re.sub(r'[^\d\.\-]+', '', value) 
        
    try:
        # Check if it's a valid number format before converting
        if re.match(r'^-?\d*\.?\d+$', value):
            return float(value)
        return None
    except:
        return None

def clean_description_for_xero(description):
    """
    Cleans up transaction descriptions for easy Xero reconciliation by removing noise.
    """
    if not isinstance(description, str):
        return ""
        
    description = description.strip()
    
    # Remove common bank reference/code noise
    description = re.sub(r'\s*\d{6}\s+\d{4}\s+\d{2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', '', description, flags=re.IGNORECASE)
    description = re.sub(r'(?:Ref\s*|Reference\s*|No\s*|Nr\s*|ID\s*):\s*[\w\d\-]+', '', description, flags=re.IGNORECASE)
    description = re.sub(r'Serial:\d+/\d+', '', description)

    # Remove transaction type prefixes
    description = re.sub(r'(?:POS Purchase|ATM Withdrawal|Immediate Payment|Internet Pmt To|Teller Transfer Debit|Direct Credit|EFT|IB Payment)\s*', '', description, flags=re.IGNORECASE)
    
    # Remove self-references and common suffixes (like the user's business name)
    description = re.sub(r'\s*-\s*Royal Panelbeaters', '', description, flags=re.IGNORECASE) 
    
    description = re.sub(r'\s{2,}', ' ', description).strip(' -').strip()
    
    return description

# --- 3. STANDARDIZATION FUNCTIONS ---

def standardize_fnb(df):
    """Specific standardization for FNB structure."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    def calculate_fnb_amount(row):
        # Extract and clean the numeric amount
        amount = clean_value(row.get('amount'))
        if amount is None: return None
        
        # Check for debit indicator in the raw amount column or balance column (FNB's convention)
        is_debit = ('dr' in str(row.get('balance', '')).lower() or 
                    '-' in str(row.get('amount', '')))

        return -abs(amount) if is_debit else abs(amount)

    df['Amount'] = df.apply(calculate_fnb_amount, axis=1)
    df['Date'] = df['date'].astype(str)
    df['Description'] = df['description'].astype(str)
    return df[['Date', 'Description', 'Amount']]

def standardize_standard(df):
    """Specific standardization for Standard Bank (Debit/Credit columns separate)."""
    df['Debits'] = df['Debits'].apply(clean_value)
    df['Credits'] = df['Credits'].apply(clean_value)
    
    # Calculate amount: Credit - Debit (Debits should be negative in Xero)
    df['Amount'] = df['Credits'].fillna(0) - df['Debits'].fillna(0)
    
    df['Date'] = df['Date'].astype(str)
    df['Description'] = df['Details'].astype(str) # 'Details' is the description column
    return df[['Date', 'Description', 'Amount']]

def standardize_absa(df):
    """Specific standardization for ABSA structure."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    # ABSA uses a single 'transaction amount R' column with a negative sign for debits
    df['Amount'] = df['transaction_amount_r'].apply(clean_value)
    df['Date'] = df['date'].astype(str)
    df['Description'] = df['description'].astype(str)
    return df[['Date', 'Description', 'Amount']]
    
def standardize_hbz(df):
    """Specific standardization for HBZ Bank (Debit/Credit columns separate)."""
    df['Debit'] = df['Debit'].apply(clean_value)
    df['Credit'] = df['Credit'].apply(clean_value)
    
    # Calculate amount: Credit - Debit (Debits should be negative in Xero)
    df['Amount'] = df['Credit'].fillna(0) - df['Debit'].fillna(0)
    
    df['Date'] = df['Date'].astype(str)
    df['Description'] = df['Particulars'].astype(str) # 'Particulars' is the description column
    return df[['Date', 'Description', 'Amount']]

def generic_fallback(df):
    """Fallback function for unknown banks."""
    st.warning("Could not apply specific standardization. CSV will contain raw extracted columns.")
    return df # Return raw dataframe for inspection

# --- 4. CORE EXTRACTION LOGIC ---

def parse_pdf_data(pdf_file_path):
    """Core function to extract tables, detect bank, and standardize the data."""
    all_transactions = pd.DataFrame()
    bank_name = "GENERIC"

    try:
        with pdfplumber.open(pdf_file_path) as pdf:
            # 1. Detect Bank (using the robust detection logic)
            full_text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
            bank_name = detect_bank_format(full_text)
            
            rules = BANK_RULES_MAP[bank_name]
            
            if bank_name == "GENERIC":
                 st.warning("⚠️ Could not reliably identify the bank statement format. Using generic parsing which may fail or produce incorrect results. Please check manually.")
            else:
                st.info(f"✅ Detected Bank: **{bank_name}**. Applying custom parsing rules.")
            
            # 2. Extract Data Page by Page
            for page in pdf.pages:
                tables = page.extract_tables(rules["table_settings"])
                
                for table in tables:
                    if table and len(table) > 1:
                        data_rows = []
                        
                        # Simple header detection logic to skip header rows
                        start_index = 0
                        for i, row in enumerate(table):
                            if any("BALANCE" in str(cell).upper() for cell in row if cell):
                                start_index = i + 1
                                break
                        data_rows = table[start_index:]
                        
                        if not data_rows: continue
                        
                        try:
                            # Apply column names only if the number of columns match
                            if len(data_rows[0]) == len(rules["columns"]):
                                df = pd.DataFrame(data_rows, columns=rules["columns"])
                            else:
                                # Fallback to no header if column count is wrong
                                df = pd.DataFrame(data_rows)
                        except:
                            df = pd.DataFrame(data_rows)

                        all_transactions = pd.concat([all_transactions, df], ignore_index=True)
                            
            # 3. Standardization & Final Formatting
            if not all_transactions.empty:
                # Remove rows with too few non-NaN values (often junk rows from PDF margins)
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
                else: # GENERIC FALLBACK
                    return generic_fallback(all_transactions.copy()), bank_name
                
                # Final filtering and column selection for the standardized result
                df_final.dropna(subset=['Amount'], inplace=True)
                
                return df_final, bank_name

    except Exception as e:
        st.error(f"An error occurred during PDF processing. Error: {e}")
        st.info("This often happens when `pdfplumber` cannot find a text layer (i.e., it's a scanned image) or the PDF is malformed.")
        return pd.DataFrame(), bank_name
    
    return pd.DataFrame(), bank_name

# --- 5. STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(page_title="🇿🇦 Free SA Bank Statement to Xero CSV Converter", layout="wide")

    st.title("🇿🇦 SA Bank Statement PDF to Xero CSV Converter")
    st.markdown("""
        ### Built by Gemini AI for accountants: Free, easy-to-use tool for South African bank statement conversion.
        This app uses **Python/pdfplumber** to extract transactions from **native (text-based) PDF** statements and format them for upload into **Xero**. 
        
        **Supported Banks (with custom rules): ABSA, FNB, Standard Bank, HBZ.**
        
        **⚠️ Important Note on Scanned Documents & Free Service:**
        * Extracting data from **scanned or image-based PDFs** requires advanced, proprietary **OCR/AI services** (which are not free).
        * This tool uses the best **free, open-source** method. It will attempt to process scanned files, but accuracy is **not guaranteed**.
        * It is best suited for **digital, native PDFs** downloaded directly from your bank.
        ---
    """)

    uploaded_files = st.file_uploader(
        "Upload your bank statement PDF files (Multiple files supported)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.subheader("Processing Files...")
        
        all_df = []
        
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            st.markdown(f"**Processing:** `{file_name}`")
            
            # Read PDF data into memory buffer
            pdf_data = BytesIO(uploaded_file.read())
            
            # Get the core transactions
            df_transactions, bank_name = parse_pdf_data(pdf_data)

            if not df_transactions.empty and 'Amount' in df_transactions.columns:
                
                # Apply Xero description cleanup
                df_transactions['Description'] = df_transactions['Description'].apply(clean_description_for_xero)
                
                # Select and rename columns for final Xero CSV format
                df_final = df_transactions.rename(columns={
                    'Date': 'Date',
                    'Description': 'Description',
                    'Amount': 'Amount'
                })
                
                # Standardize Date format to DD/MM/YYYY (Xero requirement)
                try:
                    df_final['Date'] = pd.to_datetime(df_final['Date'], errors='coerce', dayfirst=True).dt.strftime('%d/%m/%Y')
                except:
                    st.warning(f"Could not standardize dates for {file_name}. Dates remain in original format and may need manual check.")
                
                
                # Final Xero structure: Date, Amount, Payee, Description, Reference
                df_xero = pd.DataFrame({
                    'Date': df_final['Date'].fillna(''),
                    'Amount': df_final['Amount'].round(2), 
                    'Payee': '', # Leave blank for manual mapping in Xero
                    'Description': df_final['Description'].astype(str),
                    'Reference': file_name.split('.')[0] # Use file name as a default reference
                })
                
                # Remove any remaining rows where the core data (Date, Amount) is missing
                df_xero.dropna(subset=['Date', 'Amount'], inplace=True)
                
                all_df.append(df_xero)
                
                st.success(f"Successfully extracted {len(df_xero)} transactions from {file_name}")

        
        # --- 6. COMBINE AND DOWNLOAD ---

        if all_df:
            final_combined_df = pd.concat(all_df, ignore_index=True)
            
            st.markdown("---")
            st.subheader("✅ All Transactions Combined and Ready for Download")
            
            # Display the final, clean dataframe
            st.dataframe(final_combined_df)
            
            # Create the Xero CSV file
            csv_output = final_combined_df.to_csv(index=False, sep=',', encoding='utf-8')
            st.download_button(
                label="⬇️ Download Xero Ready CSV File",
                data=csv_output,
                file_name="SA_Bank_Statements_Xero_Export.csv",
                mime="text/csv"
            )

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
