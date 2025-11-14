import os
import io
import json
from typing import List, Dict

import streamlit as st
import pandas as pd

# Optional imports for PDF parsing and export
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    canvas = None


# ------------------ GC PROFILES ------------------ #

GC_DIR = "gc_profiles"


def load_gc_profiles() -> Dict[str, dict]:
    profiles = {}
    if not os.path.isdir(GC_DIR):
        return profiles

    for fname in os.listdir(GC_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(GC_DIR, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            name = data.get("gc_name", os.path.splitext(fname)[0])
            profiles[name] = data
        except Exception:
            continue
    return profiles


# ------------------ PDF / DATA UTILITIES ------------------ #

def extract_text_from_pdf(file) -> str:
    """Extract raw text from a PDF file-like object."""
    if PyPDF2 is None:
        return "PyPDF2 not installed on server."

    reader = PyPDF2.PdfReader(file)
    pages_text: List[str] = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    return "\n".join(pages_text)


def naive_parse_line_items(raw_text: str, max_items: int = 150) -> pd.DataFrame:
    """
    Naive parser for demo:
    - Split lines
    - Filter to lines that look somewhat like scope items
    - Limit to max_items so we don't flood the UI
    """
    raw_lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    filtered = []
    for ln in raw_lines:
        if len(ln) < 5:
            continue  # too short, likely junk
        if ln.isupper():
            continue  # section headers / titles
        if any(ch.isdigit() for ch in ln):
            filtered.append(ln)
        elif len(filtered) < 30:
            filtered.append(ln)

    lines = filtered[:max_items]

    data = []
    for ln in lines:
        data.append(
            {
                "Include": True,
                "Item": ln,
                "Description": "",
                "Division": "General",
                "Quantity": 1.0,
                "Unit Cost": 0.0,
            }
        )

    if not data:
        data.append(
            {
                "Include": True,
                "Item": "Example Item",
                "Description": "Sample description",
                "Division": "General",
                "Quantity": 1.0,
                "Unit Cost": 100.0,
            }
        )

    return pd.DataFrame(data)


def apply_markup(df: pd.DataFrame, global_markup_amount: float) -> pd.DataFrame:
    """
    Apply a flat dollar markup to the total and allocate it proportionally
    across all included line items.
    """
    df = df.copy()

    # Filter to included rows only
    if "Include" in df.columns:
        df = df[df["Include"] == True]

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["Unit Cost"] = pd.to_numeric(df["Unit Cost"], errors="coerce").fillna(0.0)

    df["Base Total"] = df["Quantity"] * df["Unit Cost"]

    subtotal = df["Base Total"].sum()

    if subtotal <= 0 or global_markup_amount <= 0:
        # No markup or no base total – nothing fancy to do
        df["Markup $ Allocated"] = 0.0
        df["Total w/ Markup"] = df["Base Total"]
    else:
        # Allocate the flat markup across rows in proportion to their base total
        proportion = df["Base Total"] / subtotal
        df["Markup $ Allocated"] = proportion * global_markup_amount
        df["Total w/ Markup"] = df["Base Total"] + df["Markup $ Allocated"]

    return df


def export_to_excel(df: pd.DataFrame, gc_profile: dict | None) -> bytes:
    """Return Excel file bytes for download."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # GC info sheet
        meta_df = None
        if gc_profile:
            meta_df = pd.DataFrame(
                {
                    "Field": ["GC Name", "License", "Contact", "Phone", "Markup %"],
                    "Value": [
                        gc_profile.get("gc_name", ""),
                        gc_profile.get("license", ""),
                        gc_profile.get("contact", ""),
                        gc_profile.get("phone", ""),
                        gc_profile.get("markup_percent", ""),
                    ],
                }
            )
            meta_df.to_excel(writer, index=False, sheet_name="GC Info")

        df.to_excel(writer, index=False, sheet_name="Bid")

    output.seek(0)
    return output.read()


def export_to_pdf(df: pd.DataFrame, gc_profile: dict | None) -> bytes:
    """
    Very basic PDF export using reportlab.
    If reportlab is not installed, we just return empty bytes.
    """
    if canvas is None:
        return b""

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x = 40
    y = height - 40

    # GC Header
    c.setFont("Helvetica-Bold", 14)
    if gc_profile:
        c.drawString(x, y, gc_profile.get("gc_name", "ZGZO.AI Bid"))
    else:
        c.drawString(x, y, "ZGZO.AI Bid")
    y -= 18

    c.setFont("Helvetica", 9)
    if gc_profile:
        header_lines = [
            f"License: {gc_profile.get('license', '')}",
            f"Contact: {gc_profile.get('contact', '')}",
            f"Phone: {gc_profile.get('phone', '')}",
        ]
        for ln in header_lines:
            if ln.strip():
                c.drawString(x, y, ln)
                y -= 12

    y -= 10

    # Table headers
    c.setFont("Helvetica-Bold", 10)
    headers = ["Item", "Division", "Qty", "Unit Cost", "Total w/ Markup"]
    col_widths = [200, 80, 40, 80, 100]

    for i, h in enumerate(headers):
        c.drawString(x + sum(col_widths[:i]), y, h)
    y -= 15

    c.setFont("Helvetica", 9)

    for _, row in df.iterrows():
        if y < 60:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica-Bold", 10)
            for i, h in enumerate(headers):
                c.drawString(x + sum(col_widths[:i]), y, h)
            y -= 15
            c.setFont("Helvetica", 9)

        values = [
            str(row.get("Item", ""))[:70],
            str(row.get("Division", ""))[:20],
            f'{row.get("Quantity", 0):.2f}',
            f'{row.get("Unit Cost", 0):.2f}',
            f'{row.get("Total w/ Markup", 0):.2f}',
        ]
        for i, v in enumerate(values):
            c.drawString(x + sum(col_widths[:i]), y, v)
        y -= 12

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


# ------------------ STREAMLIT STATE ------------------ #

if "line_items" not in st.session_state:
    st.session_state.line_items = None

if "global_markup" not in st.session_state:
    st.session_state.global_markup = 10.0

if "selected_gc" not in st.session_state:
    st.session_state.selected_gc = None


# ------------------ APP LAYOUT ------------------ #

st.set_page_config(page_title="ZGZO.AI – Bid Generator Demo", layout="wide")

gc_profiles = load_gc_profiles()

st.title("ZGZO.AI – Bid Generator (Demo)")
st.caption("Upload plans → generate rough line items → edit → apply markup → export bid.")

st.markdown("---")

# GC selection row
gc_col, info_col = st.columns([1.3, 2.7])

with gc_col:
    gc_names = list(gc_profiles.keys())
    gc_selected_name = st.selectbox(
        "Select GC Profile",
        options=["(No profile)"] + gc_names,
        index=0 if not gc_names else 1,
    )

    if gc_selected_name != "(No profile)":
        st.session_state.selected_gc = gc_profiles.get(gc_selected_name)
    else:
        st.session_state.selected_gc = None

with info_col:
    gc = st.session_state.selected_gc
    if gc:
        st.markdown(f"**GC:** {gc.get('gc_name', '')}")
        st.markdown(f"**License:** {gc.get('license', '')}")
        contact_line = []
        if gc.get("contact"):
            contact_line.append(gc["contact"])
        if gc.get("phone"):
            contact_line.append(gc["phone"])
        if contact_line:
            st.markdown("**Contact:** " + " • ".join(contact_line))
        st.caption(gc.get("legal", ""))
    else:
        st.info("No GC profile selected. You can still use the demo without a profile.")

st.markdown("---")

left_col, right_col = st.columns([2, 3])

# Left: upload and generate
with left_col:
    st.subheader("1. Upload Plans (PDF)")
    uploaded = st.file_uploader("Upload a plans or spec PDF", type=["pdf"])

    if uploaded is not None:
        if st.button("Generate Rough Line Items"):
            with st.spinner("Extracting text and creating rough line items..."):
                text = extract_text_from_pdf(uploaded)
                df = naive_parse_line_items(text)
                st.session_state.line_items = df

# Right: line items table
with right_col:
    st.subheader("2. Review & Edit Line Items")

    if st.session_state.line_items is None:
        st.info("Upload a PDF and click 'Generate Rough Line Items' to begin.")
    else:
        edited_df = st.data_editor(
            st.session_state.line_items,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Include": st.column_config.CheckboxColumn(default=True),
                "Quantity": st.column_config.NumberColumn(step=1.0, min_value=0.0),
                "Unit Cost": st.column_config.NumberColumn(step=1.0, min_value=0.0),
            },
            key="line_items_editor",
        )
        st.session_state.line_items = edited_df

st.markdown("---")

# Markup + totals + export

st.subheader("3. Apply Markup & Export Bid")
if st.session_state.line_items is not None:
    sub_cols = st.columns([1, 3])
    with sub_cols[0]:
        # Flat dollar markup – no default from GC for now
        markup_amount = st.number_input(
            "Global Markup ($)",
            min_value=0.0,
            value=float(st.session_state.global_markup),
            step=500.0,
        )
        st.session_state.global_markup = markup_amount

    df_with_totals = apply_markup(
        st.session_state.line_items,
        st.session_state.global_markup,
    )

    with sub_cols[1]:
        subtotal = df_with_totals["Base Total"].sum()
        total_with_markup = df_with_totals["Total w/ Markup"].sum()
        total_markup_allocated = df_with_totals["Markup $ Allocated"].sum()

        st.metric("Base Total", f"${subtotal:,.2f}")
        st.metric("Markup (flat $)", f"${total_markup_allocated:,.2f}")
        st.metric("Total with Markup", f"${total_with_markup:,.2f}")


    st.markdown("### 4. Export Bid")

    excel_bytes = export_to_excel(df_with_totals, st.session_state.selected_gc)
    st.download_button(
        "⬇️ Download Excel Bid",
        data=excel_bytes,
        file_name="zgzo_bid.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    pdf_bytes = export_to_pdf(df_with_totals, st.session_state.selected_gc)
    if pdf_bytes:
        st.download_button(
            "⬇️ Download PDF Bid",
            data=pdf_bytes,
            file_name="zgzo_bid.pdf",
            mime="application/pdf",
        )
    else:
        st.caption("PDF export requires `reportlab` to be installed (pip install reportlab).")
else:
    st.info("Once you have line items, you'll be able to apply markup and export here.")
