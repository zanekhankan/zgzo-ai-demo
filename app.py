import os
import io
import json
from typing import List, Dict

import streamlit as st
import pandas as pd

# PDF + export libs
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    canvas = None

# AI (Groq)
from groq import Groq


# ------------- GC PROFILES ------------- #

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


# ------------- PDF / PARSING ------------- #

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
    Naive parser:
    - Split lines
    - Filter obvious junk
    - Limit to max_items
    """
    raw_lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    filtered = []
    for ln in raw_lines:
        if len(ln) < 5:
            continue
        if ln.isupper():
            continue
        if any(ch.isdigit() for ch in ln):
            filtered.append(ln)
        elif len(filtered) < 30:
            filtered.append(ln)

    lines = filtered[:max_items]

    rows = []
    for ln in lines:
        rows.append(
            {
                "Include": True,
                "Item": ln,
                "Description": "",
                "Division": "General",
                "Unit": "LS",
                "Quantity": 1.0,
                "Unit Cost": 0.0,
            }
        )

    if not rows:
        rows.append(
            {
                "Include": True,
                "Item": "Example Item",
                "Description": "Sample scope",
                "Division": "General",
                "Unit": "LS",
                "Quantity": 1.0,
                "Unit Cost": 100.0,
            }
        )

    return pd.DataFrame(rows)


# ------------- BRAIN v1 (Groq AI) ------------- #

def ai_clean_line_items(raw_text: str, max_items: int = 40) -> pd.DataFrame:
    """
    Use Groq (Llama 3) to turn messy spec text into a clean list of bid items.
    Returns a DataFrame with columns:
    Include, Item, Division, Unit, Quantity, Unit Cost
    """
    if not raw_text or len(raw_text.strip()) < 20:
        return naive_parse_line_items(raw_text, max_items=max_items)

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    system_prompt = (
        "You are an estimator assistant for a general contractor. "
        "Your job is to read construction specifications and output ONLY a structured list "
        "of actionable bid line items.\n\n"
        "Rules:\n"
        "- Focus on scope items the GC would actually price (demo, concrete, framing, finishes, etc.).\n"
        "- Ignore admin text, code references, boilerplate, legal text, and instructions to bidders.\n"
        "- Group similar scope into practical bid items.\n"
        "- Use concise, contractor-friendly item names.\n"
        "- Use simple CSI-style divisions (e.g. '01 General', '02 Demolition', '03 Concrete', "
        "'06 Carpentry', '08 Openings', '09 Finishes', '21/22/23 MEP').\n"
        "- If quantity is not clearly known, set it to null.\n"
        "- Unit can be LS, SF, LF, EA, CY, etc.\n"
        "- Do NOT invent crazy detail or fake quantities.\n\n"
        "Output ONLY valid JSON. No commentary, no markdown, no extra text.\n"
        "JSON must be a list of objects with keys: item, division, unit, quantity."
    )

    user_prompt = f"""
You are given construction spec text. Extract up to {max_items} practical bid items.

Spec text:
\"\"\"
{raw_text[:12000]}
\"\"\"

Return JSON like:
[
  {{"item": "Mobilization & site setup", "division": "01 General", "unit": "LS", "quantity": 1}},
  {{"item": "Remove existing carpet flooring", "division": "02 Demolition", "unit": "SF", "quantity": 1500}}
]
"""

    try:
        resp = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        content = resp.choices[0].message.content.strip()
        items = json.loads(content)
    except Exception:
        return naive_parse_line_items(raw_text, max_items=max_items)

    rows = []
    for obj in items:
        item = str(obj.get("item", "")).strip()
        if not item:
            continue

        division = str(obj.get("division", "General")).strip()
        unit = str(obj.get("unit", "LS")).strip()
        qty = obj.get("quantity", None)

        try:
            qty_val = float(qty) if qty is not None else 1.0
        except Exception:
            qty_val = 1.0

        rows.append(
            {
                "Include": True,
                "Item": item,
                "Description": "",
                "Division": division,
                "Unit": unit,
                "Quantity": qty_val,
                "Unit Cost": 0.0,
            }
        )

    if not rows:
        return naive_parse_line_items(raw_text, max_items=max_items)

    return pd.DataFrame(rows)


# ------------- MARKUP & EXPORT ------------- #

def apply_markup(df: pd.DataFrame, markup_amount: float) -> pd.DataFrame:
    """
    Apply a flat dollar markup to the total and allocate it
    proportionally across all included line items.
    """
    df = df.copy()

    if "Include" in df.columns:
        df = df[df["Include"] == True]

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["Unit Cost"] = pd.to_numeric(df["Unit Cost"], errors="coerce").fillna(0.0)

    df["Base Total"] = df["Quantity"] * df["Unit Cost"]
    subtotal = df["Base Total"].sum()

    if subtotal <= 0 or markup_amount <= 0:
        df["Markup $ Allocated"] = 0.0
        df["Total w/ Markup"] = df["Base Total"]
    else:
        proportion = df["Base Total"] / subtotal
        df["Markup $ Allocated"] = proportion * markup_amount
        df["Total w/ Markup"] = df["Base Total"] + df["Markup $ Allocated"]

    return df


def export_to_excel(df: pd.DataFrame, gc_profile: dict | None) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        if gc_profile:
            meta_df = pd.DataFrame(
                {
                    "Field": ["GC Name", "License", "Contact", "Phone"],
                    "Value": [
                        gc_profile.get("gc_name", ""),
                        gc_profile.get("license", ""),
                        gc_profile.get("contact", ""),
                        gc_profile.get("phone", ""),
                    ],
                }
            )
            meta_df.to_excel(writer, index=False, sheet_name="GC Info")

        df.to_excel(writer, index=False, sheet_name="Bid")

    output.seek(0)
    return output.read()


def export_to_pdf(df: pd.DataFrame, gc_profile: dict | None) -> bytes:
    if canvas is None:
        return b""

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x = 40
    y = height - 40

    c.setFont("Helvetica-Bold", 14)
    if gc_profile:
        c.drawString(x, y, gc_profile.get("gc_name", "ZGZO.AI Bid"))
    else:
        c.drawString(x, y, "ZGZO.AI Bid")
    y -= 18

    c.setFont("Helvetica", 9)
    if gc_profile:
        lines = [
            f"License: {gc_profile.get('license', '')}",
            f"Contact: {gc_profile.get('contact', '')}",
            f"Phone: {gc_profile.get('phone', '')}",
        ]
        for ln in lines:
            if ln.strip():
                c.drawString(x, y, ln)
                y -= 12

    y -= 10

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
            f"{row.get('Quantity', 0):.2f}",
            f"${row.get('Unit Cost', 0):.2f}",
            f"${row.get('Total w/ Markup', 0):.2f}",
        ]
        for i, v in enumerate(values):
            c.drawString(x + sum(col_widths[:i]), y, v)
        y -= 12

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


# ------------- STREAMLIT STATE ------------- #

if "line_items" not in st.session_state:
    st.session_state.line_items = None

if "global_markup" not in st.session_state:
    st.session_state.global_markup = 0.0  # flat $

if "selected_gc" not in st.session_state:
    st.session_state.selected_gc = None

if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""


# ------------- APP LAYOUT ------------- #

st.set_page_config(page_title="ZGZO.AI – Bid Generator (Demo)", layout="wide")

gc_profiles = load_gc_profiles()

st.title("ZGZO.AI – Bid Generator (Demo)")
st.caption("Upload specs → AI organizes line items → you edit → apply flat markup → export bid.")

st.markdown("---")

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
        contact_parts = []
        if gc.get("contact"):
            contact_parts.append(gc["contact"])
        if gc.get("phone"):
            contact_parts.append(gc["phone"])
        if contact_parts:
            st.markdown("**Contact:** " + " • ".join(contact_parts))
        st.caption(gc.get("legal", ""))
    else:
        st.info("No GC profile selected. You can still use the demo without a profile.")

st.markdown("---")

left_col, right_col = st.columns([2, 3])

with left_col:
    st.subheader("1. Upload Specs (PDF)")
    uploaded = st.file_uploader("Upload a plans/spec PDF", type=["pdf"])

    if uploaded is not None:
        if st.button("Generate Rough Line Items"):
            with st.spinner("Extracting text and creating rough line items..."):
                text = extract_text_from_pdf(uploaded)
                st.session_state.raw_text = text
                df = naive_parse_line_items(text)
                st.session_state.line_items = df

with right_col:
    st.subheader("2. Review & Edit Line Items")

    if st.session_state.line_items is None:
        st.info("Upload a PDF and click 'Generate Rough Line Items' to begin.")
    else:
        ai_col, _ = st.columns([1, 3])
        with ai_col:
            if st.button("✨ AI Clean & Organize Items"):
                with st.spinner("Letting ZGZO.AI clean and organize the spec..."):
                    df_ai = ai_clean_line_items(st.session_state.raw_text or "")
                    st.session_state.line_items = df_ai

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

st.subheader("3. Apply Markup & Export Bid")

if st.session_state.line_items is not None:
    sub_cols = st.columns([1, 3])
    with sub_cols[0]:
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
        total_markup_allocated = df_with_totals["Markup $ Allocated"].sum()
        total_with_markup = df_with_totals["Total w/ Markup"].sum()

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
