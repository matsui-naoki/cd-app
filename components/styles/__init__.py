"""
Styles module for CD Analyzer
"""

import os


def get_custom_css() -> str:
    """Read custom CSS from file"""
    css_path = os.path.join(os.path.dirname(__file__), 'custom.css')
    try:
        with open(css_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""


def inject_custom_css(st):
    """Inject custom CSS into Streamlit app"""
    css = get_custom_css()
    if css:
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
