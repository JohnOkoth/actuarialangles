iimport streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="GA Test", layout="centered")

# Google Analytics script
components.html("""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-9S5SM84Q3T"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', 'G-9S5SM84Q3T');
</script>
""", height=0)

st.title("GA Test on Render")
st.write("Check Google Analytics Realtime → You should see 1 active user.")
