import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="GA Test", page_icon="📊")

components.html("""
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-9S5SM84Q3T"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', 'G-9S5SM84Q3T');
</script>
""", height=0)

st.title("Google Analytics Test")
st.write("Visit your GA dashboard and check Realtime.")
