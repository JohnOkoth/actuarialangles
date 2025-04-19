import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="GA Test", layout="centered")

# Updated Google Analytics Injection
ga_script = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-9S5SM84Q3T"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  
  gtag('config', 'G-9S5SM84Q3T', {
    'send_page_view': false
  });

  // Manually send page_view immediately
  gtag('event', 'page_view', {
    page_title: 'GA Test',
    page_path: window.location.pathname
  });
</script>
"""

components.html(ga_script, height=0, width=0)

st.title("GA Test on Render")
st.write("Check Google Analytics Realtime → You should now see 1 active user (forced page_view).")
