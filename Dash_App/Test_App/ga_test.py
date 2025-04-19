import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="GA Test", layout="centered")

# Google Analytics with enhanced configuration
ga_script = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-9S5SM84Q3T"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  
  // Enhanced configuration
  gtag('config', 'G-9S5SM84Q3T', {
    'page_title': 'GA Test',
    'page_path': window.location.pathname,
    'send_page_view': true
  });
</script>
"""

# Inject with height=0 and key to avoid duplicate injection
components.html(ga_script, height=0, width=0)

st.title("GA Test on Render")
st.write("Check Google Analytics Realtime → You should see 1 active user.")

# Debug information
st.write("""
## Debugging Tips:
1. Open your browser's developer tools (F12) and check the Network tab for calls to www.googletagmanager.com
2. Disable ad blockers for this page
3. Check the GA4 Realtime report in a separate tab
4. Try accessing your app in incognito mode
""")