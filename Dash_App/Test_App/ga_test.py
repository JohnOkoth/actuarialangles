import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="GA4 - Streamlit gtag Fix", layout="centered")

# Inject proper GA gtag script
ga_proper_script = """
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-9S5SM84Q3T"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  // Setup config without sending automatic page_view
  gtag('config', 'G-9S5SM84Q3T', { 'send_page_view': false });

  // Manually send page_view AFTER page load
  window.addEventListener('load', function() {
    gtag('event', 'page_view', {
      page_title: document.title,
      page_path: window.location.pathname
    });
    console.log("✅ Page View event manually fired.");
  });

  // Function to manually fire button click events
  function sendButtonClickEvent(buttonName) {
    gtag('event', 'button_click', {
      button_name: buttonName
    });
    console.log("✅ Button Click event fired for:", buttonName);
  }
</script>
"""

# Inject HTML invisibly
components.html(ga_proper_script, height=0, width=0)

# Streamlit App Content
st.title("GA4 Manual gtag Tracking - Streamlit")
st.write("Manual 'page_view' and 'button_click' events fired using proper gtag setup.")

# Button with event tracking
if st.button("Get Quote"):
    components.html("""
    <script>
    sendButtonClickEvent('Get Quote');
    </script>
    """, height=0, width=0)
    st.success("Button clicked! 'button_click' event manually sent with gtag.js.")
