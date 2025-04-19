import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="GA Test - App Loaded Event", layout="centered")

# Inject Google Analytics + Custom "app_loaded" Event
ga_script = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-9S5SM84Q3T"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  // Basic Configuration (disable automatic page_view)
  gtag('config', 'G-9S5SM84Q3T', { 'send_page_view': false });

  // Send manual page_view
  gtag('event', 'page_view', {
    page_title: 'GA Test - App Loaded Event',
    page_path: window.location.pathname
  });

  // 🔥 Send custom app_loaded event
  gtag('event', 'app_loaded', {
    app_name: 'Actuarial Angles App',
    load_time: new Date().toISOString()
  });
</script>
"""

# Inject the script invisibly
components.html(ga_script, height=0, width=0)

st.title("GA Test with Custom Event")
st.write("Custom 'app_loaded' event sent to Google Analytics. Check your Realtime report!")

# Helpful Debug Tips
st.write("""
## Debugging Tips:
1. Open Chrome Developer Tools → Network tab → Search for "collect" or "gtag".
2. Look inside GA4 Realtime → Events → You should see "app_loaded".
3. Disable ad blockers or test in Incognito mode.
4. Make sure you are using the Measurement ID (`G-G-9S5SM84Q3T`).
""")
