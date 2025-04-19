import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="GA4 - Streamlit Button Tracking", layout="centered")

# Inject GA tracking + button click handling
ga_script = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-9S5SM84Q3T"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());

// Wait until page fully loads
window.addEventListener('load', function() {
    // Config first
    gtag('config', 'G-9S5SM84Q3T', { 'send_page_view': false });

    // Fire page_view manually
    gtag('event', 'page_view', {
        page_title: document.title,
        page_path: window.location.pathname
    });

    // Fire custom app_loaded event
    gtag('event', 'app_loaded', {
        app_name: 'Streamlit GA Test',
        load_time: new Date().toISOString()
    });

    console.log('✅ Streamlit app: Manual page_view + app_loaded events sent.');
});

// Function to track button clicks
function trackButtonClick(buttonName) {
    gtag('event', 'button_click', {
        button_name: buttonName
    });
    console.log('✅ GA4: button_click event sent for:', buttonName);
}
</script>
"""

# Inject GA Script
components.html(ga_script, height=0, width=0)

# Main Streamlit App Content
st.title("GA4 Test - Streamlit Button Click Tracking")
st.write("This page sends 'page_view', 'app_loaded', and tracks button clicks!")

# Create a button
if st.button("Get Quote"):
    # Inject JavaScript to track click
    components.html("""
    <script>
    trackButtonClick('Get Quote');
    </script>
    """, height=0, width=0)

    # Also do normal Streamlit action
    st.success("Button clicked! 🎯 Event should be visible in GA4 soon.")
