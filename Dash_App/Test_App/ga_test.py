import streamlit as st
import streamlit.components.v1 as components

import random
import string

st.set_page_config(page_title="GA4 Manual Tracking via Fetch", layout="centered")

# Generate a random Client ID for tracking
def generate_cid():
    return ''.join(random.choices(string.digits, k=10))

cid = generate_cid()  # Generate once for the user session

# Inject manual fetch-based GA4 tracking
ga_manual_script = f"""
<script>
// Your Measurement ID
const MEASUREMENT_ID = "G-9S5SM84Q3T";
// Random Client ID
const CLIENT_ID = "{cid}";

// Function to manually send event to GA4
function sendGAEvent(eventName, additionalParams = {{}}, debug=false) {{
    let params = {{
        v: '2',                 // API Version
        tid: MEASUREMENT_ID,    // Tracking ID
        cid: CLIENT_ID,         // Client ID (random for now)
        en: eventName,          // Event name
        dl: window.location.href,  // Document location
        dt: document.title         // Document title
    }};
    
    // Merge additional parameters
    for (const key in additionalParams) {{
        params[key] = additionalParams[key];
    }}

    fetch('https://www.google-analytics.com/g/collect', {{
        method: 'POST',
        body: new URLSearchParams(params)
    }})
    .then(response => {{
        if (debug) {{
            console.log('GA Event Sent:', eventName, params, 'Status:', response.status);
        }}
    }})
    .catch(error => {{
        console.error('Error sending GA event:', error);
    }});
}}

// Send page_view event immediately
window.addEventListener('load', function() {{
    sendGAEvent('page_view', {{}}, true);
}});
</script>
"""

# Inject Script
components.html(ga_manual_script, height=0, width=0)

# Main App
st.title("GA4 Manual Tracking (Fetch Method)")
st.write("This app sends 'page_view' and 'button_click' events manually using fetch.")

# Button to send custom click event
if st.button("Get Quote"):
    components.html("""
    <script>
    sendGAEvent('button_click', {button_name: 'Get Quote'}, true);
    </script>
    """, height=0, width=0)
    st.success("Button clicked! 'button_click' event manually sent via fetch.")

