# -*- coding: utf-8 -*-
import base64

# Read the profile picture
with open('MN pic.jpg', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode()

# Read the HTML file
with open('linkedin_post_2_catboost_infographic.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Replace img src with base64 embedded image
html = html.replace('src="MN pic.jpg"', f'src="data:image/jpeg;base64,{img_data}"')

# Fix the height - reduce from 1350 to 1180 for tighter fit
html = html.replace('height: 1350px;', 'height: 1180px;')

# Write the updated HTML
with open('linkedin_post_2_catboost_infographic.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('HTML updated: embedded profile picture + fixed height')

