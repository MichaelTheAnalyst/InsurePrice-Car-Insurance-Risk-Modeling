# -*- coding: utf-8 -*-
"""
Generate high-quality PNG image from LinkedIn infographic HTML using Playwright
"""
import os
import sys

def generate_with_playwright():
    """Use Playwright for better full-page capture"""
    from playwright.sync_api import sync_playwright
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_file = os.path.join(current_dir, "linkedin_post_2_catboost_infographic.html")
    output_file = os.path.join(current_dir, "linkedin_post_2_catboost_infographic.png")
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1080, 'height': 1400})
        
        # Load the HTML file
        page.goto(f'file:///{html_file.replace(os.sep, "/")}')
        page.wait_for_timeout(2000)  # Wait for fonts to load
        
        # Get the infographic element and screenshot it
        infographic = page.locator('.infographic')
        infographic.screenshot(path=output_file, scale='device')
        
        browser.close()
    
    print(f"SUCCESS! Infographic saved to: {output_file}")

def generate_with_html2image():
    """Fallback to html2image"""
    from html2image import Html2Image
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_file = os.path.join(current_dir, "linkedin_post_2_catboost_infographic.html")
    
    hti = Html2Image(
        output_path=current_dir,
        size=(1080, 1400),
        custom_flags=['--default-background-color=0a0a0a', '--hide-scrollbars']
    )
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    hti.screenshot(html_str=html_content, save_as='linkedin_post_2_catboost_infographic.png')
    print("SUCCESS! Infographic saved.")

if __name__ == "__main__":
    try:
        generate_with_playwright()
    except ImportError:
        print("Playwright not found, installing...")
        os.system(f"{sys.executable} -m pip install playwright")
        os.system(f"{sys.executable} -m playwright install chromium")
        print("Installed! Run script again.")
    except Exception as e:
        print(f"Playwright error: {e}")
        print("Trying html2image fallback...")
        try:
            generate_with_html2image()
        except Exception as e2:
            print(f"Error: {e2}")
