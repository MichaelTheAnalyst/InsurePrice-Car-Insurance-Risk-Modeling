import os
import sys
import streamlit.web.cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "insureprice_dashboard.py"]
    sys.exit(stcli.main())
