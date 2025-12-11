# -Inventory-Sales-Prediction-and-Management
"A Flask-based Walmart retail analytics system with ML-powered weekly sales forecasting, interactive store dashboards, and full inventory management. Includes automated preprocessing, secure user authentication, low-stock alerts, and responsive Plotly visualizations."


---

## Features

- User signup / login (SQLite `database/users.db`)
- Dashboard and analytics pages (`/dashboard`, `/status`, `/top-store-metrics`)
- Forecast API: `POST /predict` — returns total & daily average sales and detailed forecast
- Pre-built templates: `login.html`, `index.html`, `analytics.html`, `top.html`
- Simple session-based auth (Flask sessions)
- Plot generation utilities for analytics (Plotly functions used in code)

---
