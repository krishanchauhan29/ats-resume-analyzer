# 🎯 ATS Resume Analyzer

An AI-powered ATS Resume Analyzer that scores your resume against industry-standard criteria across 19 job roles with detailed feedback and downloadable reports.

## 🔴 Live Demo
👉 [Click here to try the live app](https://ats-resume-analyzer292003.streamlit.app/)

## ✨ Features
- 📄 **Multi-format Input** — PDF, Image (PNG/JPG), DOCX
- 🎯 **ATS Score (0-100)** — Industry-standard weighted scoring
- 🔍 **Keywords Analysis** — Role-specific keyword matching
- 📋 **Section Analysis** — Contact, Sections, Action Verbs, Quantification
- 💡 **Recommendations** — Personalized actionable feedback
- 📊 **Preview Reports** — View before downloading
- 📥 **Download Reports** — Excel (4 sheets) + PDF

## 🎯 Supported Job Roles (19)
Data Science, Data Analyst, ML Engineer, AI Engineer, Data Engineer, NLP Engineer, Computer Vision Engineer, Research Scientist, Product Analyst, Software Engineer, Full Stack Developer, DevOps Engineer, Cloud Architect, Cybersecurity Analyst, Blockchain Developer, Business Analyst, Finance Analyst, Marketing Analyst, HR Analyst

## 🏆 Scoring Criteria
| Category | Weight |
|----------|--------|
| Keywords Match | 45% |
| Quantification | 15% |
| Section Structure | 15% |
| Action Verbs | 7% |
| Contact Info | 13% |
| Formatting | 5% |

## 🛠️ Tech Stack
- **Python** — NLTK, PDFPlumber, Pytesseract
- **Streamlit** — Interactive dashboard
- **Plotly** — Score gauge + visualizations
- **ReportLab** — PDF report generation
- **XlsxWriter** — Excel report generation

## 🚀 Run Locally
```bash
git clone https://github.com/krishanchauhan29/ats-resume-analyzer.git
cd ats-resume-analyzer
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## 👤 Built By
**Krishan Kumar Chauhan**
M.Tech Data Science | Gautam Buddha University
[LinkedIn](https://www.linkedin.com/in/krishan-chauhan-714011232/) | [GitHub](https://github.com/krishanchauhan29)