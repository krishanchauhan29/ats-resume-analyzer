import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pdfplumber
import pytesseract
from PIL import Image
from io import BytesIO
import re
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

st.set_page_config(page_title="ATS Resume Analyzer", page_icon="🎯", layout="wide")

# ==================== KEYWORDS DATABASE ====================
TECHNICAL_KEYWORDS = {
    'Data Science': ['machine learning', 'deep learning', 'data analysis', 'python', 'sql',
                     'statistics', 'visualization', 'pandas', 'numpy', 'scikit-learn',
                     'tensorflow', 'pytorch', 'nlp', 'computer vision', 'feature engineering',
                     'model deployment', 'a/b testing', 'hypothesis testing', 'regression',
                     'classification', 'clustering', 'neural network', 'random forest', 'tableau', 'power bi', 'excel', 'eda','streamlit', 'seaborn', 'matplotlib', 'data cleaning', 'kpi'],

    'Data Analyst': ['sql', 'excel', 'power bi', 'tableau', 'python', 'data visualization',
                     'dashboard', 'reporting', 'kpi', 'metrics', 'data cleaning', 'eda',
                     'pivot table', 'statistical analysis', 'business intelligence',
                     'data mining', 'etl', 'google analytics', 'pandas', 'numpy'],

    'ML Engineer': ['python', 'machine learning', 'deep learning', 'pytorch', 'tensorflow',
                    'mlops', 'docker', 'kubernetes', 'aws', 'fastapi', 'flask',
                    'model deployment', 'ci/cd', 'git', 'neural network', 'transformer',
                    'bert', 'llm', 'cuda', 'rest api', 'scikit-learn'],

    'AI Engineer': ['python', 'machine learning', 'llm', 'langchain', 'pytorch',
                    'tensorflow', 'transformer', 'bert', 'gpt', 'rag', 'openai',
                    'vector database', 'mlops', 'docker', 'fastapi', 'huggingface',
                    'prompt engineering', 'fine-tuning', 'cuda', 'embeddings'],

    'Data Engineer': ['python', 'sql', 'spark', 'hadoop', 'airflow', 'kafka', 'etl',
                      'data pipeline', 'aws', 'gcp', 'azure', 'docker', 'kubernetes',
                      'postgresql', 'mongodb', 'databricks', 'snowflake', 'dbt',
                      'data warehouse', 'data lake', 'ci/cd', 'pyspark'],

    'Research Scientist': ['python', 'pytorch', 'tensorflow', 'research', 'publications',
                           'deep learning', 'machine learning', 'statistics', 'mathematics',
                           'linear algebra', 'probability', 'nlp', 'computer vision',
                           'reinforcement learning', 'arxiv', 'experimentation', 'cuda'],

    'NLP Engineer': ['python', 'nlp', 'bert', 'gpt', 'transformer', 'huggingface',
                     'spacy', 'nltk', 'text classification', 'named entity recognition',
                     'sentiment analysis', 'language model', 'fine-tuning', 'pytorch',
                     'tokenization', 'word embeddings', 'rag', 'llm', 'langchain'],

    'Computer Vision Engineer': ['python', 'opencv', 'pytorch', 'tensorflow', 'cnn',
                                  'image classification', 'object detection', 'yolo',
                                  'image segmentation', 'deep learning', 'resnet',
                                  'data augmentation', 'grad-cam', 'transfer learning',
                                  'cuda', 'mediapipe', 'pillow'],

    'Product Analyst': ['sql', 'excel', 'product analytics', 'a/b testing', 'user research',
                        'google analytics', 'mixpanel', 'amplitude', 'funnel analysis',
                        'cohort analysis', 'retention', 'kpi', 'roadmap',
                        'data visualization', 'tableau', 'power bi', 'python'],

    'Software Engineer': ['python', 'java', 'javascript', 'c++', 'sql', 'git', 'docker',
                          'kubernetes', 'aws', 'react', 'node.js', 'rest api', 'microservices',
                          'agile', 'scrum', 'ci/cd', 'linux', 'mongodb', 'postgresql'],

    'Full Stack Developer': ['javascript', 'react', 'node.js', 'python', 'sql', 'html',
                              'css', 'mongodb', 'postgresql', 'rest api', 'git', 'docker',
                              'aws', 'typescript', 'next.js', 'express.js', 'graphql'],

    'DevOps Engineer': ['docker', 'kubernetes', 'aws', 'gcp', 'azure', 'ci/cd', 'linux',
                        'terraform', 'ansible', 'jenkins', 'git', 'python', 'bash',
                        'monitoring', 'prometheus', 'grafana', 'nginx', 'helm'],

    'Cloud Architect': ['aws', 'azure', 'gcp', 'terraform', 'kubernetes', 'docker',
                        'microservices', 'serverless', 'cloud security', 'networking',
                        'vpc', 'iam', 'lambda', 's3', 'rds', 'cloudformation'],

    'Cybersecurity Analyst': ['network security', 'penetration testing', 'firewall', 'siem',
                               'python', 'vulnerability assessment', 'incident response',
                               'ethical hacking', 'wireshark', 'owasp', 'iso 27001',
                               'risk assessment', 'cryptography', 'soc', 'compliance'],

    'Blockchain Developer': ['solidity', 'ethereum', 'web3', 'smart contracts', 'python',
                              'javascript', 'defi', 'nft', 'truffle', 'hardhat',
                              'ipfs', 'cryptography', 'react', 'node.js', 'polygon'],

    'Business Analyst': ['sql', 'excel', 'power bi', 'tableau', 'requirements gathering',
                         'stakeholder management', 'business intelligence', 'process improvement',
                         'data visualization', 'reporting', 'agile', 'scrum', 'jira',
                         'documentation', 'use case', 'gap analysis', 'kpi', 'brd'],

    'Finance Analyst': ['excel', 'financial modeling', 'sql', 'power bi', 'accounting',
                        'valuation', 'dcf', 'financial statements', 'forecasting',
                        'budgeting', 'variance analysis', 'bloomberg', 'vba',
                        'investment analysis', 'risk analysis', 'ifrs', 'gaap'],

    'Marketing Analyst': ['google analytics', 'excel', 'sql', 'tableau', 'seo',
                          'sem', 'social media analytics', 'crm', 'hubspot',
                          'a/b testing', 'email marketing', 'google ads',
                          'market research', 'customer segmentation', 'roi analysis',
                          'power bi', 'python', 'conversion optimization'],

    'HR Analyst': ['excel', 'hris', 'sql', 'power bi', 'recruitment analytics',
                   'workforce planning', 'compensation analysis', 'tableau',
                   'employee engagement', 'attrition analysis', 'performance management',
                   'hr metrics', 'python', 'data visualization', 'workday', 'sap hr'],

    'General': ['communication', 'teamwork', 'problem solving', 'leadership', 'analytical',
                'project management', 'critical thinking', 'attention to detail',
                'time management', 'collaboration', 'presentation', 'research',
                'adaptability', 'multitasking', 'stakeholder management']
}

ACTION_VERBS = [
    'achieved', 'built', 'created', 'designed', 'developed', 'implemented',
    'improved', 'increased', 'launched', 'led', 'managed', 'optimized',
    'reduced', 'delivered', 'deployed', 'automated', 'analyzed', 'generated',
    'established', 'enhanced', 'collaborated', 'coordinated', 'executed',
    'published', 'trained', 'mentored', 'researched', 'architected', 'engineered'
]

CONTACT_PATTERNS = {
    'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    'phone': r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
    'linkedin': r'linkedin\.com/in/[\w-]+',
    'github': r'github\.com/[\w-]+'
}

# ==================== TEXT EXTRACTION ====================
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if page_text:
                text += page_text + "\n"
    text = re.sub(r'(?<=[A-Z])\s(?=[A-Z])', '', text)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    # Fix merged words from LaTeX: "PowerBI" → "Power BI", "RandomForest" → "Random Forest"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Fix common merged keywords
    text = text.replace('PowerBI', 'Power BI')
    text = text.replace('ScikitLearn', 'Scikit-learn')
    text = text.replace('Scikit-learn', 'scikit-learn')
    text = text.replace('RandomForest', 'Random Forest')
    text = text.replace('FeatureEngineering', 'Feature Engineering')
    text = text.replace('DeepLearning', 'Deep Learning')
    text = text.replace('MachineLearning', 'Machine Learning')
    text = text.replace('DataScience', 'Data Science')
    text = text.replace('NeuralNetwork', 'Neural Network')
    text = text.lower()
    return text.strip()

def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text.strip()

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif name.endswith(('.png', '.jpg', '.jpeg')):
        return extract_text_from_image(uploaded_file)
    elif name.endswith('.docx'):
        from docx import Document
        doc = Document(uploaded_file)
        return '\n'.join([para.text for para in doc.paragraphs])
    return ""

# ==================== ATS SCORING ENGINE ====================
def analyze_contact_info(text):
    score = 0
    found = {}
    details = []
    
    patterns = {
        'email': [r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'],
        'phone': [r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'],
        'linkedin': [r'linkedin\.com/in/[\w-]+', r'\blinkedin\b'],
        'github': [r'github\.com/[\w-]+', r'\bgithub\b'],
    }
    
    for key, pattern_list in patterns.items():
        matched = False
        for pattern in pattern_list:
            if re.search(pattern, text, re.IGNORECASE):
                matched = True
                break
        if matched:
            score += 25
            found[key] = True
            details.append(f"✅ {key.capitalize()} found")
        else:
            found[key] = False
            details.append(f"❌ {key.capitalize()} missing")
    
    return min(score, 100), found, details

def analyze_sections(text):
    text_lower = text.lower()
    core_sections = ['experience', 'education', 'skills', 'projects']
    bonus_sections = ['summary', 'certifications', 'publications', 'achievements']
    core_found = [s for s in core_sections if s in text_lower]
    bonus_found = [s for s in bonus_sections if s in text_lower]
    found_sections = core_found + bonus_found
    missing_sections = [s for s in core_sections + bonus_sections if s not in text_lower]
    score = min((len(core_found) / 4 * 70) + (len(bonus_found) / 4 * 30), 100)
    return round(score, 1), [s.title() for s in found_sections], [s.title() for s in missing_sections]

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    common_fixes = [
        (r'\bnum\s+py\b', 'numpy'),
        (r'\bsci\s+py\b', 'scipy'),
        (r'\bgit\s+hub\b', 'github'),
        (r'\bsci\s+kit\b', 'scikit'),
        (r'\bten\s+sor\b', 'tensor'),
        (r'\bpy\s+torch\b', 'pytorch'),
        (r'\bopen\s+cv\b', 'opencv'),
        (r'\bmy\s+sql\b', 'mysql'),
        (r'\bno\s+sql\b', 'nosql'),
        (r'\bx\s+g\s*boost\b', 'xgboost'),
        (r'\bpow\s+er\s+bi\b', 'power bi'),
        (r'\btab\s+leau\b', 'tableau'),
    ]
    for pattern, replacement in common_fixes:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'\bml\s+model\b', 'machine learning model', text)
    text = re.sub(r'\bdl\s+model\b', 'deep learning model', text)
    text = re.sub(r'\bml\s+pipeline\b', 'machine learning pipeline', text)
    text = re.sub(r'\bml\b', 'machine learning', text)
    text = re.sub(r'\bml\s*&\s*analytics\b', 'machine learning analytics', text)
    text = re.sub(r'\bml\s*model\b', 'machine learning model', text)
    text = re.sub(r'\bml\s*pipeline\b', 'machine learning pipeline', text)
    text = re.sub(r'\bdl\b', 'deep learning', text)
    text = re.sub(r'\bnlp\b', 'natural language processing nlp', text)
    text = re.sub(r'\bcv\b', 'computer vision cv', text)
    return text

def keyword_found(kw, text):
    if kw.lower() in text:
        return True
    kw_compact = re.sub(r'[\s\-_]', '', kw.lower())
    text_compact = re.sub(r'[\s\-_]', '', text)
    if len(kw_compact) > 3 and kw_compact in text_compact:
        return True
    synonyms = {
        'machine learning': ['ml ', ' ml,', 'ml-', 'ml&'],
        'deep learning': ['dl ', ' dl,', 'dl-'],
        'deep learning': ['dl ', ' dl,', 'dl-', 'deeplearning'],
        'machine learning': ['ml ', ' ml,', 'ml-', 'machinelearning'],
        'natural language processing': ['nlp', 'naturallanguageprocessing'],
        'computer vision': ['cv ', 'computervision'],
        'neural network': ['nn ', 'neural net'],
        'natural language processing': ['nlp'],
        'power bi': ['powerbi'],
        'scikit-learn': ['sklearn', 'scikit learn'],
    }
    for canonical, alts in synonyms.items():
        if kw.lower() == canonical:
            for alt in alts:
                if alt in text:
                    return True
    return False

def analyze_keywords(text, job_role='Data Science'):
    text_normalized = normalize_text(text)
    role_keywords = TECHNICAL_KEYWORDS.get(job_role, [])
    general_keywords = TECHNICAL_KEYWORDS['General']
    role_found = [kw for kw in role_keywords if keyword_found(kw, text_normalized)]
    role_missing = [kw for kw in role_keywords if not keyword_found(kw, text_normalized)]
    general_found = [kw for kw in general_keywords if keyword_found(kw, text_normalized)]
    role_score = len(role_found) / max(len(role_keywords), 1) * 100
    gen_score = len(general_found) / max(len(general_keywords), 1) * 100
    kw_score = min((role_score * 0.80) + (gen_score * 0.20), 100)
    found_keywords = role_found + general_found
    return round(kw_score, 1), found_keywords, role_missing
    
    # Check with variations
    
    role_found = [kw for kw in role_keywords if keyword_found(kw, text_lower)]
    role_missing = [kw for kw in role_keywords if not keyword_found(kw, text_lower)]
    general_found = [kw for kw in general_keywords if keyword_found(kw, text_lower)]
    
    role_score = len(role_found) / max(len(role_keywords), 1) * 100
    gen_score = len(general_found) / max(len(general_keywords), 1) * 100
    kw_score = min((role_score * 0.80) + (gen_score * 0.20), 100)
    
    found_keywords = role_found + general_found
    return round(kw_score, 1), found_keywords, role_missing

def analyze_action_verbs(text):
    text_lower = text.lower()
    found_verbs = [v for v in ACTION_VERBS if v in text_lower]
    score = min(len(found_verbs) / 8 * 100, 100)
    details = []
    if len(found_verbs) >= 8:
        details.append("✅ Strong use of action verbs")
    elif len(found_verbs) >= 4:
        details.append("⚠️ Moderate use of action verbs — add more")
    else:
        details.append("❌ Very few action verbs — add impactful words")
    return round(score, 1), found_verbs, details

def analyze_quantification(text):
    patterns = [
        r'\d+\s*%', r'\d+\.\d+\s*%',
        r'\d+\s*(employees|records|slices|users|clients|projects)',
        r'\+\d+pp', r'\d+x\s+',
        r'(reduced|improved|increased|decreased)\s+by\s+\d+',
    ]
    matches = []
    for pattern in patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        matches.extend(found)
    score = min(len(matches) / 5 * 100, 100)
    details = []
    if len(matches) >= 5:
        details.append("✅ Good quantification of achievements")
    elif len(matches) >= 2:
        details.append("⚠️ Some numbers present — add more metrics")
    else:
        details.append("❌ No quantified achievements — add numbers/percentages")
    return round(score, 1), len(matches), details

def analyze_formatting(text):
    word_count = len(text.split())
    fmt = 0
    fmt += 40 if 250 <= word_count <= 900 else 15
    fmt += 35 if (text.count('-') + text.count('*') + text.count('•')) >= 5 else 10
    fmt += 25 if word_count >= 200 else 5
    details = []
    if 250 <= word_count <= 900:
        details.append(f"✅ Good length ({word_count} words)")
    elif word_count < 250:
        details.append(f"❌ Too short ({word_count} words)")
    else:
        details.append(f"⚠️ Too long ({word_count} words)")
    return min(fmt, 100), details

def calculate_ats_score(text, job_role):
    contact_score, contact_found, contact_details = analyze_contact_info(text)
    section_score, found_sections, missing_sections = analyze_sections(text)
    keyword_score, found_keywords, missing_keywords = analyze_keywords(text, job_role)
    verb_score, found_verbs, verb_details = analyze_action_verbs(text)
    quant_score, quant_count, quant_details = analyze_quantification(text)
    format_score, format_details = analyze_formatting(text)

    weights = {
        'Keywords Match': (keyword_score, 0.45),
        'Quantification': (quant_score, 0.15),
        'Section Structure': (section_score, 0.15),
        'Action Verbs': (verb_score, 0.07),
        'Contact Info': (contact_score, 0.13),
        'Formatting': (format_score, 0.05),
    }

    final_score = sum(score * weight for score, weight in weights.values())

    return {
        'final_score': round(final_score, 1),
        'weights': weights,
        'contact': (contact_score, contact_found, contact_details),
        'sections': (section_score, found_sections, missing_sections),
        'keywords': (keyword_score, found_keywords, missing_keywords),
        'verbs': (verb_score, found_verbs, verb_details),
        'quantification': (quant_score, quant_count, quant_details),
        'formatting': (format_score, format_details)
    }

def get_recommendations(results, job_role):
    recs = []
    score = results['final_score']
    _, _, missing_kw = results['keywords']
    _, found_sec, missing_sec = results['sections']
    _, contact_found, _ = results['contact']
    _, quant_count, _ = results['quantification']
    _, found_verbs, _ = results['verbs']

    if not contact_found.get('linkedin'):
        recs.append("🔗 Add your LinkedIn profile URL — recruiters expect it")
    if not contact_found.get('github'):
        recs.append("💻 Add your GitHub profile URL — essential for tech roles")
    if missing_kw[:5]:
        recs.append(f"🎯 Add these missing keywords: {', '.join(missing_kw[:5])}")
    if quant_count < 3:
        recs.append("📊 Add more numbers — e.g. '86% accuracy', 'reduced time by 40%'")
    if len(found_verbs) < 5:
        recs.append("💪 Start bullet points with action verbs: Built, Developed, Achieved")
    if 'Projects' not in found_sec:
        recs.append("📁 Add a Projects section — critical for fresher resumes")
    if 'Summary' not in found_sec and 'Objective' not in found_sec:
        recs.append("📝 Add a Professional Summary — first thing recruiters read")
    if missing_sec:
        recs.append(f"📋 Consider adding: {', '.join(missing_sec[:3])}")
    if score >= 80:
        recs.append("🌟 Excellent resume! Tailor keywords per job description for best results")
    elif score >= 60:
        recs.append("👍 Good resume! Address the above points to reach 80+")
    else:
        recs.append("⚠️ Resume needs significant improvement — focus on keywords and structure")
    return recs

# ==================== REPORTS ====================
def generate_excel_report(results, job_role):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        score = results['final_score']
        grade = 'Excellent' if score >= 80 else 'Good' if score >= 60 else 'Needs Improvement'
        pd.DataFrame([{'ATS Score': f"{score}/100", 'Job Role': job_role, 'Grade': grade}]).to_excel(
            writer, sheet_name='ATS Score', index=False)

        weights = results['weights']
        pd.DataFrame([{'Category': k, 'Score': f"{v[0]}/100", 'Weight': f"{int(v[1]*100)}%",
                       'Weighted Score': f"{round(v[0]*v[1], 1)}"}
                      for k, v in weights.items()]).to_excel(writer, sheet_name='Section Scores', index=False)

        _, found_kw, missing_kw = results['keywords']
        max_len = max(len(found_kw), len(missing_kw), 1)
        pd.DataFrame({'Found Keywords': found_kw + ['']*(max_len-len(found_kw)),
                      'Missing Keywords': missing_kw + ['']*(max_len-len(missing_kw))}).to_excel(
            writer, sheet_name='Keywords Analysis', index=False)

        recs = get_recommendations(results, job_role)
        pd.DataFrame({'Recommendations': recs}).to_excel(writer, sheet_name='Recommendations', index=False)
    return output.getvalue()

def generate_pdf_report(results, job_role):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.units import inch

    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter,
                            rightMargin=inch*0.5, leftMargin=inch*0.5,
                            topMargin=inch*0.5, bottomMargin=inch*0.5)
    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                 fontSize=18, textColor=colors.HexColor('#2196F3'))
    elements.append(Paragraph("ATS Resume Analysis Report", title_style))
    elements.append(Paragraph(f"Target Role: {job_role} | Generated by ATS Resume Analyzer", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    score = results['final_score']
    grade = 'Excellent' if score >= 80 else 'Good' if score >= 60 else 'Needs Improvement'
    score_color = '#4CAF50' if score >= 80 else '#FF9800' if score >= 60 else '#E53935'
    score_style = ParagraphStyle('Score', parent=styles['Heading1'],
                                 textColor=colors.HexColor(score_color), fontSize=20)
    elements.append(Paragraph(f"ATS Score: {score}/100 — {grade}", score_style))
    elements.append(Spacer(1, 0.15*inch))

    elements.append(Paragraph("Section Scores", styles['Heading2']))
    weights = results['weights']
    table_data = [['Category', 'Score', 'Weight', 'Weighted']]
    for k, v in weights.items():
        table_data.append([k, f"{v[0]}/100", f"{int(v[1]*100)}%", f"{round(v[0]*v[1], 1)}"])
    t = Table(table_data, colWidths=[2.5*inch, 1*inch, 1*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2196F3')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.15*inch))

    elements.append(Paragraph("Keywords Analysis", styles['Heading2']))
    _, found_kw, missing_kw = results['keywords']
    elements.append(Paragraph(f"Found ({len(found_kw)}): {', '.join(found_kw[:10])}", styles['Normal']))
    elements.append(Paragraph(f"Missing ({len(missing_kw)}): {', '.join(missing_kw[:10])}", styles['Normal']))
    elements.append(Spacer(1, 0.15*inch))

    elements.append(Paragraph("Recommendations", styles['Heading2']))
    for rec in get_recommendations(results, job_role):
        clean = re.sub(r'[^\x00-\x7F]+', '', rec).strip()
        elements.append(Paragraph(f"• {clean}", styles['Normal']))

    doc.build(elements)
    return output.getvalue()

# ==================== MAIN UI ====================
st.title("🎯 ATS Resume Analyzer")
st.markdown("**Upload your resume and get an industry-standard ATS score with detailed feedback!**")
st.markdown("---")

st.sidebar.image("https://img.icons8.com/color/96/resume.png", width=80)
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

job_role = st.sidebar.selectbox(
    "🎯 Target Job Role",
    ["Data Science", "Data Analyst", "ML Engineer", "AI Engineer",
     "Data Engineer", "NLP Engineer", "Computer Vision Engineer",
     "Research Scientist", "Product Analyst", "Software Engineer",
     "Full Stack Developer", "DevOps Engineer", "Cloud Architect",
     "Cybersecurity Analyst", "Blockchain Developer", "Business Analyst",
     "Finance Analyst", "Marketing Analyst", "HR Analyst"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Supported Formats:**
- 📄 PDF (Recommended)
- 🖼️ Image (PNG, JPG)
- 📝 DOCX (Word)

For best results, upload a PDF. Images work too, though a clear scan gives better accuracy!
""")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Resume",
    type=['pdf', 'png', 'jpg', 'jpeg', 'docx']
)

if uploaded_file is not None:
    with st.spinner("🔍 Extracting resume content..."):
        text = extract_text(uploaded_file)
        time.sleep(2)
    with st.spinner("📖 Parsing resume structure..."):
        time.sleep(2)
    with st.spinner("🎯 Matching keywords for your target role..."):
        time.sleep(3)
    with st.spinner("📊 Calculating weighted ATS score..."):
        results = calculate_ats_score(text, job_role)
        time.sleep(2)
    with st.spinner("💡 Generating personalized recommendations..."):
        time.sleep(2)

    if not text or len(text.strip()) < 50:
        st.error("❌ Could not extract text. Try a different format or clearer image.")
    else:
        score = results['final_score']

        if score >= 80:
            grade, color, emoji = "Excellent", "#4CAF50", "🟢"
        elif score >= 60:
            grade, color, emoji = "Good", "#FF9800", "🟡"
        elif score >= 40:
            grade, color, emoji = "Average", "#FF5722", "🟠"
        else:
            grade, color, emoji = "Needs Improvement", "#E53935", "🔴"

        col1, col2 = st.columns([1, 2])
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "ATS Score", 'font': {'size': 18}},
                number={'suffix': '/100', 'font': {'size': 28}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 40], 'color': '#FFEBEE'},
                        {'range': [40, 60], 'color': '#FFF3E0'},
                        {'range': [60, 80], 'color': '#FFF9C4'},
                        {'range': [80, 100], 'color': '#E8F5E9'}
                    ],
                    'threshold': {'line': {'color': color, 'width': 4}, 'thickness': 0.75, 'value': score}
                }
            ))
            fig.update_layout(height=300, margin=dict(t=50, b=0, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"## {emoji} {grade}")
            st.markdown(f"### Your resume scored **{score}/100** for **{job_role}** roles")
            st.markdown("---")
            for category, (cat_score, weight) in results['weights'].items():
                col_a, col_b = st.columns([3, 1])
                col_a.markdown(f"**{category}** ({int(weight*100)}% weight)")
                col_b.markdown(f"**{cat_score}/100**")
                st.progress(int(cat_score))

        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Section Analysis", "🔍 Keywords", "💡 Recommendations", "📊 Preview Report", "📥 Download"
        ])

        with tab1:
            st.subheader("📋 Section-by-Section Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📞 Contact Info")
                for d in results['contact'][2]:
                    st.markdown(d)
                st.markdown("#### 📁 Sections")
                for s in results['sections'][1]:
                    st.markdown(f"✅ {s}")
                for s in results['sections'][2][:4]:
                    st.markdown(f"❌ {s} missing")
            with col2:
                st.markdown("#### 💪 Action Verbs")
                for d in results['verbs'][2]:
                    st.markdown(d)
                if results['verbs'][1]:
                    st.markdown(f"**Found:** {', '.join(results['verbs'][1][:8])}")
                st.markdown("#### 📊 Quantification")
                for d in results['quantification'][2]:
                    st.markdown(d)
                st.markdown(f"**Metrics found:** {results['quantification'][1]}")
                st.markdown("#### 📝 Formatting")
                for d in results['formatting'][1]:
                    st.markdown(d)

        with tab2:
            st.subheader(f"🔍 Keywords Analysis for {job_role}")
            _, found_kw, missing_kw = results['keywords']
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### ✅ Found ({len(found_kw)})")
                for kw in found_kw:
                    st.markdown(f"✅ {kw}")
            with col2:
                st.markdown(f"### ❌ Missing ({len(missing_kw)})")
                for kw in missing_kw[:15]:
                    st.markdown(f"❌ {kw}")
            fig = px.bar(
                pd.DataFrame({'Status': ['Found', 'Missing'], 'Count': [len(found_kw), len(missing_kw)]}),
                x='Status', y='Count', color='Status',
                color_discrete_map={'Found': '#4CAF50', 'Missing': '#E53935'},
                title=f'Keywords Coverage for {job_role}'
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("💡 Personalized Recommendations")
            for i, rec in enumerate(get_recommendations(results, job_role), 1):
                st.markdown(f"**{i}.** {rec}")
            st.markdown("---")
            fig = px.bar(
                x=list(results['weights'].keys()),
                y=[v[0] for v in results['weights'].values()],
                color=[v[0] for v in results['weights'].values()],
                color_continuous_scale='RdYlGn',
                title='Score by Category',
                labels={'x': 'Category', 'y': 'Score'}
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("📊 Report Preview")
            st.markdown("#### 🏆 ATS Score Summary")
            st.dataframe(pd.DataFrame([{
                'ATS Score': f"{score}/100", 'Grade': grade, 'Target Role': job_role,
                'Keywords Found': len(results['keywords'][1]),
                'Keywords Missing': len(results['keywords'][2]),
                'Sections Found': len(results['sections'][1])
            }]), use_container_width=True)

            st.markdown("#### 📋 Section Scores")
            st.dataframe(pd.DataFrame([
                {'Category': k, 'Score': f"{v[0]}/100", 'Weight': f"{int(v[1]*100)}%"}
                for k, v in results['weights'].items()
            ]), use_container_width=True)

            st.markdown("#### 🔍 Keywords Preview")
            _, found_kw, missing_kw = results['keywords']
            max_len = max(len(found_kw), len(missing_kw), 1)
            st.dataframe(pd.DataFrame({
                'Found Keywords': found_kw + ['']*(max_len-len(found_kw)),
                'Missing Keywords': missing_kw + ['']*(max_len-len(missing_kw))
            }), use_container_width=True)

            st.markdown("#### 💡 Recommendations")
            st.dataframe(pd.DataFrame({'Recommendations': get_recommendations(results, job_role)}),
                        use_container_width=True)

        with tab5:
            st.subheader("📥 Download Full Report")
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📊 Download Excel Report",
                    data=generate_excel_report(results, job_role),
                    file_name="ats_resume_report.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=generate_pdf_report(results, job_role),
                    file_name="ats_resume_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

else:
    st.markdown("## 👋 Welcome to ATS Resume Analyzer!")
    st.markdown("Upload your resume from the sidebar to get started.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🎯 **ATS Score**\nIndustry-standard weighted scoring across 6 key categories")
    with col2:
        st.info("🔍 **Keywords Analysis**\nRole-specific keyword matching with found & missing list")
    with col3:
        st.info("💡 **Recommendations**\nPersonalized actionable tips to improve your resume")

st.markdown("---")
st.caption("Built by Krishan Kumar Chauhan | M.Tech Data Science, GBU | ATS Resume Analyzer v1.0")