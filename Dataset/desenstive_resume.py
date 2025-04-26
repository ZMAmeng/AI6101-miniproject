# -*- coding: utf-8 -*-
"""resume_pii_anonymization.py

简历数据集脱敏处理工具
"""

import pandas as pd
import re
import json
from collections import defaultdict
import hashlib
from tqdm.auto import tqdm
from typing import List, Dict, Any, Tuple, Set, Optional
import argparse

# 尝试导入presidio库，如果不可用则提供警告
try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    PRESIDIO_AVAILABLE = True
except ImportError:
    print("警告: Presidio库未安装，将使用基本的正则表达式进行脱敏处理")
    print("如需完整功能，请安装: pip install presidio-analyzer presidio-anonymizer spacy")
    print("安装spacy后，还需要: python -m spacy download en_core_web_lg")
    PRESIDIO_AVAILABLE = False

tqdm.pandas()

# 定义可用的敏感信息类型
SENSITIVE_INFO_TYPES = {
    "PERSON_NAME": "个人姓名",
    "EMAIL_ADDRESS": "电子邮件地址",
    "PHONE_NUMBER": "电话号码",
    "SOCIAL_MEDIA": "社交媒体链接",
    "DATE_OF_BIRTH": "出生日期",
    "GENDER": "性别信息",
    "MARITAL_STATUS": "婚姻状况",
    "FAMILY_INFO": "家庭信息",
    "ADDRESS": "地址信息",
    "EDUCATION_DATES": "教育经历日期",
    "WORK_DATES": "工作经历日期",
    "PHOTO_REFERENCES": "照片引用",
    "AGE": "年龄信息",
    "NATIONALITY": "国籍信息",
    "RELIGION": "宗教信息",
    "LOCATION": "位置信息",
    "COMPANY_NAME": "公司名称",
    "SCHOOL_NAME": "学校名称"
}


# 配置简历数据集的识别规则
def configure_resume_recognizers(selected_types: Optional[List[str]] = None):
    """
    针对简历数据集特性的识别器配置

    Args:
        selected_types: 可选的敏感信息类型列表，如果为None则使用所有类型

    Returns:
        list: 配置好的识别器列表
    """
    if not PRESIDIO_AVAILABLE:
        return []

    # 如果未指定类型，使用所有类型
    if selected_types is None:
        selected_types = list(SENSITIVE_INFO_TYPES.keys())

    recognizers = []

    # 1. 个人姓名识别器
    if "PERSON_NAME" in selected_types:
        name_patterns = [
            Pattern(
                name="resume_name_pattern",
                regex=r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                score=0.7
            ),
            Pattern(
                name="resume_name_with_middle_pattern",
                regex=r'\b([A-Z][a-z]+\s+[A-Z][a-z]*\.\s+[A-Z][a-z]+)\b',
                score=0.7
            ),
            Pattern(
                name="father_name_pattern",
                regex=r"Father(?:'|')?s Name\s*[:：]\s*([^\n,]+)",
                score=0.8
            )
        ]
        name_recognizer = PatternRecognizer(
            supported_entity="PERSON_NAME",
            patterns=name_patterns,
            context=["name", "full name", "father", "mother"]
        )
        recognizers.append(name_recognizer)

    # 2. 电子邮件地址识别器
    if "EMAIL_ADDRESS" in selected_types:
        email_pattern = Pattern(
            name="resume_email_pattern",
            regex=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            score=0.9
        )
        email_recognizer = PatternRecognizer(
            supported_entity="EMAIL_ADDRESS",
            patterns=[email_pattern],
            context=["email", "e-mail", "mail"]
        )
        recognizers.append(email_recognizer)

    # 3. 电话号码识别器
    if "PHONE_NUMBER" in selected_types:
        phone_patterns = [
            Pattern(
                name="resume_phone_pattern1",
                regex=r'\b(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){1,2}\d{3,4}[-.\s]?\d{3,4}\b',
                score=0.85
            ),
            Pattern(
                name="resume_phone_pattern2",
                regex=r'(?:Phone|Tel|Mobile|Contact)(?:\s*(?:Number|No|#|\:))?\s*[:：]?\s*(\+?[\d\s\(\)\-\.]{7,})',
                score=0.85
            )
        ]
        phone_recognizer = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=phone_patterns,
            context=["phone", "mobile", "cell", "telephone", "contact"]
        )
        recognizers.append(phone_recognizer)

    # 4. 社交媒体链接识别器
    if "SOCIAL_MEDIA" in selected_types:
        social_media_patterns = [
            Pattern(
                name="linkedin_pattern",
                regex=r'(?:linkedin\.com/in/[a-zA-Z0-9_-]+)',
                score=0.85
            ),
            Pattern(
                name="github_pattern",
                regex=r'(?:github\.com/[a-zA-Z0-9_-]+)',
                score=0.85
            ),
            Pattern(
                name="twitter_pattern",
                regex=r'(?:twitter\.com/[a-zA-Z0-9_-]+)',
                score=0.85
            ),
            Pattern(
                name="facebook_pattern",
                regex=r'(?:facebook\.com/[a-zA-Z0-9_.-]+)',
                score=0.85
            )
        ]
        social_media_recognizer = PatternRecognizer(
            supported_entity="SOCIAL_MEDIA",
            patterns=social_media_patterns,
            context=["profile", "social", "media", "link"]
        )
        recognizers.append(social_media_recognizer)

    # 5. 出生日期识别器
    if "DATE_OF_BIRTH" in selected_types:
        dob_patterns = [
            Pattern(
                name="dob_pattern1",
                regex=r'(?:Date\s+of\s+Birth|DOB|Birth\s+Date)(?:\s*(?:\:|\())?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
                score=0.85
            ),
            Pattern(
                name="dob_pattern2",
                regex=r'(?:Date\s+of\s+Birth|DOB|Birth\s+Date)(?:\s*(?:\:|\())?\s*(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
                score=0.85
            ),
            Pattern(
                name="dob_pattern3",
                regex=r'(?:Date\s+of\s+Birth|DOB|Birth\s+Date)(?:\s*(?:\:|\())?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
                score=0.85
            )
        ]
        dob_recognizer = PatternRecognizer(
            supported_entity="DATE_OF_BIRTH",
            patterns=dob_patterns,
            context=["birth", "born", "DOB"]
        )
        recognizers.append(dob_recognizer)

    # 6. 性别信息识别器
    if "GENDER" in selected_types:
        gender_patterns = [
            Pattern(
                name="gender_pattern1",
                regex=r'(?:Gender|Sex)\s*[:：]\s*([^\n,]+)',
                score=0.85
            ),
            Pattern(
                name="gender_pattern2",
                regex=r'\((?:M|F|Male|Female|男|女)\)',
                score=0.7
            ),
            Pattern(
                name="gender_pattern3",
                regex=r'Gender\s*[:-]?\s*(Male|Female|M|F|男|女)',
                score=0.85
            )
        ]
        gender_recognizer = PatternRecognizer(
            supported_entity="GENDER",
            patterns=gender_patterns,
            context=["gender", "sex", "male", "female"]
        )
        recognizers.append(gender_recognizer)

    # 7. 婚姻状况识别器
    if "MARITAL_STATUS" in selected_types:
        marital_patterns = [
            Pattern(
                name="marital_status_pattern",
                regex=r'(?:Marital\s+Status|Marriage\s+Status)\s*[:：]\s*([^\n,]+)',
                score=0.85
            )
        ]
        marital_recognizer = PatternRecognizer(
            supported_entity="MARITAL_STATUS",
            patterns=marital_patterns,
            context=["marital", "married", "single", "divorced"]
        )
        recognizers.append(marital_recognizer)

    # 8. 地址识别器
    if "ADDRESS" in selected_types:
        address_patterns = [
            Pattern(
                name="address_pattern1",
                regex=r'\b\d+\s+[A-Za-z0-9\s,]+(?:Street|Avenue|Road|Blvd|Drive|Lane|Place|Way|Apt|Suite|St|Rd|Dr|Ave)\b',
                score=0.7
            ),
            Pattern(
                name="address_pattern2",
                regex=r'(?:Address|Location)\s*[:：]\s*([^\n]+)',
                score=0.7
            ),
            Pattern(
                name="address_pattern3",
                regex=r'\b\d+/[A-Za-z0-9],?\s+[A-Za-z0-9\s,]+,\s+[A-Za-z\s]+,\s+[A-Za-z\s]+\b',
                score=0.7
            )
        ]
        address_recognizer = PatternRecognizer(
            supported_entity="ADDRESS",
            patterns=address_patterns,
            context=["address", "location", "residence", "live"]
        )
        recognizers.append(address_recognizer)

    # 9. 教育日期识别器
    if "EDUCATION_DATES" in selected_types:
        education_date_patterns = [
            Pattern(
                name="education_date_pattern1",
                regex=r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+to\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
                score=0.75
            ),
            Pattern(
                name="education_date_pattern2",
                regex=r'\b(?:19|20)\d{2}\s+to\s+(?:19|20)\d{2}\b',
                score=0.7
            ),
            Pattern(
                name="education_date_pattern3",
                regex=r'\b(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}\b',
                score=0.7
            ),
            Pattern(
                name="education_date_pattern4",
                regex=r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
                score=0.6
            )
        ]
        education_date_recognizer = PatternRecognizer(
            supported_entity="EDUCATION_DATES",
            patterns=education_date_patterns,
            context=["education", "university", "college", "school", "degree"]
        )
        recognizers.append(education_date_recognizer)

    # 10. 工作日期识别器
    if "WORK_DATES" in selected_types:
        work_date_patterns = [
            Pattern(
                name="work_date_pattern1",
                regex=r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+to\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
                score=0.75
            ),
            Pattern(
                name="work_date_pattern2",
                regex=r'\b(?:19|20)\d{2}\s+to\s+(?:19|20)\d{2}|[Pp]resent\b',
                score=0.7
            ),
            Pattern(
                name="work_date_pattern3",
                regex=r'\b(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}|[Pp]resent\b',
                score=0.7
            )
        ]
        work_date_recognizer = PatternRecognizer(
            supported_entity="WORK_DATES",
            patterns=work_date_patterns,
            context=["experience", "work", "job", "company", "employment"]
        )
        recognizers.append(work_date_recognizer)

    # 11. 公司名称识别器
    if "COMPANY_NAME" in selected_types:
        company_patterns = [
            Pattern(
                name="company_name_pattern1",
                regex=r'[Cc]ompany\s*[-:]?\s*([^\n,]+)',
                score=0.7
            ),
            Pattern(
                name="company_name_pattern2",
                regex=r'[Cc]ompany\s+[Nn]ame\s*[:：]\s*([^\n,]+)',
                score=0.8
            ),
            Pattern(
                name="company_name_pattern3",
                regex=r'[Ee]mployer\s*[:：]\s*([^\n,]+)',
                score=0.8
            ),
            Pattern(
                name="company_name_pattern4",
                regex=r'[Ww]orked\s+(?:at|for|with)\s+([A-Z][A-Za-z0-9\s&,.]+(?:Inc|LLC|Ltd|Limited|Corp|Corporation|Co|Company))',
                score=0.7
            )
        ]
        company_recognizer = PatternRecognizer(
            supported_entity="COMPANY_NAME",
            patterns=company_patterns,
            context=["company", "employer", "organization", "firm"]
        )
        recognizers.append(company_recognizer)

    # 12. 学校名称识别器
    if "SCHOOL_NAME" in selected_types:
        school_patterns = [
            Pattern(
                name="school_name_pattern1",
                regex=r'(?:University|College|Institute|School)\s+of\s+([^\n,]+)',
                score=0.7
            ),
            Pattern(
                name="school_name_pattern2",
                regex=r'(?:University|College|Institute|School)\s*[:：]\s*([^\n,]+)',
                score=0.7
            ),
            Pattern(
                name="school_name_pattern3",
                regex=r'(?:[A-Z][a-z]+\s+)+(?:University|College|Institute|School)',
                score=0.7
            )
        ]
        school_recognizer = PatternRecognizer(
            supported_entity="SCHOOL_NAME",
            patterns=school_patterns,
            context=["education", "university", "college", "school", "institute"]
        )
        recognizers.append(school_recognizer)

    # 13. 年龄信息识别器
    if "AGE" in selected_types:
        age_patterns = [
            Pattern(
                name="age_pattern1",
                regex=r'(?:Age|Years)\s*[:：]\s*(\d{1,2})',
                score=0.8
            ),
            Pattern(
                name="age_pattern2",
                regex=r'(\d{1,2})\s+[Yy]ears\s+[Oo]ld',
                score=0.8
            ),
            Pattern(
                name="age_pattern3",
                regex=r'[Aa]ge[:：]?\s*(\d{1,2})',
                score=0.8
            )
        ]
        age_recognizer = PatternRecognizer(
            supported_entity="AGE",
            patterns=age_patterns,
            context=["age", "years old", "year old"]
        )
        recognizers.append(age_recognizer)

    return recognizers


# 初始化分析引擎和匿名化引擎
def init_engines(selected_types=None):
    """初始化分析引擎和匿名化引擎"""
    if not PRESIDIO_AVAILABLE:
        return None, None

    analyzer = AnalyzerEngine()
    for recognizer in configure_resume_recognizers(selected_types):
        analyzer.registry.add_recognizer(recognizer)

    anonymizer = AnonymizerEngine()

    return analyzer, anonymizer


# 提取简历ID或生成唯一标识符
def extract_resume_id(resume_content):
    """
    从简历内容中提取ID或生成唯一标识符

    Args:
        resume_content: 简历内容

    Returns:
        str: 简历ID或唯一标识符
    """
    # 尝试从内容中提取电子邮件作为ID
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_content)
    if email_match:
        # 对电子邮件进行哈希处理以保护隐私
        return f"email-{hashlib.md5(email_match.group().encode()).hexdigest()[:8]}"

    # 尝试从内容中提取姓名作为ID
    name_match = re.search(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b', resume_content)
    if name_match:
        # 对姓名进行哈希处理以保护隐私
        return f"name-{hashlib.md5(name_match.group().encode()).hexdigest()[:8]}"

    # 如果都没找到，使用内容的哈希值
    return f"resume-{hashlib.md5(resume_content.encode()).hexdigest()[:12]}"


# 额外的敏感信息提取函数
def extract_additional_pii(text):
    """
    提取额外的敏感信息

    Args:
        text: 文本内容

    Returns:
        list: 敏感信息列表
    """
    results = []

    # 1. 提取出生日期和性别组合
    dob_gender_match = re.search(r'Date\s+of\s+Birth\s*\(\s*Gender\s*\)\s*:\s*(\d{4}-\d{2}-\d{2})\s*\(\s*([MF])\s*\)',
                                 text)
    if dob_gender_match:
        dob = dob_gender_match.group(1)
        gender = dob_gender_match.group(2)

        results.append({
            "entity_type": "DATE_OF_BIRTH",
            "start": dob_gender_match.start(1),
            "end": dob_gender_match.start(1) + len(dob),
            "score": 0.9,
            "text": dob
        })

        results.append({
            "entity_type": "GENDER",
            "start": dob_gender_match.start(2),
            "end": dob_gender_match.start(2) + len(gender),
            "score": 0.9,
            "text": gender
        })

    # 2. 提取家庭信息
    family_match = re.search(r'(?:Father|Mother)(?:\'|\')?s\s+Name\s*[:：]\s*([^\n]+)', text)
    if family_match:
        family_info = family_match.group(1).strip()

        results.append({
            "entity_type": "FAMILY_INFO",
            "start": family_match.start(1),
            "end": family_match.start(1) + len(family_info),
            "score": 0.85,
            "text": family_info
        })

    # 3. 提取婚姻状况
    marital_match = re.search(r'Marital\s+Status\s*[:：]\s*([^\n,]+)', text)
    if marital_match:
        marital_status = marital_match.group(1).strip()

        results.append({
            "entity_type": "MARITAL_STATUS",
            "start": marital_match.start(1),
            "end": marital_match.start(1) + len(marital_status),
            "score": 0.85,
            "text": marital_status
        })

    # 4. 提取国籍信息
    nationality_match = re.search(r'Nationality\s*[:：]\s*([^\n,]+)', text)
    if nationality_match:
        nationality = nationality_match.group(1).strip()

        results.append({
            "entity_type": "NATIONALITY",
            "start": nationality_match.start(1),
            "end": nationality_match.start(1) + len(nationality),
            "score": 0.85,
            "text": nationality
        })

    # 5. 提取照片引用
    photo_match = re.search(r'(?:Photo|Picture|Image)\s*[:：]?\s*([^\n]+\.(?:jpg|jpeg|png|gif))', text)
    if photo_match:
        photo_ref = photo_match.group(1).strip()

        results.append({
            "entity_type": "PHOTO_REFERENCES",
            "start": photo_match.start(1),
            "end": photo_match.start(1) + len(photo_ref),
            "score": 0.8,
            "text": photo_ref
        })

    return results


# 使用正则表达式的基本脱敏处理
def basic_anonymize_text(text, selected_types=None):
    """
    使用基本的正则表达式进行脱敏处理

    Args:
        text: 文本内容
        selected_types: 可选的敏感信息类型列表

    Returns:
        tuple: (脱敏后的文本, 敏感信息字典)
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text, {}

    # 如果未指定类型，使用所有类型
    if selected_types is None:
        selected_types = list(SENSITIVE_INFO_TYPES.keys())

    anonymized_text = text
    pii_entities = defaultdict(list)

    # 定义正则表达式模式和替换值
    patterns = []

    # 个人姓名
    if "PERSON_NAME" in selected_types:
        patterns.extend([
            (r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b', "PERSON_NAME", "<NAME>"),
            (r"Father(?:'|')?s Name\s*[:：]\s*([^\n,]+)", "PERSON_NAME", "Father's Name: <NAME>"),
            (r"Mother(?:'|')?s Name\s*[:：]\s*([^\n,]+)", "PERSON_NAME", "Mother's Name: <NAME>")
        ])

    # 电子邮件地址
    if "EMAIL_ADDRESS" in selected_types:
        patterns.append(
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "EMAIL_ADDRESS", "<EMAIL>")
        )

    # 电话号码
    if "PHONE_NUMBER" in selected_types:
        patterns.extend([
            (r'\b(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){1,2}\d{3,4}[-.\s]?\d{3,4}\b', "PHONE_NUMBER", "<PHONE>"),
            (
            r'(?:Phone|Tel|Mobile|Contact)(?:\s*(?:Number|No|#|\:))?\s*[:：]?\s*(\+?[\d\s\(\)\-\.]{7,})', "PHONE_NUMBER",
            lambda m: m.group().replace(m.group(1), "<PHONE>"))
        ])

    # 社交媒体链接
    if "SOCIAL_MEDIA" in selected_types:
        patterns.extend([
            (r'(?:linkedin\.com/in/[a-zA-Z0-9_-]+)', "SOCIAL_MEDIA", "<LINKEDIN>"),
            (r'(?:github\.com/[a-zA-Z0-9_-]+)', "SOCIAL_MEDIA", "<GITHUB>"),
            (r'(?:twitter\.com/[a-zA-Z0-9_-]+)', "SOCIAL_MEDIA", "<TWITTER>"),
            (r'(?:facebook\.com/[a-zA-Z0-9_.-]+)', "SOCIAL_MEDIA", "<FACEBOOK>")
        ])

    # 出生日期
    if "DATE_OF_BIRTH" in selected_types:
        patterns.extend([
            (r'(?:Date\s+of\s+Birth|DOB|Birth\s+Date)(?:\s*(?:\:|\())?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
             "DATE_OF_BIRTH", lambda m: m.group().replace(m.group(1), "<DOB>")),
            (r'(?:Date\s+of\s+Birth|DOB|Birth\s+Date)(?:\s*(?:\:|\())?\s*(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
             "DATE_OF_BIRTH", lambda m: m.group().replace(m.group(1), "<DOB>")),
            (r'(?:Date\s+of\s+Birth|DOB|Birth\s+Date)(?:\s*(?:\:|\())?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
             "DATE_OF_BIRTH", lambda m: m.group().replace(m.group(1), "<DOB>"))
        ])

    # 性别信息
    if "GENDER" in selected_types:
        patterns.extend([
            (r'(?:Gender|Sex)\s*[:：]\s*([^\n,]+)', "GENDER", lambda m: m.group().replace(m.group(1), "<GENDER>")),
            (r'\((?:M|F|Male|Female|男|女)\)', "GENDER", "(<GENDER>)"),
            (r'Gender\s*[:-]?\s*(Male|Female|M|F|男|女)', "GENDER", lambda m: m.group().replace(m.group(1), "<GENDER>"))
        ])

    # 婚姻状况
    if "MARITAL_STATUS" in selected_types:
        patterns.append(
            (r'(?:Marital\s+Status|Marriage\s+Status)\s*[:：]\s*([^\n,]+)', "MARITAL_STATUS",
             lambda m: m.group().replace(m.group(1), "<MARITAL_STATUS>"))
        )

    # 地址信息
    if "ADDRESS" in selected_types:
        patterns.extend([
            (r'\b\d+\s+[A-Za-z0-9\s,]+(?:Street|Avenue|Road|Blvd|Drive|Lane|Place|Way|Apt|Suite|St|Rd|Dr|Ave)\b',
             "ADDRESS", "<ADDRESS>"),
            (
            r'(?:Address|Location)\s*[:：]\s*([^\n]+)', "ADDRESS", lambda m: m.group().replace(m.group(1), "<ADDRESS>")),
            (r'\b\d+/[A-Za-z0-9],?\s+[A-Za-z0-9\s,]+,\s+[A-Za-z\s]+,\s+[A-Za-z\s]+\b', "ADDRESS", "<ADDRESS>")
        ])

    # 教育日期
        # 教育日期
        if "EDUCATION_DATES" in selected_types:
            patterns.extend([
                (
                r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+to\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
                "EDUCATION_DATES", "<EDUCATION_DATES>"),
                (r'\b(?:19|20)\d{2}\s+to\s+(?:19|20)\d{2}\b', "EDUCATION_DATES", "<EDUCATION_DATES>"),
                (r'\b(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}\b', "EDUCATION_DATES", "<EDUCATION_DATES>"),
                (r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b', "EDUCATION_DATES", "<DATE>")
            ])

        # 工作日期
        if "WORK_DATES" in selected_types:
            patterns.extend([
                (
                r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+to\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
                "WORK_DATES", "<WORK_DATES>"),
                (r'\b(?:19|20)\d{2}\s+to\s+(?:19|20)\d{2}|[Pp]resent\b', "WORK_DATES", "<WORK_DATES>"),
                (r'\b(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}|[Pp]resent\b', "WORK_DATES", "<WORK_DATES>")
            ])

        # 公司名称
        if "COMPANY_NAME" in selected_types:
            patterns.extend([
                (r'[Cc]ompany\s*[-:]?\s*([^\n,]+)', "COMPANY_NAME",
                 lambda m: m.group().replace(m.group(1), "<COMPANY>")),
                (r'[Cc]ompany\s+[Nn]ame\s*[:：]\s*([^\n,]+)', "COMPANY_NAME",
                 lambda m: m.group().replace(m.group(1), "<COMPANY>")),
                (r'[Ee]mployer\s*[:：]\s*([^\n,]+)', "COMPANY_NAME",
                 lambda m: m.group().replace(m.group(1), "<COMPANY>")),
                (
                r'[Ww]orked\s+(?:at|for|with)\s+([A-Z][A-Za-z0-9\s&,.]+(?:Inc|LLC|Ltd|Limited|Corp|Corporation|Co|Company))',
                "COMPANY_NAME", lambda m: m.group().replace(m.group(1), "<COMPANY>"))
            ])

        # 学校名称
        if "SCHOOL_NAME" in selected_types:
            patterns.extend([
                (r'(?:University|College|Institute|School)\s+of\s+([^\n,]+)', "SCHOOL_NAME",
                 lambda m: m.group().replace(m.group(1), "<SCHOOL>")),
                (r'(?:University|College|Institute|School)\s*[:：]\s*([^\n,]+)', "SCHOOL_NAME",
                 lambda m: m.group().replace(m.group(1), "<SCHOOL>")),
                (r'(?:[A-Z][a-z]+\s+)+(?:University|College|Institute|School)', "SCHOOL_NAME", "<SCHOOL>")
            ])

        # 年龄信息
        if "AGE" in selected_types:
            patterns.extend([
                (r'(?:Age|Years)\s*[:：]\s*(\d{1,2})', "AGE", lambda m: m.group().replace(m.group(1), "<AGE>")),
                (r'(\d{1,2})\s+[Yy]ears\s+[Oo]ld', "AGE", lambda m: m.group().replace(m.group(1), "<AGE>")),
                (r'[Aa]ge[:：]?\s*(\d{1,2})', "AGE", lambda m: m.group().replace(m.group(1), "<AGE>"))
            ])

        # 应用所有模式
        for pattern, entity_type, replacement in patterns:
            matches = re.finditer(pattern, anonymized_text)
            for match in matches:
                if callable(replacement):
                    # 如果替换值是函数，则调用它
                    entity_text = match.group(1) if match.groups() else match.group(0)
                    if entity_text not in pii_entities[entity_type.lower()]:
                        pii_entities[entity_type.lower()].append(entity_text)

                    # 应用替换函数
                    segment = match.group(0)
                    replaced_segment = replacement(match)
                    anonymized_text = anonymized_text.replace(segment, replaced_segment)
                else:
                    # 直接替换
                    entity_text = match.group(1) if match.groups() else match.group(0)
                    if entity_text not in pii_entities[entity_type.lower()]:
                        pii_entities[entity_type.lower()].append(entity_text)

                    anonymized_text = anonymized_text.replace(match.group(0), replacement)

        return anonymized_text, dict(pii_entities)

# 使用Presidio进行高级脱敏处理
def presidio_anonymize_text(text, analyzer, anonymizer, selected_types=None):
    """
    使用Presidio进行高级脱敏处理

    Args:
        text: 文本内容
        analyzer: Presidio分析引擎
        anonymizer: Presidio匿名化引擎
        selected_types: 可选的敏感信息类型列表

    Returns:
        tuple: (脱敏后的文本, 敏感信息字典)
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text, {}

    # 如果未指定类型，使用所有类型
    if selected_types is None:
        selected_types = list(SENSITIVE_INFO_TYPES.keys())

    try:
        # 映射Presidio支持的实体类型
        presidio_entities = []

        # 添加Presidio内置的实体类型
        if "PERSON_NAME" in selected_types:
            presidio_entities.extend(["PERSON", "NRP"])
        if "EMAIL_ADDRESS" in selected_types:
            presidio_entities.append("EMAIL_ADDRESS")
        if "PHONE_NUMBER" in selected_types:
            presidio_entities.append("PHONE_NUMBER")
        if "ADDRESS" in selected_types:
            presidio_entities.append("ADDRESS")
        if "LOCATION" in selected_types:
            presidio_entities.append("LOCATION")

        # 添加自定义实体类型
        for entity_type in selected_types:
            if entity_type not in ["PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "ADDRESS", "LOCATION"]:
                presidio_entities.append(entity_type)

        # 识别敏感实体
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=presidio_entities,
            score_threshold=0.65
        )

        # 添加额外提取的敏感信息
        additional_results = extract_additional_pii(text)
        for additional_entity in additional_results:
            # 如果实体类型在选定类型中，则添加
            if additional_entity["entity_type"] in selected_types:
                results.append(
                    RecognizerResult(
                        entity_type=additional_entity["entity_type"],
                        start=additional_entity["start"],
                        end=additional_entity["end"],
                        score=additional_entity["score"],
                        analysis_explanation=None
                    )
                )

        # 收集识别到的敏感信息
        pii_entities = defaultdict(list)
        for result in results:
            entity_text = text[result.start:result.end]
            entity_type = result.entity_type.lower()
            if entity_text not in pii_entities[entity_type]:
                pii_entities[entity_type].append(entity_text)

        # 配置匿名化操作
        operators = {
            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
            "PERSON_NAME": OperatorConfig("replace", {"new_value": "<NAME>"}),
            "PERSON": OperatorConfig("replace", {"new_value": "<NAME>"}),
            "NRP": OperatorConfig("replace", {"new_value": "<NAME>"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
            "SOCIAL_MEDIA": OperatorConfig("replace", {"new_value": "<SOCIAL_MEDIA>"}),
            "DATE_OF_BIRTH": OperatorConfig("replace", {"new_value": "<DOB>"}),
            "GENDER": OperatorConfig("replace", {"new_value": "<GENDER>"}),
            "MARITAL_STATUS": OperatorConfig("replace", {"new_value": "<MARITAL_STATUS>"}),
            "FAMILY_INFO": OperatorConfig("replace", {"new_value": "<FAMILY_INFO>"}),
            "ADDRESS": OperatorConfig("replace", {"new_value": "<ADDRESS>"}),
            "EDUCATION_DATES": OperatorConfig("replace", {"new_value": "<EDUCATION_DATES>"}),
            "WORK_DATES": OperatorConfig("replace", {"new_value": "<WORK_DATES>"}),
            "PHOTO_REFERENCES": OperatorConfig("replace", {"new_value": "<PHOTO>"}),
            "AGE": OperatorConfig("replace", {"new_value": "<AGE>"}),
            "NATIONALITY": OperatorConfig("replace", {"new_value": "<NATIONALITY>"}),
            "RELIGION": OperatorConfig("replace", {"new_value": "<RELIGION>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
            "COMPANY_NAME": OperatorConfig("replace", {"new_value": "<COMPANY>"}),
            "SCHOOL_NAME": OperatorConfig("replace", {"new_value": "<SCHOOL>"})
        }

        # 匿名化处理
        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators
        )

        return anonymized.text, dict(pii_entities)
    except Exception as e:
        print(f"使用Presidio处理文本时出错: {e}")
        import traceback
        traceback.print_exc()

        # 出错时使用基本的正则表达式替换方法
        print("尝试使用基本方法处理文本...")
        return basic_anonymize_text(text, selected_types)

# 脱敏处理函数
def process_text(text: str, analyzer=None, anonymizer=None, selected_types=None):
    """
    处理文本，返回脱敏后的文本和识别到的敏感信息

    Args:
        text: 文本内容
        analyzer: Presidio分析引擎
        anonymizer: Presidio匿名化引擎
        selected_types: 可选的敏感信息类型列表

    Returns:
        tuple: (anonymized_text, pii_entities)
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text, {}

    try:
        # 如果Presidio可用且引擎已初始化，则使用Presidio
        if PRESIDIO_AVAILABLE and analyzer and anonymizer:
            return presidio_anonymize_text(text, analyzer, anonymizer, selected_types)
        else:
            # 否则使用基本的正则表达式替换
            return basic_anonymize_text(text, selected_types)
    except Exception as e:
        print(f"处理文本时出错: {e}")
        import traceback
        traceback.print_exc()

        # 最后的备用方案：返回原文本和空的敏感信息字典
        return text, {}

# 数据预处理管道
# 修改后的数据处理函数
def process_resume_dataset(input_path: str, output_path: str, pii_json_path: str,
                           selected_types=None, sample_size=None, content_column=None):
    """
    执行完整数据处理流程

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        pii_json_path: 敏感信息JSON文件路径
        selected_types: 可选的敏感信息类型列表
        sample_size: 抽样数量(调试用)
        content_column: 简历内容所在的列名(如果为None，则合并相关列)
    """
    try:
        # 初始化引擎
        analyzer, anonymizer = init_engines(selected_types)

        # 读取数据
        print(f"开始读取数据: {input_path}")
        df = pd.read_csv(
            input_path,
            parse_dates=False,
            encoding='utf-8',
            engine='c',
            memory_map=True,
            on_bad_lines='warn'  # 处理错误行
        )

        print(f"成功读取数据，共 {len(df)} 条记录")
        print(f"数据集列名: {df.columns.tolist()}")

        # 数据抽样 (调试时启用)
        if sample_size:
            df = df.sample(min(sample_size, len(df)), random_state=42)
            print(f"已抽样 {len(df)} 条记录用于处理")

        # 创建按简历ID组织的敏感信息字典
        organized_entities = {}

        # 如果没有指定内容列，则合并相关列创建一个完整的简历内容
        if content_column is None or content_column not in df.columns:
            print("未找到指定的内容列，将尝试合并相关列...")

            # 确定可能包含简历内容的列
            content_columns = []
            for col in df.columns:
                # 检查列名是否包含这些关键词
                if any(keyword in col.lower() for keyword in
                       ['description', 'detail', 'skill', 'education', 'company']):
                    content_columns.append(col)

            if not content_columns:
                # 如果没有找到明确的内容列，使用所有非ID列
                content_columns = [col for col in df.columns if col.lower() != 'id']

            print(f"将合并以下列作为简历内容: {content_columns}")

            # 创建一个新列，合并所有相关列的内容
            df['combined_content'] = df.apply(
                lambda row: '\n\n'.join(
                    str(row[col]) for col in content_columns if pd.notna(row[col]) and str(row[col]).strip()),
                axis=1
            )
            content_column = 'combined_content'

        # 执行脱敏并收集敏感信息
        print("⏳ 开始PII脱敏处理...")

        # 定义处理函数
        def process_resume(resume_content):
            if not isinstance(resume_content, str) or len(resume_content.strip()) == 0:
                return resume_content

            resume_id = extract_resume_id(resume_content)
            try:
                anonymized_text, pii_entities = process_text(
                    resume_content, analyzer, anonymizer, selected_types
                )

                # 如果识别到敏感信息，则保存
                if pii_entities:
                    organized_entities[resume_id] = pii_entities

                return anonymized_text
            except Exception as e:
                print(f"处理简历 {resume_id} 时出错: {e}")
                return resume_content

        # 应用处理函数
        df['anonymized_content'] = df[content_column].progress_apply(process_resume)

        # 定期保存敏感信息，避免全部处理完才保存
        def save_pii_data(current_pii_data, path, iteration):
            temp_path = f"{path}.part{iteration}"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(current_pii_data, f, ensure_ascii=False, indent=2)
            print(f"已保存中间敏感信息到: {temp_path}")

        # 每1000条记录保存一次
        save_interval = 1000
        for i in range(0, len(df), save_interval):
            if i > 0 and len(organized_entities) > 0:
                save_pii_data(organized_entities, pii_json_path, i // save_interval)

        # 保存脱敏后的数据
        print(f"保存处理结果到: {output_path}")
        df.to_csv(
            output_path,
            index=False,
            encoding='utf-8',
            quoting=2  # 对非数值字段强制添加引号
        )

        # 保存敏感信息JSON
        print(f"保存敏感信息到: {pii_json_path}")
        with open(pii_json_path, 'w', encoding='utf-8') as f:
            json.dump(organized_entities, f, ensure_ascii=False, indent=2)

        # 输出一些统计信息
        pii_count = len(organized_entities)
        print(f"包含敏感信息的简历数量: {pii_count} ({pii_count / len(df) * 100:.2f}%)")

        # 统计各类型敏感信息数量
        entity_type_counts = defaultdict(int)
        entity_counts = 0
        for resume_data in organized_entities.values():
            for entity_type, entities_list in resume_data.items():
                entity_type_counts[entity_type] += len(entities_list)
                entity_counts += len(entities_list)

        print(f"总共提取了 {entity_counts} 个敏感实体")
        print("各类型敏感实体统计:")
        for entity_type, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {entity_type}: {count}")

        print(f"✅ 处理完成！")

    except Exception as e:
        print(f"处理数据集时出错: {e}")
        import traceback
        traceback.print_exc()


# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='简历数据集脱敏处理工具')
    parser.add_argument('--input', type=str, default='/scratch/duanyiyang/ForgettingLLm/datasets/resume/UpdatedResumeDataSet.csv', help='输入CSV文件路径')
    parser.add_argument('--output', type=str, default='/scratch/duanyiyang/ForgettingLLm/datasets/resume/UpdatedResumeDataSet_ano.csv', help='输出CSV文件路径')
    parser.add_argument('--pii-json', type=str, default='/scratch/duanyiyang/ForgettingLLm/datasets/resume/PIIinfo.csv', help='敏感信息JSON文件路径')
    parser.add_argument('--content-column', type=str, default='content', help='简历内容所在的列名')
    parser.add_argument('--sample', type=int, default=None, help='抽样数量(调试用)')
    parser.add_argument('--types', type=str, nargs='+', default=None,
                        choices=list(SENSITIVE_INFO_TYPES.keys()),
                        help='要处理的敏感信息类型')

    return parser.parse_args()

# 执行示例
if __name__ == "__main__":
    args = parse_args()

    # 打印可用的敏感信息类型
    print("可用的敏感信息类型:")
    for type_key, type_desc in SENSITIVE_INFO_TYPES.items():
        print(f"  - {type_key}: {type_desc}")

    # 打印选择的敏感信息类型
    if args.types:
        print("\n已选择的敏感信息类型:")
        for type_key in args.types:
            print(f"  - {type_key}: {SENSITIVE_INFO_TYPES[type_key]}")
    else:
        print("\n将处理所有敏感信息类型")

    # 执行处理
    process_resume_dataset(
        input_path=args.input,
        output_path=args.output,
        pii_json_path=args.pii_json,
        selected_types=args.types,
        sample_size=args.sample,
        content_column=args.content_column
    )

