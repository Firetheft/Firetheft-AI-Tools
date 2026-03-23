import torch
import json
from typing import Tuple, Any, Dict, Optional, List
import logging
import re
import html

logger = logging.getLogger(__name__)

def comprehensive_cleanup(text: str) -> str:
    """
    综合文本清理函数，智能处理中英文标点符号
    """
    processed_text = str(text)

    try:
        def _decode_match(match):
            return chr(int(match.group(1), 16))
        
        processed_text = re.sub(r'(?i)\\+u([0-9a-f]{4})', _decode_match, processed_text)
    except Exception:
        pass

    processed_text = processed_text.strip()
    
    # 处理换行符
    processed_text = processed_text.replace('\\n', '\n')
    processed_text = re.sub(r'\r\n|\r', '\n', processed_text)
    
    # 移除HTML标签
    processed_text = re.sub(r'<[^>]+>', ' ', processed_text)
    processed_text = html.unescape(processed_text)
    
    # 移除Markdown格式
    processed_text = re.sub(r'^\s*[-*_]{3,}\s*$', '', processed_text, flags=re.MULTILINE)
    processed_text = re.sub(r'!\[.*?\]\(.*?\)', '', processed_text)
    processed_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', processed_text)
    processed_text = re.sub(r'^\s*#+\s+(.*)\s*$', r'\1', processed_text, flags=re.MULTILINE)
    processed_text = re.sub(r'^\s*>\s+', '', processed_text, flags=re.MULTILINE)
    processed_text = re.sub(r'^\s*[-*+]\s+', '', processed_text, flags=re.MULTILINE)
    
    # 移除Markdown加粗、斜体等
    processed_text = re.sub(r'(\*\*\*|___)(.*?)\1', r'\2', processed_text)
    processed_text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', processed_text)
    processed_text = re.sub(r'(\*|_)(.*?)\1', r'\2', processed_text)
    processed_text = re.sub(r'(~~)(.*?)\1', r'\2', processed_text)
    processed_text = re.sub(r'`(.*?)`', r'\1', processed_text)

    # 定义中英文标点符号
    chinese_commas = '，、'
    chinese_periods = '。！？；：'
    chinese_punctuation = chinese_commas + chinese_periods
    
    english_commas = ',;'
    english_periods = '.!?:'
    english_punctuation = english_commas + english_periods
    
    # 处理英文标点符号：确保后面有且只有一个空格（如果后面跟的是字母或数字）
    for punct in english_punctuation:
        # 英文标点后如果是字母数字，确保有一个空格
        processed_text = re.sub(
            rf'{re.escape(punct)}[ \t]*([a-zA-Z0-9\u4e00-\u9fff])', 
            rf'{punct} \1', 
            processed_text
        )
        # 英文标点后如果是多个空格，压缩为一个
        processed_text = re.sub(
            rf'{re.escape(punct)}[ \t]{{2,}}', 
            rf'{punct} ', 
            processed_text
        )
    
    # 处理中文标点符号：移除后面的空格
    for punct in chinese_punctuation:
        processed_text = re.sub(
            rf'{re.escape(punct)}[ \t]+', 
            punct, 
            processed_text
        )
    
    # 移除中文标点前的空格
    for punct in chinese_punctuation:
        processed_text = re.sub(
            rf'[ \t]+{re.escape(punct)}', 
            punct, 
            processed_text
        )
    
    # 处理连续的逗号（中英文混合）
    all_commas = chinese_commas + english_commas
    processed_text = re.sub(
        rf'([{re.escape(all_commas)}])[ \t]*([{re.escape(all_commas)}])', 
        r'\1', 
        processed_text
    )
    
    # 处理句号类标点后跟逗号类标点的情况（保留句号）
    all_periods = chinese_periods + english_periods
    processed_text = re.sub(
        rf'([{re.escape(all_periods)}])[ \t]*[{re.escape(all_commas)}]', 
        r'\1', 
        processed_text
    )
    
    # 移除重复的内容（保持原有逻辑）
    processed_text = re.sub(
        r'(.+?)(?:[,，、 \t]+\1)+', 
        r'\1', 
        processed_text, 
        flags=re.IGNORECASE
    )
    
    # 压缩多余空格（但保留英文标点后的单个空格）
    processed_text = re.sub(r'[ \t]{2,}', ' ', processed_text)
    
    # 压缩多余换行
    processed_text = re.sub(r'\n{3,}', '\n\n', processed_text)
    
    # 移除每行首尾空格
    processed_text = re.sub(r'^[ \t]+|[ \t]+$', '', processed_text, flags=re.MULTILINE)
    
    # 移除行首的标点符号
    all_punct = chinese_punctuation + english_punctuation
    processed_text = re.sub(
        rf'^\s*[{re.escape(all_punct)}]+', 
        '', 
        processed_text, 
        flags=re.MULTILINE
    )

    return processed_text.strip()


class TextViewerNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text_input": ("STRING", {"forceInput": True, "default": ""}), 
                "auto_format": ("BOOLEAN", {
                    "default": True,
                    "label_on": "启用格式化",
                    "label_off": "保持原文本"
                }),
            },
            "optional": {
                "exclude_text": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "要排除的单词或短语，用英文逗号分隔..."
                }),
                "find_text": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "要查找的单词或短语，用英文逗号分隔..."
                }),
                "replace_text": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "要替换的单词或短语，用英文逗号分隔..."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ('formatted_text',)
    OUTPUT_NODE = True
    FUNCTION = "process_and_display"
    CATEGORY = "📜Firetheft AI Tools"

    def process_and_display(
        self, 
        text_input: str, 
        auto_format: bool, 
        exclude_text: Optional[str] = "", 
        find_text: Optional[str] = "", 
        replace_text: Optional[str] = ""
    ) -> Dict:

        processed_text = str(text_input) 

        # 第一次格式化（如果启用）
        if auto_format:
            processed_text = comprehensive_cleanup(processed_text)

        # 排除指定文本
        if exclude_text and exclude_text.strip():
            exclusion_list = [term.strip() for term in exclude_text.split(',') if term.strip()]
            
            for term_to_exclude in exclusion_list:
                pattern = re.escape(term_to_exclude)
                processed_text = re.sub(pattern, '', processed_text, flags=re.IGNORECASE)
        
        # 查找和替换
        if find_text and find_text.strip() and replace_text is not None:
            find_list = [term.strip() for term in find_text.split(',') if term.strip()]
            replace_list = [term.strip() for term in replace_text.split(',')]

            if len(find_list) == len(replace_list):
                for i in range(len(find_list)):
                    pattern = re.escape(find_list[i])
                    processed_text = re.sub(pattern, replace_list[i], processed_text, flags=re.IGNORECASE)
            else:
                logger.warning(
                    f"[TextViewerNode] 查找和替换列表的长度不匹配。"
                    f"查找: {len(find_list)} 项, 替换: {len(replace_list)} 项。将跳过替换。"
                )

        # 第二次格式化（如果启用）
        if auto_format:
            processed_text = comprehensive_cleanup(processed_text)

        formatted_text = processed_text.strip()

        return {
            "ui": {
                "text_output": [formatted_text]
            },
            "result": (formatted_text,)
        }


class ConcatTextViewerNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "auto_format": ("BOOLEAN", {
                    "default": True,
                    "label_on": "启用格式化",
                    "label_off": "保持原文本"
                }),
            },
            "optional": {
                "text_input_1": ("STRING", {"multiline": False, "default": "", "forceInput": False}),
                "text_input_2": ("STRING", {"multiline": False, "default": "", "forceInput": False}),
                "text_input_3": ("STRING", {"multiline": False, "default": "", "forceInput": False}),
                "text_input_4": ("STRING", {"multiline": False, "default": "", "forceInput": False}),
                "separator": ("STRING", {"multiline": False, "default": ""}),
                "exclude_text": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "要排除的单词或短语，用英文逗号分隔..."
                }),
                "find_text": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "要查找的单词或短语，用英文逗号分隔..."
                }),
                "replace_text": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "要替换的单词或短语，用英文逗号分隔..."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ('formatted_text',)
    OUTPUT_NODE = True
    FUNCTION = "process_and_display"
    CATEGORY = "📜Firetheft AI Tools"

    def process_and_display(
        self, 
        auto_format: bool, 
        text_input_1: Optional[str] = "",
        text_input_2: Optional[str] = "",
        text_input_3: Optional[str] = "",
        text_input_4: Optional[str] = "",
        separator: Optional[str] = "",
        exclude_text: Optional[str] = "",
        find_text: Optional[str] = "", 
        replace_text: Optional[str] = ""
    ) -> Dict: 

        # 处理分隔符
        processed_separator = str(separator)
        if processed_separator:
            processed_separator = processed_separator.replace('\\n', '\n').replace('\\t', '\t')
            
        # 合并非空输入
        all_inputs = [text_input_1, text_input_2, text_input_3, text_input_4]
        non_empty_inputs = [str(i).strip() for i in all_inputs if i and str(i).strip()]
        
        if non_empty_inputs:
            processed_text = processed_separator.join(non_empty_inputs)
        else:
            processed_text = ""

        if not processed_text:
            return {
                "ui": {"text_output": [""]},
                "result": ("",)
            }

        # 第一次格式化（如果启用）
        if auto_format:
            processed_text = comprehensive_cleanup(processed_text)

        # 排除指定文本
        if exclude_text and exclude_text.strip():
            exclusion_list = [term.strip() for term in exclude_text.split(',') if term.strip()]
            
            for term_to_exclude in exclusion_list:
                pattern = re.escape(term_to_exclude)
                processed_text = re.sub(pattern, '', processed_text, flags=re.IGNORECASE)
        
        # 查找和替换
        if find_text and find_text.strip() and replace_text is not None:
            find_list = [term.strip() for term in find_text.split(',') if term.strip()]
            replace_list = [term.strip() for term in replace_text.split(',')]

            if len(find_list) == len(replace_list):
                for i in range(len(find_list)):
                    pattern = re.escape(find_list[i])
                    processed_text = re.sub(pattern, replace_list[i], processed_text, flags=re.IGNORECASE)
            else:
                logger.warning(
                    f"[ConcatTextViewerNode] 查找和替换列表的长度不匹配。"
                    f"查找: {len(find_list)} 项, 替换: {len(replace_list)} 项。将跳过替换。"
                )

        # 第二次格式化（如果启用）
        if auto_format:
            processed_text = comprehensive_cleanup(processed_text)

        formatted_text = processed_text.strip()

        return {
            "ui": {
                "text_output": [formatted_text]
            },
            "result": (formatted_text,)
        }


NODE_CLASS_MAPPINGS = {
    "TextViewerNode": TextViewerNode,
    "ConcatTextViewerNode": ConcatTextViewerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextViewerNode": "Text Viewer（文本查看器）",
    "ConcatTextViewerNode": "Concat Text Viewer（文本连结查看器）"
}