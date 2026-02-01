# -*- coding: utf-8 -*-

"""
PDF Content Extraction and Manufacturing Analysis Tool
用于从多具身智能体协同生产调度PDF中提取和分析详细信息
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """
    从PDF提取内容的多种方案
    """
    
    print(f"📄 Attempting to extract content from: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return {}
    
    extracted_data = {
        'file': pdf_path,
        'methods_tried': [],
        'text_content': '',
        'structured_data': {},
        'manufacturing_concepts': {}
    }
    
    # Method 1: Try pdfplumber (best for structured data)
    try:
        import pdfplumber
        print("🔍 Method 1: Using pdfplumber...")
        
        with pdfplumber.open(pdf_path) as pdf:
            print(f"  Pages found: {len(pdf.pages)}")
            extracted_data['methods_tried'].append('pdfplumber')
            
            all_text = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    all_text.append(f"\n--- Page {i+1} ---\n{text}")
                
                # Try table extraction
                tables = page.extract_tables()
                if tables:
                    print(f"  Tables found on page {i+1}: {len(tables)}")
        
        extracted_data['text_content'] = ''.join(all_text)
        print("✅ pdfplumber extraction successful")
        return extracted_data
        
    except ImportError:
        print("  ℹ️  pdfplumber not installed")
    except Exception as e:
        print(f"  ❌ pdfplumber failed: {e}")
    
    # Method 2: Try PyPDF2
    try:
        from PyPDF2 import PdfReader
        print("🔍 Method 2: Using PyPDF2...")
        extracted_data['methods_tried'].append('PyPDF2')
        
        reader = PdfReader(pdf_path)
        print(f"  Pages found: {len(reader.pages)}")
        
        all_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                all_text.append(f"\n--- Page {i+1} ---\n{text}")
        
        extracted_data['text_content'] = ''.join(all_text)
        print("✅ PyPDF2 extraction successful")
        return extracted_data
        
    except ImportError:
        print("  ℹ️  PyPDF2 not installed")
    except Exception as e:
        print(f"  ❌ PyPDF2 failed: {e}")
    
    # Method 3: Try pymupdf (best for complex PDFs)
    try:
        import fitz  # pymupdf
        print("🔍 Method 3: Using PyMuPDF...")
        extracted_data['methods_tried'].append('pymupdf')
        
        doc = fitz.open(pdf_path)
        print(f"  Pages found: {len(doc)}")
        
        all_text = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text:
                all_text.append(f"\n--- Page {i+1} ---\n{text}")
            
            # Check if page is image-based
            if not text.strip():
                print(f"  ⚠️  Page {i+1} appears to be image-based (needs OCR)")
        
        extracted_data['text_content'] = ''.join(all_text)
        print("✅ PyMuPDF extraction successful")
        return extracted_data
        
    except ImportError:
        print("  ℹ️  PyMuPDF not installed")
    except Exception as e:
        print(f"  ❌ PyMuPDF failed: {e}")
    
    # Method 4: Try Tesseract OCR (for scanned documents)
    try:
        import pytesseract
        from pdf2image import convert_from_path
        print("🔍 Method 4: Using Tesseract OCR for scanned PDF...")
        extracted_data['methods_tried'].append('tesseract_ocr')
        
        # Convert PDF pages to images first
        print("  Converting PDF to images...")
        images = convert_from_path(pdf_path)
        
        all_text = []
        for i, image in enumerate(images):
            print(f"  OCR processing page {i+1}...")
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            if text:
                all_text.append(f"\n--- Page {i+1} ---\n{text}")
        
        extracted_data['text_content'] = ''.join(all_text)
        print("✅ Tesseract OCR extraction successful")
        return extracted_data
        
    except ImportError as e:
        print(f"  ℹ️  OCR dependencies not installed: {e}")
    except Exception as e:
        print(f"  ❌ Tesseract OCR failed: {e}")
    
    print("\n⚠️  All extraction methods failed or unavailable")
    return extracted_data


def analyze_manufacturing_content(text: str) -> Dict[str, Any]:
    """
    从提取的文本中识别制造相关概念
    """
    
    print("\n🏭 Analyzing manufacturing concepts...")
    
    analysis = {
        'job_types': [],
        'equipment_types': [],
        'constraints': [],
        'objectives': [],
        'agents': [],
        'processes': [],
        'metrics': []
    }
    
    # Define manufacturing keywords
    keywords = {
        'job_types': [
            'milling', 'turning', 'drilling', 'assembly', 'inspection', 
            '铣削', '车削', '钻孔', '装配', '检验', '焊接', '涂漆', '清洗'
        ],
        'equipment': [
            'robot', 'cnc', 'machine', 'agv', 'gripper', 'spindle',
            '机器人', '数控', '夹爪', '主轴', '输送'
        ],
        'constraints': [
            'deadline', 'capacity', 'inventory', 'tool life', 'precision',
            '截止时间', '容量', '库存', '刀具', '精度', '约束'
        ],
        'objectives': [
            'minimize', 'maximize', 'optimize', 'makespan', 'cost', 'quality',
            '最小化', '最大化', '优化', '成本', '质量', '效率'
        ],
        'metrics': [
            'oee', 'utilization', 'throughput', 'lead time', 'on-time',
            'OEE', '利用率', '吞吐量', '交期', '准时'
        ]
    }
    
    # Search for keywords in text
    text_lower = text.lower()
    for category, keyword_list in keywords.items():
        for keyword in keyword_list:
            if keyword.lower() in text_lower and keyword not in analysis.get(category.replace('job_types', 'job_types'), []):
                if category == 'job_types':
                    analysis['job_types'].append(keyword)
                elif category == 'equipment':
                    analysis['equipment_types'].append(keyword)
                elif category == 'constraints':
                    analysis['constraints'].append(keyword)
                elif category == 'objectives':
                    analysis['objectives'].append(keyword)
                elif category == 'metrics':
                    analysis['metrics'].append(keyword)
    
    return analysis


def generate_implementation_recommendations(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据PDF分析生成实现建议
    """
    
    print("\n💡 Generating implementation recommendations...")
    
    recommendations = {
        'identified_components': [],
        'missing_features': [],
        'integration_points': [],
        'priority_modules': []
    }
    
    # Based on job types found
    if analysis.get('job_types'):
        recommendations['identified_components'].append({
            'type': 'Operation Types',
            'items': list(set(analysis['job_types'])),
            'required_models': 'OperationType enum with identified operations'
        })
    
    # Based on equipment types
    if analysis.get('equipment_types'):
        recommendations['identified_components'].append({
            'type': 'Agent Types',
            'items': list(set(analysis['equipment_types'])),
            'required_models': 'AgentType enum with identified equipment'
        })
    
    # Identify constraints that need modeling
    constraints_found = list(set(analysis.get('constraints', [])))
    if constraints_found:
        recommendations['missing_features'].append({
            'feature': 'Constraint Modeling',
            'constraints_identified': constraints_found,
            'implementation': 'ResourceConstraint type with dynamic constraint updates',
            'priority': 'HIGH'
        })
    
    # Identify objectives
    objectives_found = list(set(analysis.get('objectives', [])))
    if objectives_found:
        recommendations['missing_features'].append({
            'feature': 'Multi-Objective Optimization',
            'objectives_identified': objectives_found,
            'implementation': 'Pareto optimization in scheduling',
            'priority': 'MEDIUM'
        })
    
    # Identify metrics
    metrics_found = list(set(analysis.get('metrics', [])))
    if metrics_found:
        recommendations['missing_features'].append({
            'feature': 'KPI Tracking',
            'metrics_identified': metrics_found,
            'implementation': 'PerformanceMetrics expansion',
            'priority': 'MEDIUM'
        })
    
    # Priority modules based on findings
    recommendations['priority_modules'] = [
        {
            'rank': 1,
            'module': 'Material Flow Management',
            'reason': 'Essential for realistic scheduling',
            'file': 'go/orchestrator/internal/workflows/material_flow.go'
        },
        {
            'rank': 2,
            'module': 'Dynamic Replanning Engine',
            'reason': 'Handle real-world disruptions',
            'file': 'go/orchestrator/internal/workflows/scheduling/dynamic_replanning.go'
        },
        {
            'rank': 3,
            'module': 'Multi-Embodiment Coordinator',
            'reason': 'Coordinate heterogeneous agents',
            'file': 'python/shannon/agents/embodiment_coordinator.py'
        },
        {
            'rank': 4,
            'module': 'Quality/Rework Handler',
            'reason': 'Production resilience',
            'file': 'go/orchestrator/internal/workflows/quality_recovery.go'
        },
        {
            'rank': 5,
            'module': 'Advanced Communication Protocol',
            'reason': 'Explicit agent coordination',
            'file': 'protos/embodiment_communication.proto'
        }
    ]
    
    return recommendations


def main():
    """Main extraction workflow"""
    
    print("=" * 80)
    print("🔬 Manufacturing PDF Content Extraction & Analysis Tool")
    print("=" * 80)
    print()
    
    # Find PDF files
    pdf_files = list(Path('.').glob('*.pdf')) + list(Path('.').glob('**/*.pdf'))
    
    if not pdf_files:
        print("📋 No PDF files found in current directory or subdirectories")
        print("\n📌 Expected PDF: 多具身智能体协同的生产调度项目.pdf")
        print("\n💡 Generating analysis based on standard manufacturing concepts...")
        
        # Generate standard recommendations
        standard_analysis = {
            'job_types': ['milling', 'turning', 'assembly', 'inspection', 'drilling'],
            'equipment_types': ['robot', 'cnc', 'agv', 'gripper'],
            'constraints': ['deadline', 'capacity', 'tool life', 'precision'],
            'objectives': ['minimize makespan', 'maximize utilization', 'on-time delivery'],
            'metrics': ['oee', 'fpy', 'otd', 'utilization']
        }
        recommendations = generate_implementation_recommendations(standard_analysis)
        
    else:
        print(f"✅ Found {len(pdf_files)} PDF file(s):")
        for pdf in pdf_files:
            print(f"  - {pdf}")
        
        # Process first PDF
        pdf_path = str(pdf_files[0])
        extracted = extract_pdf_content(pdf_path)
        
        if extracted.get('text_content'):
            print(f"\n📊 Extracted {len(extracted['text_content'])} characters")
            
            # Analyze content
            analysis = analyze_manufacturing_content(extracted['text_content'])
            print(f"\n✅ Analysis Results:")
            print(f"  Job Types: {len(analysis.get('job_types', []))} identified")
            print(f"  Equipment: {len(analysis.get('equipment_types', []))} identified")
            print(f"  Constraints: {len(analysis.get('constraints', []))} identified")
            print(f"  Objectives: {len(analysis.get('objectives', []))} identified")
            print(f"  Metrics: {len(analysis.get('metrics', []))} identified")
            
            # Generate recommendations
            recommendations = generate_implementation_recommendations(analysis)
        else:
            print("⚠️  Could not extract text content")
            recommendations = generate_implementation_recommendations({})
    
    # Output recommendations
    print("\n" + "=" * 80)
    print("📋 IMPLEMENTATION RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n🎯 Identified Components:")
    for comp in recommendations.get('identified_components', []):
        print(f"\n  {comp['type']}:")
        print(f"    Items: {', '.join(comp['items'])}")
        print(f"    Model: {comp['required_models']}")
    
    print("\n❌ Missing Features:")
    for feature in recommendations.get('missing_features', []):
        print(f"\n  {feature['feature']}:")
        print(f"    Priority: {feature['priority']}")
        print(f"    Implementation: {feature['implementation']}")
        if 'constraints_identified' in feature:
            print(f"    Constraints: {', '.join(feature['constraints_identified'])}")
        if 'objectives_identified' in feature:
            print(f"    Objectives: {', '.join(feature['objectives_identified'])}")
        if 'metrics_identified' in feature:
            print(f"    Metrics: {', '.join(feature['metrics_identified'])}")
    
    print("\n⭐ Priority Implementation Modules:")
    for module in recommendations.get('priority_modules', []):
        print(f"\n  [{module['rank']}] {module['module']}")
        print(f"      File: {module['file']}")
        print(f"      Reason: {module['reason']}")
    
    # Save recommendations to JSON
    output_file = 'pdf_analysis_recommendations.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Recommendations saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
