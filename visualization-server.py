#!/usr/bin/env python3
"""
多智能体交互可视化 Web 服务器
用于展示生产调度中智能体之间的交互流程
"""

import os
import json
import glob
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import mimetypes


class AgentVisualizationHandler(SimpleHTTPRequestHandler):
    """处理智能体可视化的 HTTP 请求"""

    def do_GET(self):
        """处理 GET 请求"""
        parsed_path = urlparse(self.path)
        
        # API 路由：获取最新的调度结果
        if parsed_path.path == '/api/latest-schedule':
            self.send_latest_schedule()
            return
        
        # 主页面：返回 HTML 可视化
        if parsed_path.path == '/' or parsed_path.path == '/index.html':
            self.send_visualization_page()
            return
        
        # 其他静态文件
        super().do_GET()

    def send_latest_schedule(self):
        """发送最新的调度结果"""
        try:
            # 查找最新的 schedule_result_*.json 文件
            current_dir = Path(__file__).parent
            schedule_files = sorted(
                glob.glob(str(current_dir / 'schedule_result_*.json')),
                key=os.path.getctime,
                reverse=True
            )

            if not schedule_files:
                self.send_json_response({
                    'error': '未找到调度结果文件',
                    'path': str(current_dir)
                }, 404)
                return

            # 读取最新的文件
            with open(schedule_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.send_json_response(data, 200)

        except Exception as e:
            self.send_json_response({
                'error': f'读取文件失败: {str(e)}'
            }, 500)

    def send_visualization_page(self):
        """发送可视化页面"""
        try:
            page_path = Path(__file__).parent / 'agent-interaction-visualization.html'
            with open(page_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(content.encode('utf-8')))
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))

        except Exception as e:
            self.send_error(500, f'读取页面失败: {str(e)}')

    def send_json_response(self, data, status_code=200):
        """发送 JSON 响应"""
        try:
            json_data = json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
            
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
            self.send_header('Content-Length', len(json_data))
            self.end_headers()
            self.wfile.write(json_data)

        except Exception as e:
            self.send_error(500, str(e))

    def do_OPTIONS(self):
        """处理 CORS 预检请求"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        """自定义日志输出"""
        print(f"[{self.log_date_time_string()}] {format % args}")


def run_server(host='localhost', port=8888):
    """启动 Web 服务器"""
    server_address = (host, port)
    httpd = HTTPServer(server_address, AgentVisualizationHandler)
    
    print(f"""
╔════════════════════════════════════════════════════════╗
║   多智能体交互可视化 Web 服务                            ║
║   Agent Interaction Visualization Server               ║
╚════════════════════════════════════════════════════════╝

📊 可视化页面: http://{host}:{port}
📡 API 端点:   http://{host}:{port}/api/latest-schedule

功能：
  • 实时显示生产扰动
  • 展示三类智能体的交互流程
  • 可视化扰动应对策略
  • 智能体协同统计

按 Ctrl+C 停止服务器...
    """)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ 服务器已关闭")
        httpd.server_close()


if __name__ == '__main__':
    import sys
    
    host = sys.argv[1] if len(sys.argv) > 1 else 'localhost'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8888
    
    run_server(host, port)
