# 🚀 Shannon 前端应用 - 部署指南

## 📋 部署前检查清单

### 环境准备
- [ ] Node.js 18+ 已安装
- [ ] npm 或 yarn 包管理器可用
- [ ] Git 版本控制配置
- [ ] 环境变量文件已准备

### 依赖验证
```bash
# 检查 Node 版本
node --version  # v18.17.0 或更高

# 检查 npm 版本
npm --version   # 9.0.0 或更高

# 清理依赖缓存
npm cache clean --force

# 重新安装依赖
npm install

# 验证安装
npm list --depth=0
```

---

## 🏗️ 构建步骤

### 1. 生产构建
```bash
# 构建应用
npm run build

# 输出验证
# 预期产物：
# - .next/standalone/
# - .next/static/
# - public/
```

### 2. 静态分析
```bash
# ESLint 检查
npm run lint

# TypeScript 类型检查
npx tsc --noEmit

# 依赖安全扫描
npm audit

# 修复已知漏洞
npm audit fix
```

### 3. 本地验证
```bash
# 构建后测试
npm run build && npm start

# 访问地址
# http://localhost:3000

# 检查关键页面
# - / (首页)
# - /run-detail (任务详情)
# - /agent-interaction (可视化)
# - /agents (智能体)
# - /settings (设置)
```

---

## 🌐 部署选项

### 选项 1: Vercel (推荐)

#### 优势
- 自动 CI/CD
- 免费 HTTPS
- 全球 CDN
- 环境变量管理
- 零配置部署

#### 步骤
```bash
# 1. 安装 Vercel CLI
npm install -g vercel

# 2. 登录 Vercel
vercel login

# 3. 部署
vercel deploy

# 4. 设置生产环境
vercel --prod
```

**vercel.json 配置示例**:
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "env": {
    "NEXT_PUBLIC_API_URL": "https://api.example.com"
  }
}
```

### 选项 2: Docker

#### Dockerfile
```dockerfile
# 构建阶段
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# 运行阶段
FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
ENV NEXT_TELEMETRY_DISABLED=1
CMD ["node", "server.js"]
```

#### 构建和运行
```bash
# 构建镜像
docker build -t shannon-frontend:latest .

# 运行容器
docker run -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=https://api.example.com \
  shannon-frontend:latest

# 推送到注册表
docker tag shannon-frontend:latest your-registry/shannon-frontend:latest
docker push your-registry/shannon-frontend:latest
```

### 选项 3: Nginx

#### nginx.conf
```nginx
upstream nextjs {
    server 127.0.0.1:3000;
}

server {
    listen 80;
    server_name app.example.com;

    gzip on;
    gzip_types text/plain text/css text/xml text/javascript 
               application/json application/javascript application/xml+rss;

    location /_next/static {
        alias /app/.next/static;
        expires 365d;
        add_header Cache-Control "public, immutable";
    }

    location ~ ^/api/ {
        proxy_pass http://nextjs;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://nextjs;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 🔒 安全配置

### 环境变量
创建 `.env.production`:
```env
# API 配置
NEXT_PUBLIC_API_URL=https://api.example.com
NEXT_PUBLIC_API_TIMEOUT=30000

# 功能开关
NEXT_PUBLIC_ENABLE_ANALYTICS=true
NEXT_PUBLIC_ENABLE_SENTRY=true

# Sentry 错误追踪
NEXT_PUBLIC_SENTRY_DSN=https://your-sentry-dsn

# 应用配置
NEXT_PUBLIC_APP_NAME=Shannon
NEXT_PUBLIC_APP_VERSION=0.1.0
```

### HTTPS/TLS
- 启用 HSTS (Strict-Transport-Security)
- 使用 Let's Encrypt 证书
- 配置 SSL/TLS 1.3+

### CSP (内容安全策略)
```
default-src 'self';
script-src 'self' 'unsafe-inline' 'unsafe-eval';
style-src 'self' 'unsafe-inline';
img-src 'self' data: https:;
font-src 'self' data:;
connect-src 'self' https://api.example.com;
```

---

## 📊 监控和日志

### 错误追踪 (Sentry)
```bash
# 安装 Sentry
npm install @sentry/nextjs

# 配置 next.config.ts
import { withSentryConfig } from "@sentry/nextjs";

export default withSentryConfig(
  { /* ... */ },
  { org: "your-org", project: "shannon" }
);
```

### 性能监控
```typescript
// lib/performance.ts
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

export function reportWebVitals(metric) {
  // 发送到分析服务
  console.log(metric);
}
```

### 应用日志
```typescript
// 在浏览器控制台中可查看结构化日志
if (process.env.NODE_ENV === 'development') {
  console.log('[APP] 应用启动...');
}
```

---

## 🧪 部署后测试

### 功能测试
```bash
# 测试关键路由
curl https://app.example.com/
curl https://app.example.com/agent-interaction

# 检查 API 连接
curl https://api.example.com/health
```

### 性能测试
```bash
# 使用 PageSpeed Insights
https://pagespeed.web.dev/

# 使用 Lighthouse
npm install -g lighthouse
lighthouse https://app.example.com/ --view
```

### 安全检查
```bash
# SSL 证书检查
https://www.sslshopper.com/ssl-checker.html

# 安全头检查
https://securityheaders.com/
```

### 跨浏览器测试
- [ ] Chrome (最新)
- [ ] Firefox (最新)
- [ ] Safari (最新)
- [ ] Edge (最新)
- [ ] iOS Safari
- [ ] Chrome Mobile

---

## 🔄 持续部署 (CI/CD)

### GitHub Actions 示例
```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Build
        run: npm run build
      
      - name: Run tests
        run: npm run lint
      
      - name: Deploy to Vercel
        uses: vercel/action@master
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
```

---

## 📈 性能基准

部署后应达到以下指标：

| 指标 | 目标 | 检查工具 |
|------|------|---------|
| LCP (Largest Contentful Paint) | < 2.5s | Lighthouse, Web Vitals |
| FID (First Input Delay) | < 100ms | Web Vitals |
| CLS (Cumulative Layout Shift) | < 0.1 | Lighthouse, Web Vitals |
| First Contentful Paint | < 1.8s | Lighthouse |
| Time to Interactive | < 3.8s | Lighthouse |

---

## 🆘 故障排查

### 构建失败
```bash
# 清理缓存
rm -rf .next node_modules
npm install

# 检查 Node 版本
node --version

# 检查依赖
npm audit
```

### 性能问题
```bash
# 分析包大小
npm run build -- --analyze

# 检查 Next.js 配置优化
# - 启用 SWR (Stale-While-Revalidate)
# - 配置图片优化
# - 启用压缩
```

### 内存泄漏
```bash
# 使用 Node 内存快照
node --inspect-brk server.js

# 在 Chrome DevTools 中分析
chrome://inspect/
```

---

## 📞 支持和维护

### 定期维护任务
- [ ] 每周检查依赖更新
- [ ] 每月安全审计
- [ ] 每月性能分析
- [ ] 季度大版本升级评估

### 监控关键指标
- 应用错误率
- API 响应时间
- 页面加载时间
- 用户活跃度

---

## ✅ 部署检查清单

- [ ] 所有依赖已更新
- [ ] 环境变量已配置
- [ ] 构建成功无错误
- [ ] Lint 检查通过
- [ ] 类型检查通过
- [ ] 本地测试通过
- [ ] 跨浏览器测试通过
- [ ] 性能指标达标
- [ ] 安全检查通过
- [ ] 监控系统就绪
- [ ] 备份已创建
- [ ] 回滚计划已准备

---

**祝部署顺利！** 🎉
