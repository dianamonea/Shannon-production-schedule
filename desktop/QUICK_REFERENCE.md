# 📌 Shannon 前端应用 - 快速参考

## 🚀 快速启动

```bash
# 开发环境
npm install
npm run dev

# 生产构建
npm run build
npm start

# 桌面应用
npm run tauri:dev
npm run tauri:build
```

## 📁 关键文件位置

| 功能 | 文件路径 |
|------|---------|
| 多智能体可视化 | `app/(app)/agent-interaction/page.tsx` |
| 全局样式 | `app/globals.css` |
| 应用布局 | `components/app-layout.tsx` |
| 错误边界 | `components/error-boundary.tsx` |
| 加载状态 | `components/loading-states.tsx` |
| Redux Store | `lib/store.ts` |
| API 客户端 | `lib/shannon/api.ts` |

## 🎯 常见任务

### 修改配色
编辑 `app/globals.css`:
- `:root` - 浅色模式
- `.dark` - 深色模式

### 添加新页面
1. 在 `app/(app)/` 创建目录
2. 添加 `page.tsx`
3. 在 `components/app-sidebar.tsx` 添加导航

### 添加新组件
1. 创建 `components/YourComponent.tsx`
2. 使用 TypeScript 和 Tailwind CSS
3. 导出组件供页面使用

### 修改 API 端点
编辑 `lib/shannon/api.ts`:
```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
```

## 🔧 常用命令

```bash
# 代码检查
npm run lint

# 类型检查
npx tsc --noEmit

# 构建分析
npm run build -- --analyze

# 清理缓存
npm cache clean --force

# 更新依赖
npm update

# 安全审计
npm audit
npm audit fix
```

## 📦 核心依赖

| 包 | 版本 | 用途 |
|----|------|------|
| next | 16.0.3 | React 框架 |
| react | 19.2.0 | UI 库 |
| @reduxjs/toolkit | 2.10.1 | 状态管理 |
| tailwindcss | 4.x | 样式框架 |
| @xyflow/react | 12.9.3 | 流程图 |
| lucide-react | 0.553.0 | 图标库 |

## 🌐 环境变量

```env
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_API_TIMEOUT=30000
```

## 📊 目录结构

```
desktop/
├── app/                  # Next.js 应用
│   ├── (app)/           # 受保护的路由
│   ├── (auth)/          # 认证路由
│   ├── api/             # API 路由
│   ├── globals.css      # 全局样式
│   └── layout.tsx       # 根布局
├── components/          # React 组件
│   ├── ui/              # shadcn/ui 组件
│   ├── error-boundary.tsx
│   └── loading-states.tsx
├── lib/                 # 工具和配置
│   ├── features/        # Redux slices
│   ├── shannon/         # API 客户端
│   └── store.ts         # Redux 存储
└── public/              # 静态资源
```

## 🎨 Tailwind CSS 常用类

```tsx
// 背景和文字
<div className="bg-background text-foreground">

// 卡片
<div className="bg-card rounded-lg shadow-md">

// 按钮
<button className="bg-primary text-primary-foreground">

// 输入框
<input className="bg-input border border-border">

// 深色模式
<div className="dark:bg-slate-800">
```

## 🔐 安全最佳实践

- ✅ 使用环境变量存储敏感信息
- ✅ 验证所有用户输入
- ✅ 使用 HTTPS
- ✅ 定期更新依赖
- ✅ 监控错误日志

## 📱 响应式设计断点

| 断点 | 像素 | Tailwind 前缀 |
|------|------|--------------|
| 超小 | 320px | (无) |
| 小 | 640px | sm: |
| 中 | 768px | md: |
| 大 | 1024px | lg: |
| 超大 | 1280px | xl: |
| 2XL | 1536px | 2xl: |

## 🧪 测试命令

```bash
# 运行 ESLint
npm run lint

# 类型检查
npx tsc --noEmit

# 构建测试
npm run build

# 本地启动
npm start
```

## 🆘 常见问题

**Q: 页面不加载？**
A: 检查 API_URL 环境变量，查看浏览器控制台错误

**Q: 样式不正确？**
A: 清理 `.next` 文件夹，重新构建

**Q: 内存不足？**
A: 增加 Node 内存: `NODE_OPTIONS=--max-old-space-size=4096`

**Q: 深色模式不工作？**
A: 检查 `<html class="dark">` 属性

## 📞 获取帮助

- 查看 [DELIVERY_NOTES.md](./DELIVERY_NOTES.md) - 完整交付说明
- 查看 [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) - 部署指南
- 查看 [README.md](./README.md) - 项目说明

## 🎯 下一步

1. 部署到生产环境
2. 设置监控和日志
3. 收集用户反馈
4. 迭代改进功能

---

**最后更新**: 2026-01-30 | **版本**: 0.1.0
