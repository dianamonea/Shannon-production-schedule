# Shannon 通信协议详解

**版本**: 1.0  
**更新日期**: 2026年1月30日  
**作者**: Shannon Team  

本文档详细介绍 Shannon 多智能体编排平台中使用的各种通信协议、使用场景及代码示例。

---

## 📚 目录

1. [协议概览](#协议概览)
2. [gRPC 协议](#grpc-协议)
3. [HTTP/REST API](#httprest-api)
4. [WebSocket 协议](#websocket-协议)
5. [Server-Sent Events (SSE)](#server-sent-events-sse)
6. [Temporal Workflow](#temporal-workflow)
7. [Redis Pub/Sub](#redis-pubsub)
8. [协议选择指南](#协议选择指南)

---

## 协议概览

Shannon 平台采用多协议架构，针对不同场景使用最合适的通信方式：

| 协议 | 用途 | 优势 | 使用场景 |
|------|------|------|----------|
| **gRPC** | 微服务间通信 | 高性能、类型安全、双向流 | Agent Core ↔ Orchestrator |
| **HTTP/REST** | 公共 API | 通用、易用、防火墙友好 | 客户端 ↔ Gateway |
| **WebSocket** | 双向实时通信 | 低延迟、持久连接 | 实时任务状态更新 |
| **SSE** | 服务端推送 | 单向流、断点续传 | 任务执行日志流 |
| **Temporal** | 工作流编排 | 可靠性、状态管理 | 复杂多步骤任务 |
| **Redis Pub/Sub** | 事件总线 | 解耦、扩展性 | 跨服务事件通知 |

---

## gRPC 协议

### 1. 概述

gRPC 是基于 HTTP/2 的高性能 RPC 框架，Shannon 使用 Protocol Buffers 定义服务接口。

### 2. 架构位置

```
┌──────────┐         gRPC          ┌──────────────┐
│ Gateway  │ ──────────────────────▶│ Orchestrator │
└──────────┘                        └──────────────┘
                                           │
                                           │ gRPC
                                           ▼
                                    ┌──────────────┐
                                    │  Agent Core  │
                                    │    (Rust)    │
                                    └──────────────┘
```

### 3. Proto 定义示例

**文件位置**: `protos/agent/agent.proto`

```protobuf
syntax = "proto3";

package shannon.agent;
option go_package = "github.com/Kocoro-lab/Shannon/go/orchestrator/internal/pb/agent";

import "google/protobuf/struct.proto";
import "google/protobuf/timestamp.proto";
import "common/common.proto";

// Agent 服务定义
service AgentService {
  // 执行任务（一次性响应）
  rpc ExecuteTask(ExecuteTaskRequest) returns (ExecuteTaskResponse);
  
  // 流式执行任务（实时返回进度）
  rpc StreamExecuteTask(ExecuteTaskRequest) returns (stream TaskUpdate);
  
  // 获取智能体能力
  rpc GetCapabilities(GetCapabilitiesRequest) returns (GetCapabilitiesResponse);
  
  // 健康检查
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
  
  // 发现工具
  rpc DiscoverTools(DiscoverToolsRequest) returns (DiscoverToolsResponse);
}

// 任务执行请求
message ExecuteTaskRequest {
  shannon.common.TaskMetadata metadata = 1;
  string query = 2;
  google.protobuf.Struct context = 3;
  shannon.common.ExecutionMode mode = 4;
  repeated string available_tools = 5;
  AgentConfig config = 6;
  SessionContext session_context = 7;
}

// 任务执行响应
message ExecuteTaskResponse {
  string task_id = 1;
  shannon.common.StatusCode status = 2;
  string result = 3;
  repeated shannon.common.ToolCall tool_calls = 4;
  repeated shannon.common.ToolResult tool_results = 5;
  shannon.common.ExecutionMetrics metrics = 6;
  string error_message = 7;
  AgentState final_state = 8;
}

// 流式任务更新
message TaskUpdate {
  string task_id = 1;
  AgentState state = 2;
  string message = 3;
  shannon.common.ToolCall tool_call = 4;
  shannon.common.ToolResult tool_result = 5;
  double progress = 6;
  string delta = 7; // Token 增量
}

// 智能体配置
message AgentConfig {
  int32 max_iterations = 1;
  int32 timeout_seconds = 2;
  bool enable_sandbox = 3;
  int64 memory_limit_mb = 4;
  bool enable_learning = 5;
}
```

### 4. Rust 服务端实现

**文件位置**: `rust/agent-core/src/grpc_server.rs`

```rust
use tonic::{Request, Response, Status};
use tracing::{debug, info};

pub mod proto {
    pub mod agent {
        tonic::include_proto!("shannon.agent");
    }
}

use proto::agent::agent_service_server::{AgentService, AgentServiceServer};
use proto::agent::*;

pub struct AgentServiceImpl {
    memory_pool: MemoryPool,
    llm: std::sync::Arc<LLMClient>,
    enforcer: std::sync::Arc<RequestEnforcer>,
}

#[tonic::async_trait]
impl AgentService for AgentServiceImpl {
    // 执行任务
    async fn execute_task(
        &self,
        request: Request<ExecuteTaskRequest>,
    ) -> Result<Response<ExecuteTaskResponse>, Status> {
        let req = request.into_inner();
        info!("Executing task: {}", req.query);
        
        // 执行任务逻辑
        let result = self.process_task(req).await
            .map_err(|e| Status::internal(e.to_string()))?;
        
        Ok(Response::new(result))
    }
    
    // 流式执行任务
    type StreamExecuteTaskStream = tokio_stream::wrappers::ReceiverStream<
        Result<TaskUpdate, Status>
    >;
    
    async fn stream_execute_task(
        &self,
        request: Request<ExecuteTaskRequest>,
    ) -> Result<Response<Self::StreamExecuteTaskStream>, Status> {
        let req = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(128);
        
        // 异步执行任务并流式返回进度
        tokio::spawn(async move {
            // 发送初始状态
            let _ = tx.send(Ok(TaskUpdate {
                task_id: req.metadata.unwrap().task_id,
                state: AgentState::Planning as i32,
                message: "开始规划".to_string(),
                progress: 0.1,
                ..Default::default()
            })).await;
            
            // ... 执行任务并持续发送更新
        });
        
        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }
    
    // 健康检查
    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        Ok(Response::new(HealthCheckResponse {
            healthy: true,
            message: "Agent Core is healthy".to_string(),
        }))
    }
}

// 启动 gRPC 服务器
pub async fn start_server(addr: String) -> anyhow::Result<()> {
    let agent_service = AgentServiceImpl::new()?;
    let svc = AgentServiceServer::new(agent_service);
    
    info!("Starting gRPC server on {}", addr);
    tonic::transport::Server::builder()
        .add_service(svc)
        .serve(addr.parse()?)
        .await?;
    
    Ok(())
}
```

### 5. Go 客户端实现

**文件位置**: `go/orchestrator/internal/activities/agent.go`

```go
package activities

import (
    "context"
    "io"
    "time"
    
    agentpb "github.com/Kocoro-lab/Shannon/go/orchestrator/internal/pb/agent"
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
    "go.uber.org/zap"
)

// 创建 gRPC 客户端连接
func createAgentClient(agentAddr string) (agentpb.AgentServiceClient, *grpc.ClientConn, error) {
    conn, err := grpc.Dial(
        agentAddr,
        grpc.WithTransportCredentials(insecure.NewCredentials()),
        grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(50*1024*1024)), // 50MB
    )
    if err != nil {
        return nil, nil, err
    }
    
    client := agentpb.NewAgentServiceClient(conn)
    return client, conn, nil
}

// 执行任务（一次性）
func ExecuteAgentTask(ctx context.Context, agentAddr string, query string) (*agentpb.ExecuteTaskResponse, error) {
    client, conn, err := createAgentClient(agentAddr)
    if err != nil {
        return nil, err
    }
    defer conn.Close()
    
    req := &agentpb.ExecuteTaskRequest{
        Query: query,
        Metadata: &commonpb.TaskMetadata{
            TaskId: generateTaskID(),
        },
        Config: &agentpb.AgentConfig{
            MaxIterations:   10,
            TimeoutSeconds:  300,
            EnableSandbox:   true,
            MemoryLimitMb:   512,
            EnableLearning:  true,
        },
    }
    
    resp, err := client.ExecuteTask(ctx, req)
    if err != nil {
        return nil, err
    }
    
    return resp, nil
}

// 流式执行任务
func StreamExecuteAgentTask(ctx context.Context, agentAddr string, query string, updateChan chan<- *agentpb.TaskUpdate) error {
    client, conn, err := createAgentClient(agentAddr)
    if err != nil {
        return err
    }
    defer conn.Close()
    
    req := &agentpb.ExecuteTaskRequest{
        Query: query,
        // ... 配置参数
    }
    
    stream, err := client.StreamExecuteTask(ctx, req)
    if err != nil {
        return err
    }
    
    // 接收流式更新
    for {
        update, err := stream.Recv()
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }
        
        // 发送更新到通道
        select {
        case updateChan <- update:
        case <-ctx.Done():
            return ctx.Err()
        }
    }
    
    return nil
}

// 健康检查
func CheckAgentHealth(ctx context.Context, agentAddr string) (bool, error) {
    client, conn, err := createAgentClient(agentAddr)
    if err != nil {
        return false, err
    }
    defer conn.Close()
    
    ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
    defer cancel()
    
    resp, err := client.HealthCheck(ctx, &agentpb.HealthCheckRequest{})
    if err != nil {
        return false, err
    }
    
    return resp.Healthy, nil
}
```

### 6. 编译 Proto 文件

**生成 Go 代码**:
```bash
# 位置: protos/
protoc --go_out=../go/orchestrator/internal/pb \
       --go_opt=paths=source_relative \
       --go-grpc_out=../go/orchestrator/internal/pb \
       --go-grpc_opt=paths=source_relative \
       agent/agent.proto
```

**生成 Rust 代码** (使用 `build.rs`):
```rust
// rust/agent-core/build.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile(
            &["../../protos/agent/agent.proto"],
            &["../../protos"],
        )?;
    Ok(())
}
```

---

## HTTP/REST API

### 1. 概述

Shannon Gateway 提供 RESTful HTTP API，是客户端（SDK、CLI、Web UI）的主要入口。

### 2. 架构位置

```
┌──────────┐         HTTP          ┌──────────┐
│  Client  │ ──────────────────────▶│ Gateway  │
│   SDK    │                        │  (8080)  │
└──────────┘                        └──────────┘
```

### 3. Go 服务端实现

**文件位置**: `go/orchestrator/cmd/gateway/main.go`

```go
package main

import (
    "net/http"
    "github.com/Kocoro-lab/Shannon/go/orchestrator/cmd/gateway/internal/handlers"
    "github.com/Kocoro-lab/Shannon/go/orchestrator/cmd/gateway/internal/middleware"
    "go.uber.org/zap"
)

func main() {
    logger, _ := zap.NewProduction()
    
    // 创建 HTTP 路由
    mux := http.NewServeMux()
    
    // 健康检查（无需认证）
    mux.HandleFunc("GET /health", healthHandler.Health)
    mux.HandleFunc("GET /readiness", healthHandler.Readiness)
    
    // 任务 API（需要认证）
    mux.Handle("POST /api/v1/tasks",
        authMiddleware(
            rateLimiter(
                idempotencyMiddleware(
                    http.HandlerFunc(taskHandler.Create)))))
    
    mux.Handle("GET /api/v1/tasks/{id}",
        authMiddleware(http.HandlerFunc(taskHandler.Get)))
    
    mux.Handle("GET /api/v1/tasks",
        authMiddleware(http.HandlerFunc(taskHandler.List)))
    
    // 会话 API
    mux.Handle("POST /api/v1/sessions",
        authMiddleware(http.HandlerFunc(sessionHandler.Create)))
    
    mux.Handle("POST /api/v1/sessions/{id}/submit",
        authMiddleware(http.HandlerFunc(sessionHandler.Submit)))
    
    // 流式端点（SSE/WebSocket）
    mux.Handle("GET /api/v1/stream/sse",
        authMiddleware(http.HandlerFunc(streamingProxy.ServeHTTP)))
    
    mux.Handle("GET /api/v1/stream/ws",
        authMiddleware(http.HandlerFunc(streamingProxy.ServeHTTP)))
    
    // 启动服务器
    logger.Info("Starting Gateway on :8080")
    http.ListenAndServe(":8080", mux)
}
```

**任务处理器实现**:

```go
// go/orchestrator/cmd/gateway/internal/handlers/task.go
package handlers

import (
    "encoding/json"
    "net/http"
    "github.com/jmoiron/sqlx"
    "go.uber.org/zap"
)

type TaskHandler struct {
    orchClient orchpb.OrchestratorServiceClient
    db         *sqlx.DB
    logger     *zap.Logger
}

// 创建任务
func (h *TaskHandler) Create(w http.ResponseWriter, r *http.Request) {
    // 解析请求
    var req struct {
        Query       string                 `json:"query"`
        Context     map[string]interface{} `json:"context"`
        Mode        string                 `json:"mode"`
        Tools       []string               `json:"tools"`
        MaxTokens   int                    `json:"max_tokens"`
    }
    
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request body", http.StatusBadRequest)
        return
    }
    
    // 验证必填字段
    if req.Query == "" {
        http.Error(w, "query is required", http.StatusBadRequest)
        return
    }
    
    // 调用 Orchestrator gRPC
    ctx := r.Context()
    resp, err := h.orchClient.SubmitTask(ctx, &orchpb.SubmitTaskRequest{
        Query: req.Query,
        Context: structpb.NewStruct(req.Context),
        Mode: req.Mode,
        Tools: req.Tools,
    })
    
    if err != nil {
        h.logger.Error("Failed to submit task", zap.Error(err))
        http.Error(w, "Internal server error", http.StatusInternalServerError)
        return
    }
    
    // 返回响应
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]interface{}{
        "task_id": resp.TaskId,
        "status": "submitted",
        "workflow_id": resp.WorkflowId,
    })
}

// 获取任务详情
func (h *TaskHandler) Get(w http.ResponseWriter, r *http.Request) {
    taskID := r.PathValue("id")
    if taskID == "" {
        http.Error(w, "task_id is required", http.StatusBadRequest)
        return
    }
    
    // 从数据库查询任务
    var task struct {
        ID          string    `db:"id" json:"id"`
        Query       string    `db:"query" json:"query"`
        Status      string    `db:"status" json:"status"`
        Result      string    `db:"result" json:"result"`
        CreatedAt   time.Time `db:"created_at" json:"created_at"`
        CompletedAt *time.Time `db:"completed_at" json:"completed_at"`
    }
    
    err := h.db.Get(&task, "SELECT * FROM tasks WHERE id = $1", taskID)
    if err == sql.ErrNoRows {
        http.Error(w, "Task not found", http.StatusNotFound)
        return
    }
    if err != nil {
        h.logger.Error("Database error", zap.Error(err))
        http.Error(w, "Internal server error", http.StatusInternalServerError)
        return
    }
    
    // 返回任务详情
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(task)
}

// 列出任务
func (h *TaskHandler) List(w http.ResponseWriter, r *http.Request) {
    // 解析查询参数
    limit := 50
    offset := 0
    status := r.URL.Query().Get("status")
    
    query := "SELECT * FROM tasks WHERE 1=1"
    args := []interface{}{}
    
    if status != "" {
        query += " AND status = $" + strconv.Itoa(len(args)+1)
        args = append(args, status)
    }
    
    query += " ORDER BY created_at DESC LIMIT $" + strconv.Itoa(len(args)+1)
    args = append(args, limit)
    
    query += " OFFSET $" + strconv.Itoa(len(args)+1)
    args = append(args, offset)
    
    var tasks []struct {
        ID        string    `db:"id" json:"id"`
        Query     string    `db:"query" json:"query"`
        Status    string    `db:"status" json:"status"`
        CreatedAt time.Time `db:"created_at" json:"created_at"`
    }
    
    err := h.db.Select(&tasks, query, args...)
    if err != nil {
        h.logger.Error("Database error", zap.Error(err))
        http.Error(w, "Internal server error", http.StatusInternalServerError)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]interface{}{
        "tasks": tasks,
        "total": len(tasks),
    })
}
```

### 4. Python 客户端 SDK

**文件位置**: `clients/python/src/shannon/client.py`

```python
"""Shannon SDK HTTP 客户端实现"""
import httpx
from typing import Optional, Dict, Any, Iterator
import json

class ShannonClient:
    """Shannon HTTP 客户端"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        bearer_token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        初始化客户端
        
        Args:
            base_url: Gateway 地址
            api_key: API Key (X-API-Key 头)
            bearer_token: JWT Token (Authorization: Bearer)
            timeout: 请求超时时间
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.bearer_token = bearer_token
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
    
    def _get_headers(self) -> Dict[str, str]:
        """构建请求头"""
        headers = {"Content-Type": "application/json"}
        
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        elif self.api_key:
            headers["X-API-Key"] = self.api_key
        
        return headers
    
    def create_task(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        mode: str = "auto",
        tools: Optional[list[str]] = None,
        max_tokens: int = 4000,
    ) -> Dict[str, Any]:
        """
        创建任务
        
        Args:
            query: 用户查询
            context: 上下文信息
            mode: 执行模式 (auto/research/code)
            tools: 可用工具列表
            max_tokens: 最大 token 数
        
        Returns:
            任务响应字典
        """
        payload = {
            "query": query,
            "context": context or {},
            "mode": mode,
            "tools": tools or [],
            "max_tokens": max_tokens,
        }
        
        response = self._client.post(
            f"{self.base_url}/api/v1/tasks",
            json=payload,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()
    
    def get_task(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务详情
        
        Args:
            task_id: 任务 ID
        
        Returns:
            任务详情字典
        """
        response = self._client.get(
            f"{self.base_url}/api/v1/tasks/{task_id}",
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()
    
    def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        列出任务
        
        Args:
            status: 过滤状态 (running/completed/failed)
            limit: 返回数量
            offset: 偏移量
        
        Returns:
            任务列表
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        
        response = self._client.get(
            f"{self.base_url}/api/v1/tasks",
            params=params,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()
    
    def create_session(self, name: Optional[str] = None) -> Dict[str, Any]:
        """创建会话（多轮对话）"""
        payload = {"name": name or f"Session-{int(time.time())}"}
        
        response = self._client.post(
            f"{self.base_url}/api/v1/sessions",
            json=payload,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()
    
    def submit_to_session(
        self,
        session_id: str,
        query: str,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """向会话提交查询"""
        payload = {"query": query, "stream": stream}
        
        response = self._client.post(
            f"{self.base_url}/api/v1/sessions/{session_id}/submit",
            json=payload,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()

# 使用示例
if __name__ == "__main__":
    # 初始化客户端
    client = ShannonClient(
        base_url="http://localhost:8080",
        api_key="sk_your_api_key_here"
    )
    
    # 创建任务
    task = client.create_task(
        query="分析最近一周的用户行为数据",
        context={"dataset": "user_events"},
        mode="research",
        tools=["web_search", "data_analysis"],
    )
    print(f"任务已创建: {task['task_id']}")
    
    # 查询任务状态
    status = client.get_task(task["task_id"])
    print(f"任务状态: {status['status']}")
    
    # 列出所有任务
    tasks = client.list_tasks(status="running")
    print(f"运行中的任务: {len(tasks['tasks'])} 个")
```

### 5. 认证中间件

```go
// go/orchestrator/cmd/gateway/internal/middleware/auth_validation_middleware.go
package middleware

import (
    "context"
    "net/http"
    "strings"
    "github.com/Kocoro-lab/Shannon/go/orchestrator/internal/auth"
    "go.uber.org/zap"
)

type AuthMiddleware struct {
    authService *auth.Service
    jwtManager  *auth.JWTManager
    logger      *zap.Logger
}

func (m *AuthMiddleware) Middleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // 检查是否跳过认证
        if skipAuth := os.Getenv("GATEWAY_SKIP_AUTH"); skipAuth == "1" {
            next.ServeHTTP(w, r)
            return
        }
        
        // 尝试从 Authorization 头获取 token
        authHeader := r.Header.Get("Authorization")
        if strings.HasPrefix(authHeader, "Bearer ") {
            token := strings.TrimPrefix(authHeader, "Bearer ")
            
            // 验证 JWT
            claims, err := m.jwtManager.VerifyToken(token)
            if err != nil {
                http.Error(w, "Invalid token", http.StatusUnauthorized)
                return
            }
            
            // 将用户信息添加到上下文
            ctx := context.WithValue(r.Context(), "user_id", claims.UserID)
            ctx = context.WithValue(ctx, "username", claims.Username)
            next.ServeHTTP(w, r.WithContext(ctx))
            return
        }
        
        // 尝试从 X-API-Key 头获取 API Key
        apiKey := r.Header.Get("X-API-Key")
        if apiKey != "" {
            user, err := m.authService.ValidateAPIKey(r.Context(), apiKey)
            if err != nil {
                http.Error(w, "Invalid API key", http.StatusUnauthorized)
                return
            }
            
            ctx := context.WithValue(r.Context(), "user_id", user.ID)
            ctx = context.WithValue(ctx, "username", user.Username)
            next.ServeHTTP(w, r.WithContext(ctx))
            return
        }
        
        // 未提供认证信息
        http.Error(w, "Authentication required", http.StatusUnauthorized)
    })
}
```

---

## WebSocket 协议

### 1. 概述

WebSocket 提供全双工通信，适合实时交互和双向数据流。

### 2. 架构位置

```
┌──────────┐      WebSocket       ┌──────────┐
│  Client  │ ◀───────────────────▶│ Gateway  │
│(Browser) │                      │  (WS)    │
└──────────┘                      └──────────┘
```

### 3. Go 服务端实现

**文件位置**: `go/orchestrator/internal/httpapi/websocket.go`

```go
package httpapi

import (
    "net/http"
    "strings"
    "time"
    "github.com/gorilla/websocket"
    "go.uber.org/zap"
)

// WebSocket 升级器配置
var upgrader = websocket.Upgrader{
    ReadBufferSize:  1024,
    WriteBufferSize: 1024,
    CheckOrigin: func(r *http.Request) bool {
        // 生产环境应该验证 Origin
        return true
    },
}

type StreamingHandler struct {
    mgr    *streaming.Manager
    logger *zap.Logger
}

// 注册 WebSocket 端点
func (h *StreamingHandler) RegisterWebSocket(mux *http.ServeMux) {
    mux.HandleFunc("/stream/ws", h.handleWS)
}

// 处理 WebSocket 连接
func (h *StreamingHandler) handleWS(w http.ResponseWriter, r *http.Request) {
    // 获取必需参数
    workflowID := r.URL.Query().Get("workflow_id")
    if workflowID == "" {
        http.Error(w, "workflow_id required", http.StatusBadRequest)
        return
    }
    
    // 升级到 WebSocket
    conn, err := upgrader.Upgrade(w, r, nil)
    if err != nil {
        h.logger.Error("Failed to upgrade to WebSocket", zap.Error(err))
        return
    }
    defer conn.Close()
    
    h.logger.Info("WebSocket connected",
        zap.String("workflow_id", workflowID),
        zap.String("remote_addr", r.RemoteAddr))
    
    // 解析过滤器
    typeFilter := make(map[string]struct{})
    if types := r.URL.Query().Get("types"); types != "" {
        for _, t := range strings.Split(types, ",") {
            t = strings.TrimSpace(t)
            if t != "" {
                typeFilter[t] = struct{}{}
            }
        }
    }
    
    // 解析断点续传参数
    lastEventID := r.URL.Query().Get("last_event_id")
    var lastStreamID string
    
    if strings.Contains(lastEventID, "-") {
        lastStreamID = lastEventID
    }
    
    // 重放历史事件（断点续传）
    if lastStreamID != "" {
        events := h.mgr.ReplayFromStreamID(workflowID, lastStreamID)
        for _, ev := range events {
            // 应用过滤器
            if len(typeFilter) > 0 {
                if _, ok := typeFilter[ev.Type]; !ok {
                    continue
                }
            }
            
            // 发送事件
            if err := conn.WriteJSON(ev); err != nil {
                h.logger.Error("Failed to write event", zap.Error(err))
                return
            }
        }
    }
    
    // 订阅新事件
    eventChan := h.mgr.Subscribe(workflowID)
    defer h.mgr.Unsubscribe(workflowID, eventChan)
    
    // 创建 ping ticker
    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()
    
    // 事件循环
    for {
        select {
        case event, ok := <-eventChan:
            if !ok {
                // 通道已关闭
                return
            }
            
            // 应用过滤器
            if len(typeFilter) > 0 {
                if _, ok := typeFilter[event.Type]; !ok {
                    continue
                }
            }
            
            // 发送事件
            if err := conn.WriteJSON(event); err != nil {
                h.logger.Error("Failed to write event", zap.Error(err))
                return
            }
            
        case <-ticker.C:
            // 发送 ping 保持连接
            if err := conn.WriteControl(
                websocket.PingMessage,
                []byte("ping"),
                time.Now().Add(10*time.Second),
            ); err != nil {
                h.logger.Error("Failed to send ping", zap.Error(err))
                return
            }
        }
    }
}
```

### 4. JavaScript 客户端实现

```javascript
// desktop/lib/websocket-client.ts
export class ShannonWebSocketClient {
  private ws: WebSocket | null = null;
  private workflowId: string;
  private baseUrl: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private lastEventId: string | null = null;
  
  constructor(workflowId: string, baseUrl: string = 'ws://localhost:8080') {
    this.workflowId = workflowId;
    this.baseUrl = baseUrl;
  }
  
  // 连接 WebSocket
  connect(onMessage: (event: any) => void, onError?: (error: Error) => void) {
    const url = new URL('/api/v1/stream/ws', this.baseUrl.replace('http', 'ws'));
    url.searchParams.set('workflow_id', this.workflowId);
    
    // 断点续传支持
    if (this.lastEventId) {
      url.searchParams.set('last_event_id', this.lastEventId);
    }
    
    this.ws = new WebSocket(url.toString());
    
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };
    
    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // 保存最后的事件 ID（用于断点续传）
        if (data.stream_id) {
          this.lastEventId = data.stream_id;
        }
        
        onMessage(data);
      } catch (err) {
        console.error('Failed to parse message:', err);
      }
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) {
        onError(new Error('WebSocket connection failed'));
      }
    };
    
    this.ws.onclose = () => {
      console.log('WebSocket closed');
      
      // 自动重连
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
        
        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
        setTimeout(() => {
          this.connect(onMessage, onError);
        }, delay);
      }
    };
  }
  
  // 发送消息（如果需要双向通信）
  send(message: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.error('WebSocket not connected');
    }
  }
  
  // 断开连接
  disconnect() {
    this.maxReconnectAttempts = 0; // 禁止自动重连
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

// 使用示例
const client = new ShannonWebSocketClient('workflow-123');

client.connect(
  (event) => {
    console.log('收到事件:', event);
    
    // 根据事件类型处理
    switch (event.type) {
      case 'task_started':
        console.log('任务开始:', event.message);
        break;
      case 'tool_call':
        console.log('工具调用:', event.message);
        break;
      case 'task_completed':
        console.log('任务完成:', event.message);
        break;
    }
  },
  (error) => {
    console.error('连接错误:', error);
  }
);
```

---

## Server-Sent Events (SSE)

### 1. 概述

SSE 是单向服务端推送协议，基于 HTTP，支持断点续传，适合日志流和进度更新。

### 2. Go 服务端实现

**文件位置**: `go/orchestrator/internal/httpapi/streaming.go`

```go
package httpapi

import (
    "context"
    "fmt"
    "net/http"
    "strconv"
    "strings"
    "time"
    
    "github.com/Kocoro-lab/Shannon/go/orchestrator/internal/streaming"
    "go.uber.org/zap"
)

type StreamingHandler struct {
    mgr    *streaming.Manager
    logger *zap.Logger
}

// 注册 SSE 端点
func (h *StreamingHandler) RegisterSSE(mux *http.ServeMux) {
    mux.HandleFunc("/stream/sse", h.handleSSE)
}

// 处理 SSE 请求
func (h *StreamingHandler) handleSSE(w http.ResponseWriter, r *http.Request) {
    // 验证 workflow_id
    workflowID := r.URL.Query().Get("workflow_id")
    if workflowID == "" {
        http.Error(w, "workflow_id required", http.StatusBadRequest)
        return
    }
    
    // 设置 SSE 响应头
    w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")
    w.Header().Set("X-Accel-Buffering", "no") // 禁用 nginx 缓冲
    
    // 刷新头部
    if flusher, ok := w.(http.Flusher); ok {
        flusher.Flush()
    }
    
    // 解析过滤器
    typeFilter := parseTypeFilter(r.URL.Query().Get("types"))
    
    // 解析断点续传参数（Last-Event-ID）
    lastEventID := r.Header.Get("Last-Event-ID")
    if lastEventID == "" {
        lastEventID = r.URL.Query().Get("last_event_id")
    }
    
    var lastStreamID string
    if strings.Contains(lastEventID, "-") {
        lastStreamID = lastEventID
    }
    
    // 重放历史事件
    if lastStreamID != "" {
        events := h.mgr.ReplayFromStreamID(workflowID, lastStreamID)
        for _, ev := range events {
            if !shouldSendEvent(ev, typeFilter) {
                continue
            }
            
            if err := h.sendSSEEvent(w, ev); err != nil {
                return
            }
        }
    }
    
    // 订阅新事件
    eventChan := h.mgr.Subscribe(workflowID)
    defer h.mgr.Unsubscribe(workflowID, eventChan)
    
    // 创建定时器发送心跳
    ticker := time.NewTicker(15 * time.Second)
    defer ticker.Stop()
    
    // 事件循环
    ctx := r.Context()
    for {
        select {
        case <-ctx.Done():
            // 客户端断开连接
            h.logger.Info("SSE client disconnected", zap.String("workflow_id", workflowID))
            return
            
        case event, ok := <-eventChan:
            if !ok {
                // 通道关闭，发送完成事件
                fmt.Fprintf(w, "event: done\ndata: {}\n\n")
                if flusher, ok := w.(http.Flusher); ok {
                    flusher.Flush()
                }
                return
            }
            
            // 应用过滤器
            if !shouldSendEvent(event, typeFilter) {
                continue
            }
            
            // 发送事件
            if err := h.sendSSEEvent(w, event); err != nil {
                h.logger.Error("Failed to send SSE event", zap.Error(err))
                return
            }
            
        case <-ticker.C:
            // 发送心跳（注释）
            fmt.Fprintf(w, ": heartbeat\n\n")
            if flusher, ok := w.(http.Flusher); ok {
                flusher.Flush()
            }
        }
    }
}

// 发送 SSE 事件
func (h *StreamingHandler) sendSSEEvent(w http.ResponseWriter, event *streaming.Event) error {
    // SSE 格式:
    // id: <stream_id>
    // event: <type>
    // data: <json_payload>
    //
    
    if event.StreamID != "" {
        fmt.Fprintf(w, "id: %s\n", event.StreamID)
    }
    
    if event.Type != "" {
        fmt.Fprintf(w, "event: %s\n", event.Type)
    }
    
    // 序列化数据
    data, err := json.Marshal(event)
    if err != nil {
        return err
    }
    
    fmt.Fprintf(w, "data: %s\n\n", string(data))
    
    // 立即刷新
    if flusher, ok := w.(http.Flusher); ok {
        flusher.Flush()
    }
    
    return nil
}

// 辅助函数：解析类型过滤器
func parseTypeFilter(types string) map[string]struct{} {
    filter := make(map[string]struct{})
    if types == "" {
        return filter
    }
    
    for _, t := range strings.Split(types, ",") {
        t = strings.TrimSpace(t)
        if t != "" {
            filter[t] = struct{}{}
        }
    }
    return filter
}

// 辅助函数：判断是否应该发送事件
func shouldSendEvent(event *streaming.Event, filter map[string]struct{}) bool {
    if len(filter) == 0 {
        return true
    }
    _, ok := filter[event.Type]
    return ok
}
```

### 3. JavaScript 客户端实现

```javascript
// desktop/lib/sse-client.ts
export class ShannonSSEClient {
  private eventSource: EventSource | null = null;
  private workflowId: string;
  private baseUrl: string;
  private lastEventId: string | null = null;
  
  constructor(workflowId: string, baseUrl: string = 'http://localhost:8080') {
    this.workflowId = workflowId;
    this.baseUrl = baseUrl;
  }
  
  // 连接 SSE
  connect(
    onMessage: (event: any) => void,
    onError?: (error: Error) => void,
    types?: string[]
  ) {
    const url = new URL('/api/v1/stream/sse', this.baseUrl);
    url.searchParams.set('workflow_id', this.workflowId);
    
    // 可选的事件类型过滤
    if (types && types.length > 0) {
      url.searchParams.set('types', types.join(','));
    }
    
    // 断点续传支持
    if (this.lastEventId) {
      url.searchParams.set('last_event_id', this.lastEventId);
    }
    
    this.eventSource = new EventSource(url.toString());
    
    // 监听所有事件类型
    this.eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // 保存事件 ID 用于断点续传
        if (event.lastEventId) {
          this.lastEventId = event.lastEventId;
        }
        
        onMessage(data);
      } catch (err) {
        console.error('Failed to parse SSE message:', err);
      }
    };
    
    // 监听特定事件类型
    this.eventSource.addEventListener('task_started', (event: any) => {
      const data = JSON.parse(event.data);
      console.log('任务开始:', data);
    });
    
    this.eventSource.addEventListener('tool_call', (event: any) => {
      const data = JSON.parse(event.data);
      console.log('工具调用:', data);
    });
    
    this.eventSource.addEventListener('task_completed', (event: any) => {
      const data = JSON.parse(event.data);
      console.log('任务完成:', data);
    });
    
    this.eventSource.addEventListener('done', () => {
      console.log('流结束');
      this.disconnect();
    });
    
    this.eventSource.onerror = (error) => {
      console.error('SSE error:', error);
      if (onError) {
        onError(new Error('SSE connection failed'));
      }
      
      // EventSource 会自动重连，无需手动处理
    };
  }
  
  // 断开连接
  disconnect() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }
}

// 使用示例
const client = new ShannonSSEClient('workflow-123');

client.connect(
  (event) => {
    console.log('收到事件:', event);
    
    // 更新 UI
    updateTaskProgress(event);
  },
  (error) => {
    console.error('连接错误:', error);
  },
  ['task_started', 'tool_call', 'task_completed'] // 只订阅这些事件
);

// 清理
window.addEventListener('beforeunload', () => {
  client.disconnect();
});
```

### 4. Python 客户端实现

```python
# clients/python/src/shannon/sse_client.py
import httpx
import json
from typing import Callable, Optional, List

class ShannonSSEClient:
    """Shannon SSE 客户端"""
    
    def __init__(
        self,
        workflow_id: str,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
    ):
        self.workflow_id = workflow_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.last_event_id: Optional[str] = None
    
    def stream_events(
        self,
        on_event: Callable[[dict], None],
        types: Optional[List[str]] = None,
        timeout: float = 300.0,
    ):
        """
        流式接收事件
        
        Args:
            on_event: 事件回调函数
            types: 过滤的事件类型列表
            timeout: 超时时间
        """
        url = f"{self.base_url}/api/v1/stream/sse"
        params = {"workflow_id": self.workflow_id}
        
        if types:
            params["types"] = ",".join(types)
        
        if self.last_event_id:
            params["last_event_id"] = self.last_event_id
        
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        with httpx.stream(
            "GET",
            url,
            params=params,
            headers=headers,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                line = line.strip()
                
                if not line:
                    continue
                
                # 解析 SSE 格式
                if line.startswith("id:"):
                    self.last_event_id = line[3:].strip()
                elif line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
                    try:
                        data = json.loads(data_str)
                        on_event(data)
                    except json.JSONDecodeError:
                        print(f"Failed to parse event data: {data_str}")
                elif line.startswith(":"):
                    # 心跳注释，忽略
                    pass

# 使用示例
def handle_event(event: dict):
    print(f"收到事件: {event['type']} - {event.get('message', '')}")
    
    if event['type'] == 'task_completed':
        print(f"任务完成: {event.get('result', '')}")

client = ShannonSSEClient(
    workflow_id="workflow-123",
    api_key="sk_your_api_key"
)

# 阻塞式接收事件
client.stream_events(
    on_event=handle_event,
    types=["task_started", "tool_call", "task_completed"],
    timeout=300.0
)
```

---

## Temporal Workflow

### 1. 概述

Temporal 是分布式工作流引擎，Shannon 用它编排复杂的多步骤任务，提供可靠性和状态管理。

### 2. 架构位置

```
┌──────────────┐    Temporal SDK    ┌──────────┐
│ Orchestrator │ ◀─────────────────▶│ Temporal │
│   (Worker)   │                    │  Server  │
└──────────────┘                    └──────────┘
```

### 3. Go Workflow 实现

```go
// go/orchestrator/internal/workflows/agent_workflow.go
package workflows

import (
    "time"
    "go.temporal.io/sdk/workflow"
    "github.com/Kocoro-lab/Shannon/go/orchestrator/internal/activities"
)

// 智能体工作流参数
type AgentWorkflowParams struct {
    Query       string
    Mode        string
    Tools       []string
    MaxTokens   int
    SessionID   string
}

// 智能体工作流
func AgentWorkflow(ctx workflow.Context, params AgentWorkflowParams) (string, error) {
    logger := workflow.GetLogger(ctx)
    logger.Info("Starting AgentWorkflow", "query", params.Query)
    
    // 1. 配置活动选项
    activityOptions := workflow.ActivityOptions{
        StartToCloseTimeout: 5 * time.Minute,
        HeartbeatTimeout:    30 * time.Second,
        RetryPolicy: &temporal.RetryPolicy{
            InitialInterval:    time.Second,
            BackoffCoefficient: 2.0,
            MaximumInterval:    time.Minute,
            MaximumAttempts:    3,
        },
    }
    ctx = workflow.WithActivityOptions(ctx, activityOptions)
    
    // 2. 执行智能体任务
    var agentResult activities.AgentExecutionResult
    err := workflow.ExecuteActivity(
        ctx,
        activities.ExecuteAgentTask,
        activities.AgentTaskInput{
            Query:     params.Query,
            Mode:      params.Mode,
            Tools:     params.Tools,
            MaxTokens: params.MaxTokens,
        },
    ).Get(ctx, &agentResult)
    
    if err != nil {
        logger.Error("Agent execution failed", "error", err)
        return "", err
    }
    
    // 3. 如果需要多步骤处理
    if agentResult.RequiresApproval {
        // 等待人工审批信号
        var approved bool
        signalChan := workflow.GetSignalChannel(ctx, "approval")
        signalChan.Receive(ctx, &approved)
        
        if !approved {
            logger.Info("Task rejected by user")
            return "", workflow.NewContinueAsNewError(ctx, AgentWorkflow, params)
        }
    }
    
    // 4. 存储结果
    var storeResult string
    err = workflow.ExecuteActivity(
        ctx,
        activities.StoreResult,
        activities.StoreResultInput{
            TaskID: agentResult.TaskID,
            Result: agentResult.Result,
        },
    ).Get(ctx, &storeResult)
    
    if err != nil {
        logger.Error("Failed to store result", "error", err)
        // 非致命错误，继续
    }
    
    logger.Info("AgentWorkflow completed", "result", agentResult.Result)
    return agentResult.Result, nil
}

// 启动工作流示例
func StartAgentWorkflow(
    client client.Client,
    query string,
    mode string,
) (string, error) {
    options := client.StartWorkflowOptions{
        ID:                 fmt.Sprintf("agent-%s", uuid.New().String()),
        TaskQueue:          "shannon-orchestrator",
        WorkflowRunTimeout: 30 * time.Minute,
    }
    
    params := AgentWorkflowParams{
        Query:     query,
        Mode:      mode,
        Tools:     []string{"web_search", "code_execution"},
        MaxTokens: 4000,
    }
    
    we, err := client.ExecuteWorkflow(context.Background(), options, AgentWorkflow, params)
    if err != nil {
        return "", err
    }
    
    // 获取工作流 ID
    return we.GetID(), nil
}
```

### 4. Activity 实现

```go
// go/orchestrator/internal/activities/agent.go
package activities

import (
    "context"
    "go.temporal.io/sdk/activity"
    "go.uber.org/zap"
)

type AgentTaskInput struct {
    Query     string
    Mode      string
    Tools     []string
    MaxTokens int
}

type AgentExecutionResult struct {
    TaskID           string
    Result           string
    RequiresApproval bool
    TokensUsed       int
    CostUSD          float64
}

// 执行智能体任务活动
func ExecuteAgentTask(ctx context.Context, input AgentTaskInput) (*AgentExecutionResult, error) {
    logger := activity.GetLogger(ctx)
    logger.Info("ExecuteAgentTask started", zap.String("query", input.Query))
    
    // 发送心跳
    go func() {
        ticker := time.NewTicker(10 * time.Second)
        defer ticker.Stop()
        
        for {
            select {
            case <-ctx.Done():
                return
            case <-ticker.C:
                activity.RecordHeartbeat(ctx, "processing")
            }
        }
    }()
    
    // 调用 Agent Core gRPC
    agentAddr := os.Getenv("AGENT_CORE_ADDR")
    if agentAddr == "" {
        agentAddr = "agent-core:50051"
    }
    
    conn, err := grpc.Dial(agentAddr, grpc.WithInsecure())
    if err != nil {
        return nil, err
    }
    defer conn.Close()
    
    client := agentpb.NewAgentServiceClient(conn)
    
    // 执行任务
    resp, err := client.ExecuteTask(ctx, &agentpb.ExecuteTaskRequest{
        Query: input.Query,
        Config: &agentpb.AgentConfig{
            MaxIterations:  10,
            TimeoutSeconds: 300,
            EnableSandbox:  true,
        },
    })
    
    if err != nil {
        logger.Error("Agent execution failed", zap.Error(err))
        return nil, err
    }
    
    // 检查是否需要审批
    requiresApproval := checkIfRequiresApproval(resp.Result)
    
    result := &AgentExecutionResult{
        TaskID:           resp.TaskId,
        Result:           resp.Result,
        RequiresApproval: requiresApproval,
        TokensUsed:       int(resp.Metrics.TotalTokens),
        CostUSD:          resp.Metrics.TotalCostUsd,
    }
    
    logger.Info("ExecuteAgentTask completed",
        zap.String("task_id", result.TaskID),
        zap.Int("tokens", result.TokensUsed))
    
    return result, nil
}

// 存储结果活动
func StoreResult(ctx context.Context, input StoreResultInput) (string, error) {
    // 存储到数据库
    // ...
    return "stored", nil
}
```

### 5. Python SDK 集成

```python
# python/llm-service/llm_service/temporal_client.py
from temporalio.client import Client
from temporalio import workflow
from datetime import timedelta

class AgentWorkflowParams:
    query: str
    mode: str
    tools: list[str]
    max_tokens: int

async def start_agent_workflow(
    query: str,
    mode: str = "auto",
) -> str:
    """启动智能体工作流"""
    
    # 连接 Temporal
    client = await Client.connect("temporal:7233")
    
    params = AgentWorkflowParams()
    params.query = query
    params.mode = mode
    params.tools = ["web_search", "code_execution"]
    params.max_tokens = 4000
    
    # 启动工作流
    handle = await client.start_workflow(
        "AgentWorkflow",
        params,
        id=f"agent-{uuid.uuid4()}",
        task_queue="shannon-orchestrator",
        execution_timeout=timedelta(minutes=30),
    )
    
    print(f"Started workflow: {handle.id}")
    
    # 等待结果（异步）
    result = await handle.result()
    
    return result

# 使用示例
import asyncio

async def main():
    result = await start_agent_workflow(
        query="分析用户行为数据",
        mode="research"
    )
    print(f"工作流结果: {result}")

asyncio.run(main())
```

---

## Redis Pub/Sub

### 1. 概述

Redis Pub/Sub 用于跨服务事件通知和实时消息传递。

### 2. 架构位置

```
┌──────────────┐      Publish      ┌───────┐
│ Orchestrator │ ─────────────────▶│ Redis │
└──────────────┘                   └───────┘
                                       │
                                       │ Subscribe
                                       ▼
                              ┌─────────────────┐
                              │  Event Listeners │
                              └─────────────────┘
```

### 3. Go 发布端实现

```go
// go/orchestrator/internal/streaming/publisher.go
package streaming

import (
    "context"
    "encoding/json"
    "github.com/redis/go-redis/v9"
    "go.uber.org/zap"
)

type EventPublisher struct {
    redis  *redis.Client
    logger *zap.Logger
}

type Event struct {
    WorkflowID string    `json:"workflow_id"`
    Type       string    `json:"type"`
    AgentID    string    `json:"agent_id"`
    Message    string    `json:"message"`
    Timestamp  time.Time `json:"timestamp"`
    StreamID   string    `json:"stream_id"`
}

// 发布事件
func (p *EventPublisher) Publish(ctx context.Context, event *Event) error {
    // 序列化事件
    data, err := json.Marshal(event)
    if err != nil {
        return err
    }
    
    // 发布到 Redis 频道
    channel := fmt.Sprintf("workflow:%s", event.WorkflowID)
    err = p.redis.Publish(ctx, channel, data).Err()
    if err != nil {
        p.logger.Error("Failed to publish event",
            zap.String("channel", channel),
            zap.Error(err))
        return err
    }
    
    // 同时存储到 Stream（用于历史回放）
    streamKey := fmt.Sprintf("stream:%s", event.WorkflowID)
    _, err = p.redis.XAdd(ctx, &redis.XAddArgs{
        Stream: streamKey,
        Values: map[string]interface{}{
            "type":    event.Type,
            "message": event.Message,
            "data":    string(data),
        },
    }).Result()
    
    if err != nil {
        p.logger.Error("Failed to add to stream", zap.Error(err))
    }
    
    return nil
}

// 发布任务开始事件
func (p *EventPublisher) PublishTaskStarted(workflowID, agentID, message string) error {
    return p.Publish(context.Background(), &Event{
        WorkflowID: workflowID,
        Type:       "task_started",
        AgentID:    agentID,
        Message:    message,
        Timestamp:  time.Now(),
    })
}
```

### 4. Go 订阅端实现

```go
// go/orchestrator/internal/streaming/subscriber.go
package streaming

import (
    "context"
    "encoding/json"
    "github.com/redis/go-redis/v9"
    "go.uber.org/zap"
)

type EventSubscriber struct {
    redis  *redis.Client
    logger *zap.Logger
}

// 订阅工作流事件
func (s *EventSubscriber) Subscribe(
    ctx context.Context,
    workflowID string,
    handler func(*Event),
) error {
    channel := fmt.Sprintf("workflow:%s", workflowID)
    
    pubsub := s.redis.Subscribe(ctx, channel)
    defer pubsub.Close()
    
    s.logger.Info("Subscribed to channel", zap.String("channel", channel))
    
    // 接收消息
    ch := pubsub.Channel()
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
            
        case msg, ok := <-ch:
            if !ok {
                return nil
            }
            
            // 解析事件
            var event Event
            if err := json.Unmarshal([]byte(msg.Payload), &event); err != nil {
                s.logger.Error("Failed to parse event", zap.Error(err))
                continue
            }
            
            // 调用处理器
            handler(&event)
        }
    }
}

// 使用示例
func main() {
    subscriber := NewEventSubscriber(redisClient, logger)
    
    ctx := context.Background()
    err := subscriber.Subscribe(ctx, "workflow-123", func(event *Event) {
        fmt.Printf("收到事件: %s - %s\n", event.Type, event.Message)
        
        // 根据事件类型处理
        switch event.Type {
        case "task_started":
            fmt.Println("任务开始")
        case "tool_call":
            fmt.Println("工具调用")
        case "task_completed":
            fmt.Println("任务完成")
        }
    })
    
    if err != nil {
        log.Fatal(err)
    }
}
```

### 5. Python 订阅实现

```python
# python/llm-service/llm_service/redis_subscriber.py
import redis
import json
from typing import Callable

class EventSubscriber:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.pubsub = self.redis.pubsub()
    
    def subscribe(
        self,
        workflow_id: str,
        on_event: Callable[[dict], None],
    ):
        """订阅工作流事件"""
        channel = f"workflow:{workflow_id}"
        self.pubsub.subscribe(channel)
        
        print(f"Subscribed to {channel}")
        
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                try:
                    event = json.loads(message['data'])
                    on_event(event)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse event: {e}")
    
    def close(self):
        self.pubsub.close()
        self.redis.close()

# 使用示例
def handle_event(event: dict):
    print(f"Event: {event['type']} - {event['message']}")

subscriber = EventSubscriber()
subscriber.subscribe("workflow-123", handle_event)
```

---

## 协议选择指南

### 1. 场景对比

| 场景 | 推荐协议 | 理由 |
|------|---------|------|
| 微服务间RPC调用 | gRPC | 高性能、类型安全、双向流 |
| 公共 API 访问 | HTTP/REST | 通用、易用、防火墙友好 |
| 实时任务更新（单向） | SSE | 简单、支持断点续传 |
| 实时任务更新（双向） | WebSocket | 低延迟、支持双向通信 |
| 复杂工作流编排 | Temporal | 可靠性、状态管理、重试 |
| 跨服务事件通知 | Redis Pub/Sub | 解耦、高性能 |

### 2. 性能对比

| 协议 | 延迟 | 吞吐量 | 资源消耗 | 复杂度 |
|------|------|--------|----------|--------|
| gRPC | 极低 (< 1ms) | 极高 | 中等 | 高 |
| HTTP/REST | 低 (1-10ms) | 高 | 低 | 低 |
| WebSocket | 极低 (< 1ms) | 高 | 高（长连接） | 中 |
| SSE | 低 (1-5ms) | 中 | 中（长连接） | 低 |
| Temporal | 中 (10-100ms) | 中 | 高 | 高 |
| Redis Pub/Sub | 极低 (< 1ms) | 极高 | 低 | 低 |

### 3. 决策树

```
需要双向通信？
├─ 是 → 使用 WebSocket 或 gRPC 双向流
└─ 否 → 需要实时推送？
    ├─ 是 → 使用 SSE 或 Redis Pub/Sub
    └─ 否 → 需要高性能 RPC？
        ├─ 是 → 使用 gRPC
        └─ 否 → 需要工作流编排？
            ├─ 是 → 使用 Temporal
            └─ 否 → 使用 HTTP/REST
```

---

## 附录

### A. 完整代码示例仓库

- **Proto 定义**: `protos/`
- **Go 实现**: `go/orchestrator/`
- **Rust 实现**: `rust/agent-core/`
- **Python SDK**: `clients/python/`
- **Desktop App**: `desktop/`

### B. 环境变量配置

```bash
# gRPC
AGENT_CORE_ADDR=agent-core:50051
ORCHESTRATOR_GRPC=orchestrator:50052

# HTTP
GATEWAY_PORT=8080
GATEWAY_SKIP_AUTH=0

# WebSocket/SSE
ADMIN_SERVER=http://orchestrator:8081

# Temporal
TEMPORAL_HOST=temporal:7233
TEMPORAL_NAMESPACE=default

# Redis
REDIS_URL=redis://redis:6379

# OpenTelemetry
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=shannon-gateway
```

### C. 调试工具

- **gRPC**: grpcurl, BloomRPC
- **HTTP**: curl, Postman
- **WebSocket**: websocat, Browser DevTools
- **SSE**: curl, EventSource (Browser)
- **Temporal**: Temporal Web UI (port 8088)
- **Redis**: redis-cli, RedisInsight

### D. 常见问题

**Q: gRPC 报 "connection refused"？**
A: 检查服务端口和防火墙配置，确保 50051/50052 端口开放。

**Q: SSE 断开后如何续传？**
A: 使用 `Last-Event-ID` 头或 `last_event_id` 参数，传入上次的 `stream_id`。

**Q: WebSocket 如何处理重连？**
A: 实现指数退避重连策略，保存 `last_event_id` 用于断点续传。

**Q: Temporal Worker 如何扩展？**
A: 增加 Worker 实例数量，所有 Worker 共享同一个 TaskQueue。

---

**文档维护**: Shannon Team  
**最后更新**: 2026年1月30日  
**版本**: 1.0  
