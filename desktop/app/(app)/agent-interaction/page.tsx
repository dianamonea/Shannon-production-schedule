"use client";

import React, { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
    Activity, 
    Zap, 
    Package, 
    Cpu, 
    RefreshCw,
    Download,
    AlertCircle
} from "lucide-react";

interface Agent {
    id: string;
    name: string;
    type: "machine" | "agv" | "robot";
    icon: React.ReactNode;
    color: string;
    status: "active" | "idle" | "processing";
}

interface Interaction {
    id: number;
    from: string;
    to: string;
    type: string;
    timestamp: string;
    message: string;
    disturbance?: string;
    response?: string;
}

interface DisturbanceEvent {
    type: string;
    severity: "low" | "medium" | "high" | "critical";
    description: string;
    affectedResource: string;
    impactDuration: number;
}

const AGENTS: Agent[] = [
    {
        id: "machine",
        name: "机床智能体",
        type: "machine",
        icon: <Cpu className="h-6 w-6" />,
        color: "from-orange-400 to-red-500",
        status: "active"
    },
    {
        id: "agv",
        name: "AGV智能体",
        type: "agv",
        icon: <Package className="h-6 w-6" />,
        color: "from-blue-400 to-cyan-500",
        status: "active"
    },
    {
        id: "robot",
        name: "机器人智能体",
        type: "robot",
        icon: <Zap className="h-6 w-6" />,
        color: "from-purple-400 to-pink-500",
        status: "active"
    }
];

export default function AgentInteractionVisualization() {
    const [interactions, setInteractions] = useState<Interaction[]>([]);
    const [disturbances, setDisturbances] = useState<DisturbanceEvent[]>([]);
    const [sessionData, setSessionData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [autoRefresh, setAutoRefresh] = useState(false);
    const [selectedAgent, setSelectedAgent] = useState<string | null>(null);

    useEffect(() => {
        fetchLatestSession();
    }, []);

    useEffect(() => {
        if (autoRefresh) {
            const interval = setInterval(fetchLatestSession, 2000);
            return () => clearInterval(interval);
        }
    }, [autoRefresh]);

    const fetchLatestSession = async () => {
        setLoading(true);
        setError(null);
        try {
            // 尝试从本地查询最新的调度结果文件
            const response = await fetch("/api/latest-schedule");
            if (!response.ok) throw new Error("Failed to fetch");
            
            const data = await response.json();
            
            if (data.disturbances_detected) {
                setDisturbances(data.disturbances_detected);
            }
            
            if (data.disturbance_responses) {
                const newInteractions = data.disturbance_responses.map((item: any, idx: number) => ({
                    id: idx,
                    from: item.agent.includes("机床") ? "machine" : item.agent.includes("AGV") ? "agv" : "robot",
                    to: "coordinator",
                    type: item.disturbance_type,
                    timestamp: item.timestamp,
                    message: `处理 ${item.disturbance_type}`,
                    disturbance: item.description,
                    response: item.response
                }));
                setInteractions(newInteractions);
            }
            
            if (data.coordination_timeline) {
                // 从协调日志中解析额外的交互
                const extraInteractions = parseCoordinationTimeline(data.coordination_timeline);
                setInteractions(prev => [...extraInteractions, ...prev]);
            }
            
            setSessionData(data);
            setLoading(false);
        } catch (error) {
            console.error("Failed to fetch session data:", error);
            setError("无法加载数据，使用演示数据");
            // 使用模拟数据进行演示
            loadMockData();
        }
    };

    const parseCoordinationTimeline = (timeline: any[]) => {
        const interactions: Interaction[] = [];
        timeline.forEach((log, idx) => {
            const agent = log.agent.includes("机床") ? "machine" : log.agent.includes("AGV") ? "agv" : "robot";
            interactions.push({
                id: idx,
                from: agent,
                to: "coordinator",
                type: "coordination",
                timestamp: log.timestamp,
                message: log.message
            });
        });
        return interactions;
    };

    const loadMockData = () => {
        const mockInteractions: Interaction[] = [
            {
                id: 1,
                from: "machine",
                to: "coordinator",
                type: "urgent_order",
                timestamp: "2026-01-29T14:27:43.885363",
                message: "检测到紧急订单",
                disturbance: "新增紧急订单 PART-URGENT，4小时内完成",
                response: "将紧急订单插入队列首位，重新调整排产序列"
            },
            {
                id: 2,
                from: "machine",
                to: "coordinator",
                type: "quality_issue",
                timestamp: "2026-01-29T14:27:43.885991",
                message: "检测到质量问题",
                disturbance: "PART-003 检测发现尺寸偏差，需返工",
                response: "为 PART-003 预留返工时间 45 分钟"
            },
            {
                id: 3,
                from: "agv",
                to: "coordinator",
                type: "agv_breakdown",
                timestamp: "2026-01-29T14:27:43.947055",
                message: "检测到AGV故障",
                disturbance: "AGV-01 导航系统故障，无法正常运输",
                response: "将 AGV-01 的任务分配给其他AGV，启用备用车辆"
            },
            {
                id: 4,
                from: "robot",
                to: "coordinator",
                type: "operator_shortage",
                timestamp: "2026-01-29T14:27:43.867233",
                message: "检测到人员短缺",
                disturbance: "夜班操作员请假，人手减少1人",
                response: "切换到全自动上下料模式，减少人工干预"
            }
        ];

        const mockDisturbances: DisturbanceEvent[] = [
            {
                type: "urgent_order",
                severity: "critical",
                description: "新增紧急订单 PART-URGENT，4小时内完成",
                affectedResource: "new_urgent_part",
                impactDuration: 0
            },
            {
                type: "quality_issue",
                severity: "medium",
                description: "PART-003 检测发现尺寸偏差，需返工",
                affectedResource: "PART-003",
                impactDuration: 45
            },
            {
                type: "machine_failure",
                severity: "high",
                description: "CNC-2 主轴轴承过热，需紧急维护",
                affectedResource: "cnc_2",
                impactDuration: 120
            },
            {
                type: "agv_breakdown",
                severity: "high",
                description: "AGV-01 导航系统故障，无法正常运输",
                affectedResource: "AGV-01",
                impactDuration: 90
            },
            {
                type: "operator_shortage",
                severity: "medium",
                description: "夜班操作员请假，人手减少1人",
                affectedResource: "operator_team",
                impactDuration: 480
            }
        ];

        setInteractions(mockInteractions);
        setDisturbances(mockDisturbances);
        setLoading(false);
    };

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case "critical":
                return "bg-red-100 text-red-800 border-red-300";
            case "high":
                return "bg-orange-100 text-orange-800 border-orange-300";
            case "medium":
                return "bg-yellow-100 text-yellow-800 border-yellow-300";
            case "low":
                return "bg-green-100 text-green-800 border-green-300";
            default:
                return "bg-gray-100 text-gray-800 border-gray-300";
        }
    };

    const getAgentColor = (agentId: string) => {
        return AGENTS.find(a => a.id === agentId)?.color || "from-gray-400 to-gray-500";
    };

    return (
        <div className="h-full overflow-y-auto bg-gradient-to-br from-blue-50/30 via-white to-purple-50/30 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
            <div className="p-4 sm:p-8 space-y-6 sm:space-y-8 max-w-7xl mx-auto">
                {/* 头部 */}
                <div className="space-y-2">
                    <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">多智能体交互可视化</h1>
                    <p className="text-muted-foreground">
                        实时监控生产调度中的三类具身智能体交互、扰动检测和应对过程
                    </p>
                </div>

            {/* 控制面板 */}
            <div className="flex flex-wrap gap-3">
                <Button 
                    onClick={fetchLatestSession}
                    variant="outline"
                    size="sm"
                >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    刷新数据
                </Button>
                <Button 
                    onClick={() => setAutoRefresh(!autoRefresh)}
                    variant={autoRefresh ? "default" : "outline"}
                    size="sm"
                >
                    <Activity className="h-4 w-4 mr-2" />
                    {autoRefresh ? "自动刷新中..." : "启用自动刷新"}
                </Button>
            </div>

            {loading && !disturbances.length ? (
                <div className="flex flex-col items-center justify-center py-16 min-h-[50vh]">
                    <div className="relative w-12 h-12 mb-4">
                        <div className="absolute inset-0 rounded-full border-4 border-primary/20" />
                        <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-primary animate-spin" />
                    </div>
                    <p className="text-lg font-medium text-foreground mb-2">加载数据中...</p>
                    <p className="text-sm text-muted-foreground">正在获取最新的生产调度数据</p>
                </div>
            ) : disturbances.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-16 min-h-[50vh] bg-white/50 dark:bg-slate-800/50 rounded-lg border border-dashed border-border">
                    <AlertCircle className="h-12 w-12 text-muted-foreground mb-4 opacity-50" />
                    <p className="text-lg font-medium text-foreground mb-2">暂无生产扰动数据</p>
                    <p className="text-sm text-muted-foreground mb-4">当前系统运行正常，未检测到异常</p>
                    <Button onClick={fetchLatestSession} variant="outline" size="sm">
                        <RefreshCw className="h-4 w-4 mr-2" />
                        刷新数据
                    </Button>
                </div>
            ) : (
                <>
                    {/* 智能体状态 */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {AGENTS.map((agent) => (
                            <Card 
                                key={agent.id}
                                className={`cursor-pointer transition-all hover:shadow-xl hover:scale-105 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm ${
                                    selectedAgent === agent.id ? "ring-2 ring-primary shadow-2xl scale-105" : ""
                                }`}
                                onClick={() => {
                                    setSelectedAgent(selectedAgent === agent.id ? null : agent.id);
                                    console.log(`已选中智能体: ${agent.name}`);
                                }}
                            >
                                <CardHeader className="pb-3">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-3">
                                            <div className={`p-3 rounded-lg bg-gradient-to-r ${agent.color} text-white`}>
                                                {agent.icon}
                                            </div>
                                            <div>
                                                <CardTitle className="text-lg">{agent.name}</CardTitle>
                                                <CardDescription className="text-xs">
                                                    {agent.type === "machine" && "CNC-1, CNC-2, CNC-3"}
                                                    {agent.type === "agv" && "AGV-01, AGV-02"}
                                                    {agent.type === "robot" && "ROBOT-01, ROBOT-02"}
                                                </CardDescription>
                                            </div>
                                        </div>
                                        <Badge variant={agent.status === "active" ? "default" : "secondary"}>
                                            {agent.status === "active" ? "运行中" : "待命"}
                                        </Badge>
                                    </div>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-2 text-sm">
                                        <div>
                                            <span className="text-muted-foreground">处理扰动数:</span>
                                            <span className="font-semibold ml-2">
                                                {interactions.filter(i => 
                                                    i.from === agent.id
                                                ).length}
                                            </span>
                                        </div>
                                        <div>
                                            <span className="text-muted-foreground">利用率:</span>
                                            <span className="font-semibold ml-2">
                                                {agent.type === "machine" ? "95%" : agent.type === "agv" ? "85%" : "90%"}
                                            </span>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>

                    {/* 扰动总览 */}
                    <Card className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm shadow-lg">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <AlertCircle className="h-5 w-5 text-orange-500" />
                                检测到的生产扰动 ({disturbances.length})
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
                                {disturbances.map((disturbance, idx) => (
                                    <div 
                                        key={idx}
                                        className={`p-4 rounded-lg border-l-4 cursor-pointer transition-all hover:shadow-md hover:translate-x-1 ${getSeverityColor(disturbance.severity)}`}
                                        onClick={() => console.log('点击扰动:', disturbance)}
                                    >
                                        <div className="flex justify-between items-start">
                                            <div>
                                                <h4 className="font-semibold capitalize">{disturbance.type}</h4>
                                                <p className="text-sm mt-1">{disturbance.description}</p>
                                                <div className="text-xs mt-2 space-x-3">
                                                    <span>资源: {disturbance.affectedResource}</span>
                                                    <span>影响时长: {disturbance.impactDuration}分钟</span>
                                                </div>
                                            </div>
                                            <Badge className="ml-2">
                                                {disturbance.severity}
                                            </Badge>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>

                    {/* 智能体交互流 */}
                    <Card className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm shadow-lg">
                        <CardHeader>
                            <CardTitle>智能体交互流程</CardTitle>
                            <CardDescription>
                                显示三类具身智能体检测扰动和执行应对策略的过程
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
                                {interactions.map((interaction, idx) => {
                                    const fromAgent = AGENTS.find(a => a.id === interaction.from);
                                    return (
                                        <div 
                                            key={interaction.id} 
                                            className="border rounded-lg p-4 bg-gradient-to-r from-white to-blue-50/30 dark:from-slate-800 dark:to-slate-700 hover:shadow-lg hover:scale-[1.02] transition-all cursor-pointer"
                                            onClick={() => console.log('查看交互详情:', interaction)}
                                        >
                                            <div className="flex items-start gap-4">
                                                {/* 时间线 */}
                                                <div className="flex flex-col items-center">
                                                    <div className={`w-3 h-3 rounded-full bg-gradient-to-r ${fromAgent?.color || 'from-gray-400 to-gray-500'}`} />
                                                    {idx < interactions.length - 1 && (
                                                        <div className="w-0.5 h-12 bg-gradient-to-b from-gray-300 to-transparent mt-1" />
                                                    )}
                                                </div>

                                                {/* 内容 */}
                                                <div className="flex-1 min-w-0">
                                                    <div className="flex items-center gap-2 flex-wrap">
                                                        <Badge variant="outline" className="font-semibold">
                                                            {fromAgent?.name}
                                                        </Badge>
                                                        <span className="text-muted-foreground text-sm">
                                                            {new Date(interaction.timestamp).toLocaleTimeString('zh-CN')}
                                                        </span>
                                                    </div>

                                                    {/* 扰动信息 */}
                                                    {interaction.disturbance && (
                                                        <div className="mt-3 p-3 bg-orange-50/80 dark:bg-orange-900/30 rounded border border-orange-200 dark:border-orange-700">
                                                            <p className="text-sm font-medium text-orange-900 dark:text-orange-200">
                                                                ⚠️ {interaction.disturbance}
                                                            </p>
                                                        </div>
                                                    )}

                                                    {/* 应对策略 */}
                                                    {interaction.response && (
                                                        <div className="mt-2 p-3 bg-emerald-50/80 dark:bg-emerald-900/30 rounded border border-emerald-200 dark:border-emerald-700">
                                                            <p className="text-sm font-medium text-emerald-900 dark:text-emerald-200">
                                                                ✅ 应对策略: {interaction.response}
                                                            </p>
                                                        </div>
                                                    )}

                                                    {/* 消息 */}
                                                    {!interaction.disturbance && (
                                                        <p className="text-sm text-muted-foreground mt-2">
                                                            {interaction.message}
                                                        </p>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </CardContent>
                    </Card>

                    {/* 协同统计 */}
                    {sessionData && (
                        <Card className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm shadow-lg">
                            <CardHeader>
                                <CardTitle>协同执行统计</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <div className="p-3 bg-orange-50/80 dark:bg-orange-900/30 rounded-lg hover:shadow-md transition-all cursor-pointer" onClick={() => console.log('机床统计')}>
                                        <p className="text-xs text-muted-foreground">机床处理扰动</p>
                                        <p className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                                            {interactions.filter(i => i.from === "machine").length}
                                        </p>
                                    </div>
                                    <div className="p-3 bg-blue-50/80 dark:bg-blue-900/30 rounded-lg hover:shadow-md transition-all cursor-pointer" onClick={() => console.log('AGV统计')}>
                                        <p className="text-xs text-muted-foreground">AGV处理扰动</p>
                                        <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                                            {interactions.filter(i => i.from === "agv").length}
                                        </p>
                                    </div>
                                    <div className="p-3 bg-purple-50/80 dark:bg-purple-900/30 rounded-lg hover:shadow-md transition-all cursor-pointer" onClick={() => console.log('机器人统计')}>
                                        <p className="text-xs text-muted-foreground">机器人处理扰动</p>
                                        <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                                            {interactions.filter(i => i.from === "robot").length}
                                        </p>
                                    </div>
                                    <div className="p-3 bg-red-50/80 dark:bg-red-900/30 rounded-lg hover:shadow-md transition-all cursor-pointer" onClick={() => console.log('总扰动统计')}>
                                        <p className="text-xs text-muted-foreground">总扰动数</p>
                                        <p className="text-2xl font-bold text-red-600 dark:text-red-400">
                                            {disturbances.length}
                                        </p>
                                    </div>
                                </div>

                                {sessionData.execution_summary && (
                                    <div className="mt-6 space-y-2 text-sm">
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">会话ID:</span>
                                            <code className="text-xs bg-muted px-2 py-1 rounded">
                                                {sessionData.session_id}
                                            </code>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">执行时间:</span>
                                            <span className="font-semibold">
                                                {sessionData.execution_summary.total_execution_time?.toFixed(2)}秒
                                            </span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">协同过程步数:</span>
                                            <span className="font-semibold">
                                                {sessionData.coordination_timeline?.length || 0}步
                                            </span>
                                        </div>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    )}
                </>
            )}
            </div>
        </div>
    );
}
