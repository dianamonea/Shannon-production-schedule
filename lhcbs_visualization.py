"""
L-HCBS Visualization Generator

Creates an interactive HTML visualization of the L-HCBS algorithm.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from python.shannon.mapf import (
    AgentType, HeterogeneousAgentFactory,
    GridMap, HeterogeneousCBS, LearningGuidedHCBS,
    BenchmarkGenerator, MetricsCalculator
)


def generate_visualization_data():
    """Generate data for visualization."""
    factory = HeterogeneousAgentFactory()
    
    # Scenario 1: Simple two-agent crossing
    print("Generating Scenario 1: Two AGVs crossing...")
    grid1 = GridMap(15, 15)
    agents1 = {
        'agv_1': factory.create_agent('agv_1', AgentType.AGV, (0.0, 0.0)),
        'agv_2': factory.create_agent('agv_2', AgentType.AGV, (14.0, 14.0))
    }
    starts1 = {'agv_1': (0, 0), 'agv_2': (14, 14)}
    goals1 = {'agv_1': (14, 14), 'agv_2': (0, 0)}
    
    solver1 = HeterogeneousCBS(grid1, agents1)
    paths1 = solver1.solve(starts1, goals1)
    
    scenario1 = {
        'name': 'Two AGVs Crossing',
        'grid': {'width': 15, 'height': 15, 'obstacles': []},
        'agents': [
            {'id': 'agv_1', 'type': 'AGV', 'color': '#4CAF50', 
             'start': starts1['agv_1'], 'goal': goals1['agv_1']},
            {'id': 'agv_2', 'type': 'AGV', 'color': '#2196F3',
             'start': starts1['agv_2'], 'goal': goals1['agv_2']}
        ],
        'paths': {aid: [list(p) for p in path] for aid, path in paths1.items()} if paths1 else {}
    }
    
    # Scenario 2: Bottleneck with wall
    print("Generating Scenario 2: Bottleneck scenario...")
    obstacles2 = set()
    for x in range(20):
        if x != 10:  # Gap at x=10
            obstacles2.add((x, 10))
    
    grid2 = GridMap(20, 20, obstacles2)
    agents2 = {
        'agv_1': factory.create_agent('agv_1', AgentType.AGV, (2.0, 5.0)),
        'agv_2': factory.create_agent('agv_2', AgentType.AGV, (17.0, 5.0)),
        'agv_3': factory.create_agent('agv_3', AgentType.AGV, (2.0, 15.0)),
    }
    starts2 = {'agv_1': (2, 5), 'agv_2': (17, 5), 'agv_3': (2, 15)}
    goals2 = {'agv_1': (17, 15), 'agv_2': (2, 15), 'agv_3': (17, 5)}
    
    solver2 = LearningGuidedHCBS(grid2, agents2, use_learning=True)
    paths2 = solver2.solve(starts2, goals2, max_iterations=5000)
    
    scenario2 = {
        'name': 'Bottleneck Navigation',
        'grid': {'width': 20, 'height': 20, 'obstacles': [list(o) for o in obstacles2]},
        'agents': [
            {'id': 'agv_1', 'type': 'AGV', 'color': '#4CAF50',
             'start': starts2['agv_1'], 'goal': goals2['agv_1']},
            {'id': 'agv_2', 'type': 'AGV', 'color': '#2196F3',
             'start': starts2['agv_2'], 'goal': goals2['agv_2']},
            {'id': 'agv_3', 'type': 'AGV', 'color': '#FF9800',
             'start': starts2['agv_3'], 'goal': goals2['agv_3']}
        ],
        'paths': {aid: [list(p) for p in path] for aid, path in paths2.items()} if paths2 else {}
    }
    
    # Scenario 3: Manufacturing floor
    print("Generating Scenario 3: Manufacturing floor...")
    obstacles3 = set()
    # Add machine blocks
    for mx, my in [(4, 4), (4, 10), (10, 4), (10, 10)]:
        for dx in range(2):
            for dy in range(2):
                obstacles3.add((mx + dx, my + dy))
    
    grid3 = GridMap(16, 16, obstacles3)
    agents3 = {
        'cnc_1': factory.create_agent('cnc_1', AgentType.CNC, (3.0, 4.0)),
        'cnc_2': factory.create_agent('cnc_2', AgentType.CNC, (3.0, 10.0)),
        'agv_1': factory.create_agent('agv_1', AgentType.AGV, (0.0, 0.0)),
        'agv_2': factory.create_agent('agv_2', AgentType.AGV, (15.0, 0.0)),
        'robot_1': factory.create_agent('robot_1', AgentType.ROBOT, (7.0, 7.0)),
    }
    
    # Only mobile agents need path planning
    mobile_agents3 = {k: v for k, v in agents3.items() if v.can_move()}
    starts3 = {
        'agv_1': (0, 0), 
        'agv_2': (15, 0),
        'robot_1': (7, 7)
    }
    goals3 = {
        'agv_1': (15, 15),
        'agv_2': (0, 15),
        'robot_1': (7, 7)  # Robot stays in place
    }
    
    solver3 = HeterogeneousCBS(grid3, mobile_agents3)
    paths3 = solver3.solve(starts3, goals3)
    
    # Add static CNC positions
    if paths3:
        paths3['cnc_1'] = [(3, 4)]
        paths3['cnc_2'] = [(3, 10)]
    
    scenario3 = {
        'name': 'Manufacturing Floor',
        'grid': {'width': 16, 'height': 16, 'obstacles': [list(o) for o in obstacles3]},
        'agents': [
            {'id': 'cnc_1', 'type': 'CNC', 'color': '#9E9E9E',
             'start': [3, 4], 'goal': [3, 4]},
            {'id': 'cnc_2', 'type': 'CNC', 'color': '#9E9E9E',
             'start': [3, 10], 'goal': [3, 10]},
            {'id': 'agv_1', 'type': 'AGV', 'color': '#4CAF50',
             'start': starts3['agv_1'], 'goal': goals3['agv_1']},
            {'id': 'agv_2', 'type': 'AGV', 'color': '#2196F3',
             'start': starts3['agv_2'], 'goal': goals3['agv_2']},
            {'id': 'robot_1', 'type': 'ROBOT', 'color': '#E91E63',
             'start': starts3['robot_1'], 'goal': goals3['robot_1']}
        ],
        'paths': {aid: [list(p) for p in path] for aid, path in paths3.items()} if paths3 else {}
    }
    
    # Algorithm comparison data
    print("Running algorithm comparison...")
    comparison_data = run_comparison()
    
    return {
        'scenarios': [scenario1, scenario2, scenario3],
        'comparison': comparison_data
    }


def run_comparison():
    """Run algorithm comparison for visualization."""
    factory = HeterogeneousAgentFactory()
    results = []
    
    # Test cases of increasing difficulty
    test_cases = [
        {'agents': 2, 'grid': 10},
        {'agents': 3, 'grid': 15},
        {'agents': 4, 'grid': 15},
        {'agents': 5, 'grid': 20},
    ]
    
    for tc in test_cases:
        n = tc['agents']
        size = tc['grid']
        
        grid = GridMap(size, size)
        agents = {}
        starts = {}
        goals = {}
        
        for i in range(n):
            aid = f'agv_{i}'
            # Spread agents around the grid
            if i % 2 == 0:
                sx, sy = i % size, 0
                gx, gy = size - 1 - (i % size), size - 1
            else:
                sx, sy = size - 1, i % size
                gx, gy = 0, size - 1 - (i % size)
            
            agents[aid] = factory.create_agent(aid, AgentType.AGV, (float(sx), float(sy)))
            starts[aid] = (sx, sy)
            goals[aid] = (gx, gy)
        
        # Test each algorithm
        case_results = {'agents': n, 'grid_size': size, 'algorithms': []}
        
        # H-CBS
        solver = HeterogeneousCBS(grid, agents)
        start = time.time()
        paths = solver.solve(starts, goals, max_iterations=2000)
        elapsed = time.time() - start
        case_results['algorithms'].append({
            'name': 'H-CBS',
            'success': paths is not None,
            'time': round(elapsed * 1000, 2),  # ms
            'cost': MetricsCalculator.compute_sum_of_costs(paths) if paths else 0
        })
        
        # L-HCBS
        solver = LearningGuidedHCBS(grid, agents, use_learning=True)
        start = time.time()
        paths = solver.solve(starts, goals, max_iterations=2000, timeout=10.0)
        elapsed = time.time() - start
        case_results['algorithms'].append({
            'name': 'L-HCBS',
            'success': paths is not None,
            'time': round(elapsed * 1000, 2),
            'cost': MetricsCalculator.compute_sum_of_costs(paths) if paths else 0,
            'iterations': solver.stats.iterations
        })
        
        results.append(case_results)
    
    return results


def generate_html(data):
    """Generate interactive HTML visualization."""
    
    html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>L-HCBS Algorithm Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            padding: 30px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #aaa;
            font-size: 1.1em;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .section {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
        }
        
        .section h2 {
            color: #00d2ff;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(0,210,255,0.3);
        }
        
        .scenario-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .tab-btn {
            padding: 12px 24px;
            border: none;
            background: rgba(255,255,255,0.1);
            color: #fff;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .tab-btn:hover, .tab-btn.active {
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        }
        
        .grid-container {
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }
        
        .grid-canvas-wrapper {
            flex: 1;
            min-width: 400px;
        }
        
        #gridCanvas {
            background: #0a0a1a;
            border-radius: 10px;
            border: 2px solid rgba(0,210,255,0.3);
        }
        
        .controls {
            flex: 0 0 300px;
        }
        
        .control-panel {
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 20px;
        }
        
        .control-row {
            margin-bottom: 15px;
        }
        
        .control-row label {
            display: block;
            margin-bottom: 5px;
            color: #aaa;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
            margin: 5px;
        }
        
        .btn-play {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
        }
        
        .btn-pause {
            background: linear-gradient(90deg, #FF9800, #f57c00);
            color: white;
        }
        
        .btn-reset {
            background: linear-gradient(90deg, #f44336, #d32f2f);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .slider {
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: rgba(255,255,255,0.2);
            outline: none;
            -webkit-appearance: none;
        }
        
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #00d2ff;
            cursor: pointer;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #00d2ff;
        }
        
        .stat-label {
            color: #888;
            font-size: 0.9em;
        }
        
        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 15px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 50%;
        }
        
        .chart-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        
        .chart-card {
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 20px;
        }
        
        .chart-card h3 {
            color: #00d2ff;
            margin-bottom: 15px;
        }
        
        .info-box {
            background: rgba(0,210,255,0.1);
            border-left: 4px solid #00d2ff;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            margin-top: 15px;
        }
        
        .timeline {
            position: relative;
            margin-top: 20px;
            height: 60px;
        }
        
        .timeline-bar {
            position: absolute;
            top: 20px;
            left: 0;
            right: 0;
            height: 8px;
            background: rgba(255,255,255,0.2);
            border-radius: 4px;
        }
        
        .timeline-progress {
            height: 100%;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            border-radius: 4px;
            transition: width 0.1s;
        }
        
        .timeline-marker {
            position: absolute;
            top: 10px;
            width: 28px;
            height: 28px;
            background: #00d2ff;
            border-radius: 50%;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            transition: left 0.1s;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        .agent-moving {
            animation: pulse 0.5s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 L-HCBS Algorithm Visualization</h1>
            <p>Learning-guided Heterogeneous Conflict-Based Search for Multi-Agent Path Finding</p>
        </div>
        
        <div class="section">
            <h2>📍 Path Planning Scenarios</h2>
            
            <div class="scenario-tabs">
                <button class="tab-btn active" onclick="switchScenario(0)">Two AGVs Crossing</button>
                <button class="tab-btn" onclick="switchScenario(1)">Bottleneck Navigation</button>
                <button class="tab-btn" onclick="switchScenario(2)">Manufacturing Floor</button>
            </div>
            
            <div class="grid-container">
                <div class="grid-canvas-wrapper">
                    <canvas id="gridCanvas" width="500" height="500"></canvas>
                    
                    <div class="timeline">
                        <div class="timeline-bar">
                            <div class="timeline-progress" id="timelineProgress"></div>
                        </div>
                        <div class="timeline-marker" id="timelineMarker">0</div>
                    </div>
                    
                    <div class="legend" id="legend"></div>
                </div>
                
                <div class="controls">
                    <div class="control-panel">
                        <h3 style="color:#00d2ff; margin-bottom:15px;">Animation Controls</h3>
                        
                        <div class="control-row">
                            <button class="btn btn-play" onclick="playAnimation()">▶ Play</button>
                            <button class="btn btn-pause" onclick="pauseAnimation()">⏸ Pause</button>
                            <button class="btn btn-reset" onclick="resetAnimation()">↺ Reset</button>
                        </div>
                        
                        <div class="control-row">
                            <label>Speed: <span id="speedValue">1x</span></label>
                            <input type="range" class="slider" id="speedSlider" min="0.5" max="3" step="0.5" value="1" onchange="updateSpeed()">
                        </div>
                        
                        <div class="control-row">
                            <label>Time Step: <span id="timeValue">0</span></label>
                            <input type="range" class="slider" id="timeSlider" min="0" max="100" value="0" onchange="seekTime()">
                        </div>
                        
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value" id="statAgents">0</div>
                                <div class="stat-label">Agents</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="statCost">0</div>
                                <div class="stat-label">Total Cost</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="statMakespan">0</div>
                                <div class="stat-label">Makespan</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="statTime">0</div>
                                <div class="stat-label">Current Step</div>
                            </div>
                        </div>
                        
                        <div class="info-box">
                            <strong>Agent Types:</strong><br>
                            🟢 AGV - Mobile transport vehicle<br>
                            🔴 Robot - Limited workspace<br>
                            ⚫ CNC - Static machine
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Algorithm Comparison</h2>
            
            <div class="chart-container">
                <div class="chart-card">
                    <h3>Runtime Comparison (ms)</h3>
                    <canvas id="runtimeChart"></canvas>
                </div>
                <div class="chart-card">
                    <h3>Solution Cost</h3>
                    <canvas id="costChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🧠 Algorithm Overview</h2>
            
            <div class="chart-container">
                <div class="chart-card">
                    <h3>Key Contributions</h3>
                    <ul style="line-height:2; padding-left:20px;">
                        <li><strong>Heterogeneous MAPF</strong> - CNC + AGV + Robot with different kinematics</li>
                        <li><strong>GNN Conflict Prediction</strong> - Predict which conflicts are hard to resolve</li>
                        <li><strong>Learning-guided Search</strong> - Prioritize high-impact conflicts</li>
                        <li><strong>Online Replanning</strong> - Handle runtime disruptions</li>
                    </ul>
                </div>
                <div class="chart-card">
                    <h3>Theoretical Guarantees</h3>
                    <canvas id="theoryChart"></canvas>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Data from Python
        const visualizationData = ''' + json.dumps(data) + ''';
        
        // State
        let currentScenario = 0;
        let currentTime = 0;
        let animationSpeed = 1;
        let isPlaying = false;
        let animationId = null;
        let maxTime = 0;
        
        // Canvas setup
        const canvas = document.getElementById('gridCanvas');
        const ctx = canvas.getContext('2d');
        
        function switchScenario(index) {
            currentScenario = index;
            resetAnimation();
            
            // Update tabs
            document.querySelectorAll('.tab-btn').forEach((btn, i) => {
                btn.classList.toggle('active', i === index);
            });
            
            // Update stats
            const scenario = visualizationData.scenarios[index];
            const paths = scenario.paths;
            const agents = scenario.agents;
            
            document.getElementById('statAgents').textContent = agents.length;
            
            if (Object.keys(paths).length > 0) {
                const costs = Object.values(paths).map(p => p.length - 1);
                const totalCost = costs.reduce((a, b) => a + b, 0);
                maxTime = Math.max(...costs);
                
                document.getElementById('statCost').textContent = totalCost;
                document.getElementById('statMakespan').textContent = maxTime;
                document.getElementById('timeSlider').max = maxTime;
            }
            
            // Update legend
            const legendDiv = document.getElementById('legend');
            legendDiv.innerHTML = agents.map(a => 
                `<div class="legend-item">
                    <div class="legend-color" style="background:${a.color}"></div>
                    <span>${a.id} (${a.type})</span>
                </div>`
            ).join('');
            
            drawGrid();
        }
        
        function drawGrid() {
            const scenario = visualizationData.scenarios[currentScenario];
            const grid = scenario.grid;
            const paths = scenario.paths;
            const agents = scenario.agents;
            
            const cellSize = Math.min(canvas.width / grid.width, canvas.height / grid.height);
            const offsetX = (canvas.width - cellSize * grid.width) / 2;
            const offsetY = (canvas.height - cellSize * grid.height) / 2;
            
            // Clear canvas
            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw grid lines
            ctx.strokeStyle = 'rgba(255,255,255,0.1)';
            ctx.lineWidth = 1;
            for (let x = 0; x <= grid.width; x++) {
                ctx.beginPath();
                ctx.moveTo(offsetX + x * cellSize, offsetY);
                ctx.lineTo(offsetX + x * cellSize, offsetY + grid.height * cellSize);
                ctx.stroke();
            }
            for (let y = 0; y <= grid.height; y++) {
                ctx.beginPath();
                ctx.moveTo(offsetX, offsetY + y * cellSize);
                ctx.lineTo(offsetX + grid.width * cellSize, offsetY + y * cellSize);
                ctx.stroke();
            }
            
            // Draw obstacles
            ctx.fillStyle = '#333';
            for (const [ox, oy] of grid.obstacles) {
                ctx.fillRect(
                    offsetX + ox * cellSize + 1,
                    offsetY + oy * cellSize + 1,
                    cellSize - 2,
                    cellSize - 2
                );
            }
            
            // Draw paths (trails)
            for (const agent of agents) {
                const path = paths[agent.id];
                if (!path || path.length === 0) continue;
                
                ctx.strokeStyle = agent.color + '40';
                ctx.lineWidth = 3;
                ctx.beginPath();
                
                const startX = offsetX + path[0][0] * cellSize + cellSize / 2;
                const startY = offsetY + path[0][1] * cellSize + cellSize / 2;
                ctx.moveTo(startX, startY);
                
                for (let i = 1; i < path.length; i++) {
                    const x = offsetX + path[i][0] * cellSize + cellSize / 2;
                    const y = offsetY + path[i][1] * cellSize + cellSize / 2;
                    ctx.lineTo(x, y);
                }
                ctx.stroke();
            }
            
            // Draw traveled path
            for (const agent of agents) {
                const path = paths[agent.id];
                if (!path || path.length === 0) continue;
                
                ctx.strokeStyle = agent.color;
                ctx.lineWidth = 4;
                ctx.beginPath();
                
                const traveledLength = Math.min(currentTime + 1, path.length);
                const startX = offsetX + path[0][0] * cellSize + cellSize / 2;
                const startY = offsetY + path[0][1] * cellSize + cellSize / 2;
                ctx.moveTo(startX, startY);
                
                for (let i = 1; i < traveledLength; i++) {
                    const x = offsetX + path[i][0] * cellSize + cellSize / 2;
                    const y = offsetY + path[i][1] * cellSize + cellSize / 2;
                    ctx.lineTo(x, y);
                }
                ctx.stroke();
            }
            
            // Draw goals (flags)
            for (const agent of agents) {
                const [gx, gy] = agent.goal;
                ctx.fillStyle = agent.color + '60';
                ctx.beginPath();
                ctx.arc(
                    offsetX + gx * cellSize + cellSize / 2,
                    offsetY + gy * cellSize + cellSize / 2,
                    cellSize / 3, 0, Math.PI * 2
                );
                ctx.fill();
                ctx.strokeStyle = agent.color;
                ctx.lineWidth = 2;
                ctx.stroke();
            }
            
            // Draw agents at current time
            for (const agent of agents) {
                const path = paths[agent.id];
                if (!path || path.length === 0) continue;
                
                const idx = Math.min(currentTime, path.length - 1);
                const [ax, ay] = path[idx];
                
                const x = offsetX + ax * cellSize + cellSize / 2;
                const y = offsetY + ay * cellSize + cellSize / 2;
                
                // Agent body
                ctx.fillStyle = agent.color;
                ctx.beginPath();
                
                if (agent.type === 'CNC') {
                    // Square for CNC
                    ctx.fillRect(x - cellSize/3, y - cellSize/3, cellSize*2/3, cellSize*2/3);
                } else if (agent.type === 'ROBOT') {
                    // Diamond for Robot
                    ctx.moveTo(x, y - cellSize/3);
                    ctx.lineTo(x + cellSize/3, y);
                    ctx.lineTo(x, y + cellSize/3);
                    ctx.lineTo(x - cellSize/3, y);
                    ctx.closePath();
                    ctx.fill();
                } else {
                    // Circle for AGV
                    ctx.arc(x, y, cellSize / 3, 0, Math.PI * 2);
                    ctx.fill();
                }
                
                // Agent label
                ctx.fillStyle = '#fff';
                ctx.font = '10px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(agent.id.split('_')[1], x, y + 4);
            }
            
            // Update timeline
            const progress = maxTime > 0 ? (currentTime / maxTime) * 100 : 0;
            document.getElementById('timelineProgress').style.width = progress + '%';
            document.getElementById('timelineMarker').style.left = progress + '%';
            document.getElementById('timelineMarker').textContent = currentTime;
            document.getElementById('statTime').textContent = currentTime;
            document.getElementById('timeValue').textContent = currentTime;
            document.getElementById('timeSlider').value = currentTime;
        }
        
        function playAnimation() {
            if (isPlaying) return;
            isPlaying = true;
            
            function animate() {
                if (!isPlaying) return;
                
                currentTime += 1;
                if (currentTime > maxTime) {
                    currentTime = 0;
                }
                
                drawGrid();
                animationId = setTimeout(animate, 500 / animationSpeed);
            }
            
            animate();
        }
        
        function pauseAnimation() {
            isPlaying = false;
            if (animationId) {
                clearTimeout(animationId);
            }
        }
        
        function resetAnimation() {
            pauseAnimation();
            currentTime = 0;
            drawGrid();
        }
        
        function updateSpeed() {
            animationSpeed = parseFloat(document.getElementById('speedSlider').value);
            document.getElementById('speedValue').textContent = animationSpeed + 'x';
        }
        
        function seekTime() {
            currentTime = parseInt(document.getElementById('timeSlider').value);
            drawGrid();
        }
        
        // Initialize charts
        function initCharts() {
            const comparison = visualizationData.comparison;
            const labels = comparison.map(c => c.agents + ' agents');
            
            // Runtime chart
            new Chart(document.getElementById('runtimeChart'), {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'H-CBS',
                            data: comparison.map(c => c.algorithms.find(a => a.name === 'H-CBS')?.time || 0),
                            backgroundColor: 'rgba(255, 99, 132, 0.7)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 1
                        },
                        {
                            label: 'L-HCBS',
                            data: comparison.map(c => c.algorithms.find(a => a.name === 'L-HCBS')?.time || 0),
                            backgroundColor: 'rgba(54, 162, 235, 0.7)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { labels: { color: '#fff' } } },
                    scales: {
                        y: { ticks: { color: '#aaa' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                        x: { ticks: { color: '#aaa' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                    }
                }
            });
            
            // Cost chart
            new Chart(document.getElementById('costChart'), {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'H-CBS',
                            data: comparison.map(c => c.algorithms.find(a => a.name === 'H-CBS')?.cost || 0),
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            fill: true,
                            tension: 0.4
                        },
                        {
                            label: 'L-HCBS',
                            data: comparison.map(c => c.algorithms.find(a => a.name === 'L-HCBS')?.cost || 0),
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            fill: true,
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { labels: { color: '#fff' } } },
                    scales: {
                        y: { ticks: { color: '#aaa' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                        x: { ticks: { color: '#aaa' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                    }
                }
            });
            
            // Theory chart (radar)
            new Chart(document.getElementById('theoryChart'), {
                type: 'radar',
                data: {
                    labels: ['Completeness', 'Optimality', 'Scalability', 'Online Adapt', 'Heterogeneous'],
                    datasets: [
                        {
                            label: 'Standard CBS',
                            data: [100, 100, 40, 20, 30],
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        },
                        {
                            label: 'L-HCBS (Ours)',
                            data: [100, 90, 75, 85, 100],
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: { legend: { labels: { color: '#fff' } } },
                    scales: {
                        r: {
                            angleLines: { color: 'rgba(255,255,255,0.2)' },
                            grid: { color: 'rgba(255,255,255,0.2)' },
                            pointLabels: { color: '#fff' },
                            ticks: { display: false }
                        }
                    }
                }
            });
        }
        
        // Initialize
        switchScenario(0);
        initCharts();
    </script>
</body>
</html>'''
    
    return html


def main():
    print("="*60)
    print("  L-HCBS Visualization Generator")
    print("="*60)
    
    # Generate data
    data = generate_visualization_data()
    
    # Generate HTML
    html = generate_html(data)
    
    # Save to file
    output_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'lhcbs_visualization.html')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✅ Visualization saved to: {output_path}")
    print("\nOpen in browser to view the interactive visualization.")
    
    return output_path


if __name__ == "__main__":
    main()
