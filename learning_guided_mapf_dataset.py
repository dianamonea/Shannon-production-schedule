"""
Learning-Guided MAPF 数据集生成器
Dataset Generator for Standard MAPF Benchmarks

生成符合MovingAI标准格式的MAPF数据集

作者：Shannon Research Team
日期：2026-02-01
"""

import random
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


@dataclass
class MAPFInstance:
    """MAPF实例"""
    map_name: str
    width: int
    height: int
    obstacles: List[Tuple[int, int]]
    starts: List[Tuple[int, int]]
    goals: List[Tuple[int, int]]
    num_agents: int
    optimal_cost: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            'map_name': self.map_name,
            'width': self.width,
            'height': self.height,
            'obstacles': self.obstacles,
            'starts': self.starts,
            'goals': self.goals,
            'num_agents': self.num_agents,
            'optimal_cost': self.optimal_cost
        }


class MapGenerator:
    """地图生成器"""
    
    @staticmethod
    def generate_empty(width: int, height: int) -> List[List[int]]:
        """生成空白地图"""
        return [[0] * width for _ in range(height)]
    
    @staticmethod
    def generate_random(width: int, height: int, obstacle_ratio: float) -> List[List[int]]:
        """生成随机障碍物地图"""
        grid = [[0] * width for _ in range(height)]
        num_obstacles = int(width * height * obstacle_ratio)
        
        positions = [(i, j) for i in range(height) for j in range(width)]
        random.shuffle(positions)
        
        for i in range(min(num_obstacles, len(positions))):
            y, x = positions[i]
            grid[y][x] = 1
        
        return grid
    
    @staticmethod
    def generate_room(width: int, height: int, room_size: int = 8, door_width: int = 2) -> List[List[int]]:
        """生成房间地图"""
        grid = [[0] * width for _ in range(height)]
        
        # 添加房间墙壁
        for i in range(room_size, height - room_size, room_size):
            for j in range(width):
                grid[i][j] = 1
        
        for j in range(room_size, width - room_size, room_size):
            for i in range(height):
                grid[i][j] = 1
        
        # 添加门
        for i in range(room_size, height - room_size, room_size):
            door_start = random.randint(1, width - door_width - 1)
            for j in range(door_start, min(door_start + door_width, width)):
                grid[i][j] = 0
        
        for j in range(room_size, width - room_size, room_size):
            door_start = random.randint(1, height - door_width - 1)
            for i in range(door_start, min(door_start + door_width, height)):
                grid[i][j] = 0
        
        return grid
    
    @staticmethod
    def generate_maze(width: int, height: int) -> List[List[int]]:
        """生成迷宫地图（使用递归分割）"""
        grid = [[0] * width for _ in range(height)]
        
        def divide(x1, y1, x2, y2):
            if x2 - x1 < 4 or y2 - y1 < 4:
                return
            
            # 选择分割方向
            if x2 - x1 > y2 - y1:
                # 垂直分割
                wall_x = random.randint(x1 + 2, x2 - 2)
                for y in range(y1, y2):
                    grid[y][wall_x] = 1
                # 开门
                door_y = random.randint(y1, y2 - 1)
                grid[door_y][wall_x] = 0
                divide(x1, y1, wall_x, y2)
                divide(wall_x + 1, y1, x2, y2)
            else:
                # 水平分割
                wall_y = random.randint(y1 + 2, y2 - 2)
                for x in range(x1, x2):
                    grid[wall_y][x] = 1
                # 开门
                door_x = random.randint(x1, x2 - 1)
                grid[wall_y][door_x] = 0
                divide(x1, y1, x2, wall_y)
                divide(x1, wall_y + 1, x2, y2)
        
        divide(0, 0, width, height)
        return grid
    
    @staticmethod
    def generate_warehouse(width: int, height: int, aisle_width: int = 3, shelf_width: int = 2) -> List[List[int]]:
        """生成仓库地图"""
        grid = [[0] * width for _ in range(height)]
        
        # 添加货架
        period = aisle_width + shelf_width
        for j in range(aisle_width, width - aisle_width, period):
            for i in range(aisle_width, height - aisle_width):
                for dj in range(shelf_width):
                    if j + dj < width:
                        grid[i][j + dj] = 1
            
            # 每隔一段距离添加通道
            for i in range(aisle_width + 5, height - aisle_width, 10):
                for dj in range(shelf_width):
                    if j + dj < width and i < height:
                        grid[i][j + dj] = 0
        
        return grid
    
    @staticmethod
    def get_free_positions(grid: List[List[int]]) -> List[Tuple[int, int]]:
        """获取所有空闲位置"""
        free = []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0:
                    free.append((i, j))
        return free


class ScenarioGenerator:
    """场景生成器"""
    
    @staticmethod
    def generate_random_scenario(
        grid: List[List[int]], 
        num_agents: int,
        min_distance: int = 5
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """生成随机起点和终点"""
        free_positions = MapGenerator.get_free_positions(grid)
        
        if len(free_positions) < num_agents * 2:
            raise ValueError(f"Not enough free positions: {len(free_positions)} < {num_agents * 2}")
        
        random.shuffle(free_positions)
        
        starts = []
        goals = []
        used = set()
        
        idx = 0
        while len(starts) < num_agents and idx < len(free_positions):
            pos = free_positions[idx]
            idx += 1
            
            if pos in used:
                continue
            
            # 找一个距离足够远的终点
            for goal_idx in range(idx, len(free_positions)):
                goal = free_positions[goal_idx]
                if goal in used:
                    continue
                
                dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
                if dist >= min_distance:
                    starts.append(pos)
                    goals.append(goal)
                    used.add(pos)
                    used.add(goal)
                    break
        
        if len(starts) < num_agents:
            # 如果距离约束太严格，放宽约束
            for pos in free_positions:
                if len(starts) >= num_agents:
                    break
                if pos in used:
                    continue
                
                for goal in free_positions:
                    if goal in used or goal == pos:
                        continue
                    
                    starts.append(pos)
                    goals.append(goal)
                    used.add(pos)
                    used.add(goal)
                    break
        
        return starts, goals


class MovingAIFormatExporter:
    """MovingAI格式导出器"""
    
    @staticmethod
    def export_map(grid: List[List[int]], filepath: Path, map_name: str):
        """导出.map文件"""
        height = len(grid)
        width = len(grid[0])
        
        with open(filepath, 'w') as f:
            f.write("type octile\n")
            f.write(f"height {height}\n")
            f.write(f"width {width}\n")
            f.write("map\n")
            
            for row in grid:
                line = ''.join('.' if cell == 0 else '@' for cell in row)
                f.write(line + '\n')
    
    @staticmethod
    def export_scenario(
        starts: List[Tuple[int, int]], 
        goals: List[Tuple[int, int]], 
        filepath: Path,
        map_filename: str,
        bucket: int = 0
    ):
        """导出.scen文件"""
        height = max(max(s[0] for s in starts), max(g[0] for g in goals)) + 1
        width = max(max(s[1] for s in starts), max(g[1] for g in goals)) + 1
        
        with open(filepath, 'w') as f:
            f.write("version 1\n")
            
            for i, (start, goal) in enumerate(zip(starts, goals)):
                optimal_length = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
                f.write(f"{bucket}\t{map_filename}\t{width}\t{height}\t")
                f.write(f"{start[1]}\t{start[0]}\t{goal[1]}\t{goal[0]}\t")
                f.write(f"{optimal_length:.6f}\n")


class DatasetGenerator:
    """数据集生成器"""
    
    def __init__(self, output_dir: str = './data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_benchmark_dataset(
        self,
        map_types: List[str] = ['random_10', 'random_20', 'room', 'maze', 'warehouse'],
        map_sizes: List[Tuple[int, int]] = [(32, 32), (64, 64), (128, 128)],
        agent_counts: List[int] = [20, 50, 100, 150, 200],
        instances_per_config: int = 25,
        seed: int = 42
    ):
        """生成完整基准数据集"""
        random.seed(seed)
        np.random.seed(seed)
        
        all_instances = []
        
        for map_type in map_types:
            print(f"\n生成 {map_type} 地图...")
            
            for width, height in map_sizes:
                # 生成地图
                if map_type == 'empty':
                    grid = MapGenerator.generate_empty(width, height)
                elif map_type == 'random_10':
                    grid = MapGenerator.generate_random(width, height, 0.1)
                elif map_type == 'random_20':
                    grid = MapGenerator.generate_random(width, height, 0.2)
                elif map_type == 'random_30':
                    grid = MapGenerator.generate_random(width, height, 0.3)
                elif map_type == 'room':
                    grid = MapGenerator.generate_room(width, height)
                elif map_type == 'maze':
                    grid = MapGenerator.generate_maze(width, height)
                elif map_type == 'warehouse':
                    grid = MapGenerator.generate_warehouse(width, height)
                else:
                    continue
                
                # 导出地图文件
                map_dir = self.output_dir / 'maps' / map_type
                map_dir.mkdir(exist_ok=True, parents=True)
                map_name = f"{map_type}_{width}x{height}"
                map_filepath = map_dir / f"{map_name}.map"
                MovingAIFormatExporter.export_map(grid, map_filepath, map_name)
                
                for num_agents in agent_counts:
                    print(f"  - {width}x{height}, {num_agents} agents...")
                    
                    for inst_id in range(instances_per_config):
                        try:
                            starts, goals = ScenarioGenerator.generate_random_scenario(
                                grid, num_agents, min_distance=5
                            )
                            
                            # 创建实例
                            obstacles = [(i, j) for i in range(height) for j in range(width) if grid[i][j] == 1]
                            
                            instance = MAPFInstance(
                                map_name=map_name,
                                width=width,
                                height=height,
                                obstacles=obstacles,
                                starts=starts,
                                goals=goals,
                                num_agents=num_agents
                            )
                            
                            all_instances.append(instance)
                            
                            # 导出场景文件
                            scen_dir = self.output_dir / 'scenarios' / map_type
                            scen_dir.mkdir(exist_ok=True, parents=True)
                            scen_name = f"{map_name}_agents{num_agents}_{inst_id}"
                            scen_filepath = scen_dir / f"{scen_name}.scen"
                            MovingAIFormatExporter.export_scenario(
                                starts, goals, scen_filepath, f"{map_name}.map"
                            )
                            
                        except ValueError as e:
                            print(f"    跳过: {e}")
                            continue
        
        # 保存实例索引
        index = {
            'total_instances': len(all_instances),
            'map_types': map_types,
            'map_sizes': [f"{w}x{h}" for w, h in map_sizes],
            'agent_counts': agent_counts,
            'instances_per_config': instances_per_config,
            'seed': seed,
            'instances': [inst.to_dict() for inst in all_instances]
        }
        
        with open(self.output_dir / 'dataset_index.json', 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"\n✅ 数据集生成完成！共 {len(all_instances)} 个实例")
        print(f"   保存位置: {self.output_dir}")
        
        return all_instances
    
    def generate_training_data(
        self,
        num_instances: int = 1000,
        agent_range: Tuple[int, int] = (10, 100),
        seed: int = 42
    ):
        """生成训练数据"""
        random.seed(seed)
        np.random.seed(seed)
        
        train_dir = self.output_dir / 'train'
        train_dir.mkdir(exist_ok=True, parents=True)
        
        instances = []
        map_types = ['random_20', 'room', 'warehouse']
        
        for i in range(num_instances):
            if i % 100 == 0:
                print(f"生成训练实例 {i}/{num_instances}...")
            
            # 随机选择参数
            map_type = random.choice(map_types)
            width = height = random.choice([32, 48, 64])
            num_agents = random.randint(agent_range[0], agent_range[1])
            
            # 生成地图
            if map_type == 'random_20':
                grid = MapGenerator.generate_random(width, height, 0.2)
            elif map_type == 'room':
                grid = MapGenerator.generate_room(width, height)
            else:
                grid = MapGenerator.generate_warehouse(width, height)
            
            try:
                starts, goals = ScenarioGenerator.generate_random_scenario(
                    grid, num_agents, min_distance=3
                )
                
                obstacles = [(i, j) for i in range(height) for j in range(width) if grid[i][j] == 1]
                
                instance = {
                    'id': i,
                    'map_type': map_type,
                    'width': width,
                    'height': height,
                    'obstacles': obstacles,
                    'starts': starts,
                    'goals': goals,
                    'num_agents': num_agents
                }
                
                instances.append(instance)
                
            except ValueError:
                continue
        
        # 分割训练集和验证集
        random.shuffle(instances)
        split_idx = int(len(instances) * 0.9)
        train_instances = instances[:split_idx]
        val_instances = instances[split_idx:]
        
        # 保存
        with open(train_dir / 'train.json', 'w') as f:
            json.dump({'instances': train_instances}, f)
        
        with open(train_dir / 'val.json', 'w') as f:
            json.dump({'instances': val_instances}, f)
        
        print(f"\n✅ 训练数据生成完成！")
        print(f"   训练集: {len(train_instances)} 实例")
        print(f"   验证集: {len(val_instances)} 实例")
        
        return train_instances, val_instances


def main():
    generator = DatasetGenerator(output_dir='./mapf_data')
    
    # 生成基准测试数据集
    print("=" * 50)
    print("生成基准测试数据集...")
    print("=" * 50)
    generator.generate_benchmark_dataset(
        map_types=['random_10', 'random_20', 'room', 'maze', 'warehouse'],
        map_sizes=[(32, 32), (64, 64)],
        agent_counts=[20, 50, 100],
        instances_per_config=10,
        seed=42
    )
    
    # 生成训练数据
    print("\n" + "=" * 50)
    print("生成训练数据...")
    print("=" * 50)
    generator.generate_training_data(
        num_instances=500,
        agent_range=(10, 80),
        seed=42
    )


if __name__ == '__main__':
    main()
