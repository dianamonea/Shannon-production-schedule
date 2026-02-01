"""
MAPF Module for Shannon

Learning-guided Heterogeneous Conflict-Based Search (L-HCBS)
for multi-agent path finding in manufacturing environments.
"""

from .heterogeneous_agent import (
    AgentType,
    KinematicConstraints,
    HeterogeneousAgent,
    AgentState,
    HeterogeneousAgentFactory,
    AgentDict
)

from .heterogeneous_cbs import (
    GridMap,
    Conflict,
    ConflictType,
    Constraint,
    CTNode,
    HeterogeneousLowLevelPlanner,
    HeterogeneousCBS,
    solve_heterogeneous_mapf
)

from .lhcbs import (
    LearningGuidedHCBS,
    SearchStatistics,
    LCTNode,
    ConflictSelector,
    BranchOrderPredictor,
    solve_with_learning
)

from .gnn_conflict_predictor import (
    ConflictPredictorNetwork,
    ConflictFeatureExtractor,
    AgentFeatures,
    EdgeFeatures,
    TrainingExample,
    ConflictPredictorTrainer
)

from .online_replanning import (
    DisruptionType,
    Disruption,
    ReplanningContext,
    PathValidator,
    PartialReplanner,
    OnlineExecutionManager,
    DisruptionGenerator
)

from .experiments import (
    BenchmarkInstance,
    BenchmarkType,
    BenchmarkGenerator,
    ExperimentResult,
    MetricsCalculator,
    PrioritizedPlanning,
    StandardCBS,
    ExperimentRunner
)

__all__ = [
    # Agents
    'AgentType',
    'KinematicConstraints',
    'HeterogeneousAgent',
    'AgentState',
    'HeterogeneousAgentFactory',
    'AgentDict',
    
    # CBS
    'GridMap',
    'Conflict',
    'ConflictType',
    'Constraint',
    'CTNode',
    'HeterogeneousLowLevelPlanner',
    'HeterogeneousCBS',
    'solve_heterogeneous_mapf',
    
    # L-HCBS
    'LearningGuidedHCBS',
    'SearchStatistics',
    'LCTNode',
    'ConflictSelector',
    'BranchOrderPredictor',
    'solve_with_learning',
    
    # GNN
    'ConflictPredictorNetwork',
    'ConflictFeatureExtractor',
    'AgentFeatures',
    'EdgeFeatures',
    'TrainingExample',
    'ConflictPredictorTrainer',
    
    # Online
    'DisruptionType',
    'Disruption',
    'ReplanningContext',
    'PathValidator',
    'PartialReplanner',
    'OnlineExecutionManager',
    'DisruptionGenerator',
    
    # Experiments
    'BenchmarkInstance',
    'BenchmarkType',
    'BenchmarkGenerator',
    'ExperimentResult',
    'MetricsCalculator',
    'PrioritizedPlanning',
    'StandardCBS',
    'ExperimentRunner'
]
