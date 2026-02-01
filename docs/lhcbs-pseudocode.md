# L-HCBS Algorithm Pseudocode

## Learning-guided Heterogeneous Conflict-Based Search

面向异构智能体的学习引导冲突搜索算法

---

## Algorithm 1: L-HCBS Main Algorithm

```
Algorithm 1: Learning-guided Heterogeneous CBS (L-HCBS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:  G = (V, E)          // Grid map
        A = {a₁, ..., aₙ}   // Heterogeneous agents with kinematics
        s = {s₁, ..., sₙ}   // Start positions
        g = {g₁, ..., gₙ}   // Goal positions
        w ≥ 1               // Suboptimality bound
        GNN_θ               // Trained conflict predictor network

Output: Π = {π₁, ..., πₙ}   // Collision-free paths for all agents

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1:  function L-HCBS(G, A, s, g, w, GNN_θ)
2:      // Initialize root node
3:      R.constraints ← ∅
4:      for each agent aᵢ ∈ A do
5:          R.paths[aᵢ] ← HeterogeneousA*(G, aᵢ, sᵢ, gᵢ, ∅)
6:      end for
7:      R.cost ← Σᵢ |R.paths[aᵢ]|
8:      R.conflicts ← DetectConflicts(R.paths, A)
9:      R.h_learned ← GNN_θ.Predict(R)        // Learning-guided heuristic
10:     
11:     OPEN ← {R}
12:     FOCAL ← {R}
13:     
14:     while OPEN ≠ ∅ do
15:         // Update focal list with bounded suboptimality
16:         f_min ← min{N.cost : N ∈ OPEN}
17:         FOCAL ← {N ∈ OPEN : N.cost ≤ w · f_min}
18:         
19:         // Select node using learned priority (fail-fast strategy)
20:         N ← argmax{N.h_learned : N ∈ FOCAL}
21:         Remove N from OPEN and FOCAL
22:         
23:         // Check for solution
24:         if N.conflicts = ∅ then
25:             return N.paths
26:         end if
27:         
28:         // Select conflict using GNN prediction
29:         C ← SelectConflict(N.conflicts, GNN_θ)
30:         
31:         // Generate child nodes (branching)
32:         for each agent aᵢ ∈ {C.agent₁, C.agent₂} do
33:             N' ← Copy(N)
34:             κ ← CreateConstraint(C, aᵢ)
35:             N'.constraints ← N'.constraints ∪ {κ}
36:             
37:             // Replan for constrained agent
38:             N'.paths[aᵢ] ← HeterogeneousA*(G, aᵢ, sᵢ, gᵢ, N'.constraints)
39:             
40:             if N'.paths[aᵢ] ≠ NULL then
41:                 N'.cost ← Σⱼ |N'.paths[aⱼ]|
42:                 N'.conflicts ← DetectConflicts(N'.paths, A)
43:                 N'.h_learned ← GNN_θ.Predict(N')
44:                 Insert N' into OPEN
45:                 if N'.cost ≤ w · f_min then
46:                     Insert N' into FOCAL
47:                 end if
48:             end if
49:         end for
50:     end while
51:     
52:     return FAILURE
53: end function
```

---

## Algorithm 2: Heterogeneous Low-Level Planner

```
Algorithm 2: Heterogeneous A* (Low-Level Planner)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:  G = (V, E)          // Grid map
        a                   // Agent with kinematics K_a
        s                   // Start position
        g                   // Goal position
        Ω                   // Set of constraints

Output: π                   // Kinematically feasible path

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1:  function HeterogeneousA*(G, a, s, g, Ω)
2:      OPEN ← {(s, 0)}
3:      g_score[s] ← 0
4:      f_score[s] ← Heuristic(s, g, a.type)
5:      came_from ← {}
6:      
7:      while OPEN ≠ ∅ do
8:          current ← argmin{f_score[n] : n ∈ OPEN}
9:          
10:         if current.pos = g then
11:             return ReconstructPath(came_from, current)
12:         end if
13:         
14:         Remove current from OPEN
15:         
16:         // Get valid moves based on agent type
17:         neighbors ← GetKinematicNeighbors(current, a.type)
18:         
19:         for each (next_pos, next_time) ∈ neighbors do
20:             // Check constraints
21:             if ViolatesConstraint(a, next_pos, next_time, Ω) then
22:                 continue
23:             end if
24:             
25:             // Check kinematic feasibility
26:             if ¬IsKinematicallyFeasible(current, next_pos, a.kinematics) then
27:                 continue
28:             end if
29:             
30:             // Compute transition cost based on agent type
31:             move_cost ← ComputeMoveCost(current, next_pos, a)
32:             tentative_g ← g_score[current] + move_cost
33:             
34:             if tentative_g < g_score[next_pos] then
35:                 came_from[next_pos] ← current
36:                 g_score[next_pos] ← tentative_g
37:                 f_score[next_pos] ← tentative_g + Heuristic(next_pos, g, a.type)
38:                 Insert (next_pos, next_time) into OPEN
39:             end if
40:         end for
41:     end while
42:     
43:     return NULL  // No path found
44: end function
```

---

## Algorithm 3: GNN Conflict Predictor

```
Algorithm 3: GNN-based Conflict Prediction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:  N                   // CT Node with current paths
        A                   // Agent set with features
        
Output: p ∈ [0,1]^(n×n)    // Conflict probability matrix

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1:  function GNN_Predict(N, A)
2:      // Extract node features for each agent
3:      for each agent aᵢ ∈ A do
4:          xᵢ ← [type_embedding(aᵢ.type),    // Agent type (one-hot)
5:                aᵢ.position,                 // Current position
6:                aᵢ.goal,                     // Goal position  
7:                aᵢ.velocity,                 // Max velocity
8:                path_length(N.paths[aᵢ]),    // Current path length
9:                remaining_distance(aᵢ)]      // Distance to goal
10:     end for
11:     
12:     // Extract edge features for agent pairs
13:     for each pair (aᵢ, aⱼ) where i < j do
14:         eᵢⱼ ← [spatial_distance(aᵢ, aⱼ),          // Euclidean distance
15:                temporal_overlap(πᵢ, πⱼ),           // Path overlap in time
16:                path_intersection_count(πᵢ, πⱼ),    // Shared cells
17:                velocity_ratio(aᵢ, aⱼ),             // Speed difference
18:                type_compatibility(aᵢ.type, aⱼ.type)] // Type pair embedding
19:     end for
20:     
21:     // Graph Neural Network forward pass
22:     H⁽⁰⁾ ← X                              // Initial node embeddings
23:     
24:     for l = 1 to L do                     // L message passing layers
25:         for each agent aᵢ do
26:             // Aggregate neighbor messages
27:             mᵢ ← Σⱼ∈𝒩(i) α(hᵢ⁽ˡ⁻¹⁾, hⱼ⁽ˡ⁻¹⁾, eᵢⱼ) · Wₘ · hⱼ⁽ˡ⁻¹⁾
28:             
29:             // Update node embedding
30:             hᵢ⁽ˡ⁾ ← ReLU(Wᵤ · [hᵢ⁽ˡ⁻¹⁾ ‖ mᵢ])
31:         end for
32:     end for
33:     
34:     // Predict edge-level conflict probabilities
35:     for each pair (aᵢ, aⱼ) do
36:         zᵢⱼ ← MLP([hᵢ⁽ᴸ⁾ ‖ hⱼ⁽ᴸ⁾ ‖ eᵢⱼ])
37:         pᵢⱼ ← σ(zᵢⱼ)                       // Sigmoid activation
38:     end for
39:     
40:     return P = {pᵢⱼ}
41: end function
```

---

## Algorithm 4: Conflict Selection

```
Algorithm 4: Learning-guided Conflict Selection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:  C = {c₁, ..., cₖ}   // Set of detected conflicts
        GNN_θ               // Trained predictor
        
Output: c*                  // Selected conflict to resolve

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1:  function SelectConflict(C, GNN_θ)
2:      if |C| = 1 then
3:          return C[0]
4:      end if
5:      
6:      best_conflict ← NULL
7:      best_score ← -∞
8:      
9:      for each conflict c ∈ C do
10:         // Get predicted severity from GNN
11:         severity ← GNN_θ.EdgeScore(c.agent₁, c.agent₂)
12:         
13:         // Compute conflict impact score
14:         score ← severity × ConflictTypeWeight(c.type)
15:         
16:         // Bonus for early-time conflicts (resolve sooner)
17:         score ← score + λ · (1 / (c.time + 1))
18:         
19:         if score > best_score then
20:             best_score ← score
21:             best_conflict ← c
22:         end if
23:     end for
24:     
25:     return best_conflict
26: end function
```

---

## Algorithm 5: Heterogeneous Conflict Detection

```
Algorithm 5: Heterogeneous Conflict Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:  Π = {π₁, ..., πₙ}   // Paths for all agents
        A = {a₁, ..., aₙ}   // Agents with footprint info

Output: C                    // Set of conflicts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1:  function DetectConflicts(Π, A)
2:      C ← ∅
3:      T_max ← max{|πᵢ| : πᵢ ∈ Π}
4:      
5:      for t = 0 to T_max do
6:          for each pair (aᵢ, aⱼ) where i < j do
7:              posᵢ ← GetPosition(πᵢ, t)
8:              posⱼ ← GetPosition(πⱼ, t)
9:              
10:             // Get agent footprints based on type
11:             Fᵢ ← GetFootprint(aᵢ, posᵢ)
12:             Fⱼ ← GetFootprint(aⱼ, posⱼ)
13:             
14:             // Vertex conflict (footprint overlap)
15:             if Fᵢ ∩ Fⱼ ≠ ∅ then
16:                 C ← C ∪ {VertexConflict(aᵢ, aⱼ, t, posᵢ)}
17:             end if
18:             
19:             // Edge conflict (swap positions)
20:             if t > 0 then
21:                 prev_posᵢ ← GetPosition(πᵢ, t-1)
22:                 prev_posⱼ ← GetPosition(πⱼ, t-1)
23:                 
24:                 if posᵢ = prev_posⱼ and posⱼ = prev_posᵢ then
25:                     C ← C ∪ {EdgeConflict(aᵢ, aⱼ, t, posᵢ, posⱼ)}
26:                 end if
27:             end if
28:             
29:             // Workspace conflict (for robots with reach)
30:             if aᵢ.type = ROBOT or aⱼ.type = ROBOT then
31:                 Wᵢ ← GetWorkspace(aᵢ, posᵢ)
32:                 Wⱼ ← GetWorkspace(aⱼ, posⱼ)
33:                 
34:                 if Wᵢ ∩ Fⱼ ≠ ∅ or Fᵢ ∩ Wⱼ ≠ ∅ then
35:                     C ← C ∪ {WorkspaceConflict(aᵢ, aⱼ, t)}
36:                 end if
37:             end if
38:         end for
39:     end for
40:     
41:     return C
42: end function
```

---

## Algorithm 6: Online Replanning

```
Algorithm 6: Dynamic Replanning with Disruptions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:  Π                   // Current solution
        t_now               // Current time
        D                   // Disruption event
        
Output: Π'                  // Updated solution

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1:  function OnlineReplan(Π, t_now, D)
2:      // Identify affected agents
3:      A_affected ← GetAffectedAgents(D, Π, t_now)
4:      
5:      if D.type = MACHINE_FAILURE then
6:          // Add obstacle at failed machine location
7:          G ← G ∪ {D.location as obstacle}
8:          // Remove failed agent
9:          A ← A \ {D.agent}
10:     
11:     else if D.type = EMERGENCY_ORDER then
12:         // Add new agent with high priority
13:         a_new ← CreateAgent(D.order)
14:         a_new.priority ← HIGH
15:         A ← A ∪ {a_new}
16:         A_affected ← A_affected ∪ {a_new}
17:     
18:     else if D.type = PATH_BLOCKED then
19:         // Temporarily block cells
20:         G ← G ∪ {D.cells as temporary obstacles}
21:     end if
22:     
23:     // Preserve committed path segments
24:     Π_committed ← {}
25:     for each agent aᵢ ∈ A do
26:         Π_committed[aᵢ] ← πᵢ[0 : t_now + T_safe]
27:     end for
28:     
29:     // Partial replanning for affected agents only
30:     s' ← {GetPosition(πᵢ, t_now + T_safe) : aᵢ ∈ A_affected}
31:     g' ← {gᵢ : aᵢ ∈ A_affected}
32:     
33:     // Compute constraints from unaffected agents
34:     Ω_fixed ← ComputeFixedConstraints(Π, A \ A_affected, t_now)
35:     
36:     // Replan affected agents
37:     Π_new ← L-HCBS(G, A_affected, s', g', w, GNN_θ, Ω_fixed)
38:     
39:     // Merge solutions
40:     Π' ← MergePaths(Π_committed, Π_new, t_now + T_safe)
41:     
42:     return Π'
43: end function
```

---

## Theoretical Properties

### Theorem 1: Completeness
```
L-HCBS is complete: If a solution exists, L-HCBS will find it.

Proof Sketch:
- L-HCBS explores the same search space as standard CBS
- Learning only affects exploration ORDER, not the space itself
- Focal search with w ≥ 1 includes all optimal solutions
- Therefore, completeness is preserved from CBS □
```

### Theorem 2: Bounded Suboptimality
```
For suboptimality bound w ≥ 1, L-HCBS returns solution Π with:
    cost(Π) ≤ w × cost(Π*)

where Π* is the optimal solution.

Proof Sketch:
- Focal list contains all nodes N with f(N) ≤ w × f_min
- Optimal solution has cost f* = f_min at some iteration
- Selected node always has f(N) ≤ w × f*
- Solution cost bounded by w × optimal □
```

### Theorem 3: GNN Learning Convergence
```
Given sufficient training data, the GNN conflict predictor 
converges to the true conflict probability distribution.

Conditions:
1. Training examples i.i.d. from search distribution
2. Network has sufficient capacity
3. Learning rate follows Robbins-Monro conditions
```

---

## Complexity Analysis

| Component | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| High-level CBS | O(2^k × n × V²) | O(2^k × n × V) |
| Low-level A* | O(V × log V) | O(V) |
| GNN Forward | O(n² × d) | O(n² × d) |
| Conflict Detection | O(n² × T) | O(n² × T) |

Where:
- n = number of agents
- V = number of vertices in grid
- k = number of conflicts resolved
- T = maximum path length
- d = GNN hidden dimension

---

## Notation Summary

| Symbol | Description |
|--------|-------------|
| G | Grid map (V, E) |
| A | Set of agents |
| aᵢ | Agent i |
| sᵢ, gᵢ | Start and goal of agent i |
| πᵢ | Path of agent i |
| Π | Solution (all paths) |
| κ | Constraint |
| Ω | Set of constraints |
| C | Set of conflicts |
| N | Constraint tree node |
| θ | GNN parameters |
| w | Suboptimality bound |
| Fᵢ | Footprint of agent i |
| Wᵢ | Workspace of agent i |

