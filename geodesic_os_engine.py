"""
================================================================================
GEODESICOS: DISTRIBUTED COLLECTIVE SENSING & MUTUAL-ELEVATION ENGINE
================================================================================
A Python reference implementation of the 4-Stage Network Evolution Architecture:
  1. Star / Single-Expert Topology (Stage 1)
  2. Federated Multi-Hub Topology (Stage 2)
  3. Hybrid Collective Graph with AI Aggregator (Stage 3)
  4. Resilient Geodesic Small-World Lattice (Stage 4)

Theoretical & Empirical Foundations:
  - Scott E. Page (2007): Diversity Prediction Theorem
  - W. Ross Ashby (1956): Law of Requisite Variety
  - Albert, Jeong, & Barabási (2000): Error & Attack Tolerance of Complex Networks
  - Duncan Watts & Steven Strogatz (1998): Small-World Collective Dynamics
  - Herbert A. Simon (1971): Attention Economics & Cognitive Channel Capacity
  - Robert Axelrod (1984): Iterated Cooperation & Reciprocal Flow
================================================================================
"""

import sys
import os
import json
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
import numpy as np
from scipy import linalg

# ------------------------------------------------------------------------------
# 1. Mathematical & Theoretical Core
# ------------------------------------------------------------------------------

def diversity_prediction_theorem(predictions: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """
    Validates Scott E. Page's Diversity Prediction Theorem:
        Collective Error = Mean Individual Error - Prediction Diversity
    """
    N = predictions.shape[0]
    c_hat = np.mean(predictions, axis=0) # Collective prediction (D,)
    
    # Collective squared error
    collective_error = float(np.sum((c_hat - truth) ** 2))
    
    # Mean individual squared error
    indiv_errors = np.sum((predictions - truth) ** 2, axis=1)
    mean_individual_error = float(np.mean(indiv_errors))
    
    # Prediction diversity: variance around collective mean
    divergences = np.sum((predictions - c_hat) ** 2, axis=1)
    prediction_diversity = float(np.mean(divergences))
    
    discrepancy = abs(collective_error - (mean_individual_error - prediction_diversity))
    assert discrepancy < 1e-7, f"Theorem identity violation: {discrepancy}"
    
    error_reduction = ((mean_individual_error - collective_error) / max(mean_individual_error, 1e-9)) * 100.0
    return {
        "collective_error": collective_error,
        "mean_individual_error": mean_individual_error,
        "prediction_diversity": prediction_diversity,
        "error_reduction_pct": error_reduction
    }

def calculate_algebraic_connectivity(adj_matrix: np.ndarray) -> float:
    """
    Computes the Fiedler eigenvalue (lambda_2 of normalized Laplacian).
    Measures graph robustness and fault-tolerance under node removal.
    """
    degrees = np.sum(adj_matrix, axis=1)
    N = adj_matrix.shape[0]
    if N <= 1 or np.min(degrees) == 0:
        return 0.0
    
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-9)))
    L_norm = np.eye(N) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    eigenvalues = np.sort(np.real(linalg.eigvals(L_norm)))
    lambda_2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    return max(0.0, lambda_2)

def calculate_phase_volume(scale_X: float, diversity_Y: float, precision_Z: float) -> float:
    """
    Ashby's Requisite Variety phase-space volume:
    Measures total sensory and regulatory capacity.
    """
    return float(scale_X * diversity_Y * precision_Z * (4.0 / 3.0 * np.pi * 0.01))

def calculate_decoupling_ratio(
    peer_bandwidth_matrix: np.ndarray,
    reciprocal_flow_matrix: np.ndarray,
    hub_bandwidth: float,
    edge_hub_dependencies: np.ndarray
) -> float:
    """
    Computes the Network Decoupling Ratio Psi_decouple:
        Psi = [ Sum_{i,j} T(i,j) * ln(1 + Rf(i,j)) ] / [ B_c * Sum_k omega(k, hub) ]
    When Psi > 1.0, the network achieves autonomous escape velocity.
    """
    N = peer_bandwidth_matrix.shape[0]
    peer_numerator = 0.0
    for i in range(N):
        for j in range(N):
            if i != j and peer_bandwidth_matrix[i, j] > 0:
                T_ij = peer_bandwidth_matrix[i, j]
                Rf_ij = reciprocal_flow_matrix[i, j]
                peer_numerator += T_ij * np.log(1.0 + max(0.0, Rf_ij))
                
    hub_denominator = hub_bandwidth * np.sum(edge_hub_dependencies)
    if hub_denominator <= 1e-9:
        return 999.0
    return float(peer_numerator / hub_denominator)

# ------------------------------------------------------------------------------
# 2. Protocol Wire Formats & Node Taxonomy
# ------------------------------------------------------------------------------

class NodeTier(Enum):
    TIER_1_CORE_HUB = "tier_1_core_hub"         # Legacy central expert / corporate sponsor
    TIER_2_REGIONAL_HUB = "tier_2_regional_hub" # Mid-tier practice pod / domain specialist
    TIER_3_PERIPHERAL = "tier_3_peripheral"     # Frontline operator / SMB / edge sensor

@dataclass
class NodeProfile:
    node_id: int
    name: str
    tier: NodeTier
    calibration_score: float = 0.5   # Historical accuracy kappa_i in [0.1, 1.0]
    confidence_stake: float = 0.5    # Self-allocated conviction c_i in (0.0, 1.0]
    hub_dependency: float = 1.0      # omega(i, hub) in [0.0, 1.0]
    total_signals_sent: int = 0
    total_capability_received: float = 0.0

@dataclass
class TelemetryPacket:
    node_id: int
    timestamp: float
    ai_absorption_latency_hours: float
    fragility_coefficient: float
    adaptation_speed_score: float
    reciprocal_utility_experienced: float
    raw_observation_vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReciprocalCapabilityPackage:
    recipient_node_id: int
    timestamp: float
    localized_future_readiness_score: float
    global_consensus_vector: np.ndarray
    peer_relative_percentile: float
    anomaly_early_warnings: List[str]
    automated_playbook_modules: List[str]
    reciprocal_value_units: float

# ------------------------------------------------------------------------------
# 3. Network Topologies & P2P Transport Fabric
# ------------------------------------------------------------------------------

class NetworkTopology:
    def __init__(self, nodes: List[NodeProfile], adjacency_matrix: np.ndarray):
        self.nodes = nodes
        self.N = len(nodes)
        self.adj = adjacency_matrix
        self.peer_bandwidth_matrix = np.copy(adjacency_matrix).astype(float)
        self.reciprocal_flow_matrix = np.zeros((self.N, self.N), dtype=float)

    @classmethod
    def create_star_topology(cls, node_count: int) -> "NetworkTopology":
        nodes = [NodeProfile(0, "Central Expert Hub", NodeTier.TIER_1_CORE_HUB, calibration_score=0.85, hub_dependency=0.0)]
        for i in range(1, node_count):
            nodes.append(NodeProfile(i, f"Edge Participant {i}", NodeTier.TIER_3_PERIPHERAL, calibration_score=0.45, hub_dependency=1.0))
        adj = np.zeros((node_count, node_count), dtype=float)
        for i in range(1, node_count):
            adj[0, i] = 1.0
            adj[i, 0] = 1.0
        return cls(nodes, adj)

    @classmethod
    def create_multi_hub_topology(cls, node_count: int, num_hubs: int = 4) -> "NetworkTopology":
        nodes = []
        for h in range(num_hubs):
            nodes.append(NodeProfile(h, f"Regional Hub {h+1}", NodeTier.TIER_2_REGIONAL_HUB, calibration_score=0.75, hub_dependency=0.4))
        for i in range(num_hubs, node_count):
            nodes.append(NodeProfile(i, f"Pod Contributor {i}", NodeTier.TIER_3_PERIPHERAL, calibration_score=0.50, hub_dependency=0.6))
            
        adj = np.zeros((node_count, node_count), dtype=float)
        for h in range(num_hubs):
            next_h = (h + 1) % num_hubs
            adj[h, next_h] = 1.0
            adj[next_h, h] = 1.0
        for i in range(num_hubs, node_count):
            assigned_hub = i % num_hubs
            adj[assigned_hub, i] = 1.0
            adj[i, assigned_hub] = 1.0
            if i > num_hubs and (i - 1) % num_hubs == assigned_hub:
                adj[i, i-1] = 1.0
                adj[i-1, i] = 1.0
        return cls(nodes, adj)

    @classmethod
    def create_small_world_lattice(cls, node_count: int, k_neighbors: int = 6, rewiring_prob: float = 0.15) -> "NetworkTopology":
        nodes = []
        for i in range(node_count):
            tier = NodeTier.TIER_3_PERIPHERAL if i >= 4 else NodeTier.TIER_2_REGIONAL_HUB
            hub_dep = 0.10 if tier == NodeTier.TIER_3_PERIPHERAL else 0.05
            nodes.append(NodeProfile(i, f"Lattice Node {i}", tier, calibration_score=0.55 + 0.3 * np.random.rand(), hub_dependency=hub_dep))

        adj = np.zeros((node_count, node_count), dtype=float)
        half_k = k_neighbors // 2
        for i in range(node_count):
            for j in range(1, half_k + 1):
                neighbor = (i + j) % node_count
                adj[i, neighbor] = 1.0
                adj[neighbor, i] = 1.0
                
        np.random.seed(42)
        for i in range(node_count):
            for j in range(1, half_k + 1):
                if np.random.rand() < rewiring_prob:
                    old_neighbor = (i + j) % node_count
                    adj[i, old_neighbor] = 0.0
                    adj[old_neighbor, i] = 0.0
                    new_neighbor = np.random.randint(0, node_count)
                    while new_neighbor == i or adj[i, new_neighbor] == 1.0:
                        new_neighbor = np.random.randint(0, node_count)
                    adj[i, new_neighbor] = 1.0
                    adj[new_neighbor, i] = 1.0

        topo = cls(nodes, adj)
        for i in range(node_count):
            for j in range(node_count):
                if i != j and adj[i, j] > 0:
                    topo.reciprocal_flow_matrix[i, j] = 2.4 + 0.6 * np.random.rand()
        return topo

    def get_average_path_length(self) -> float:
        N = self.N
        dist = np.full((N, N), np.inf)
        np.fill_diagonal(dist, 0.0)
        for i in range(N):
            for j in range(N):
                if self.adj[i, j] > 0:
                    dist[i, j] = 1.0
        for k in range(N):
            dist = np.minimum(dist, dist[:, k, None] + dist[None, k, :])
        finite_mask = np.isfinite(dist) & (dist > 0)
        return float(np.mean(dist[finite_mask])) if np.any(finite_mask) else float(N)

# ------------------------------------------------------------------------------
# 4. Quadratic Signal Weighting & Reciprocal Arbiter
# ------------------------------------------------------------------------------

class QuadraticSignalAggregator:
    def __init__(self, dimension: int, alpha_novelty: float = 1.35):
        self.D = dimension
        self.alpha = alpha_novelty

    def compute_weights(self, packets: List[TelemetryPacket], profiles: Dict[int, NodeProfile]) -> Tuple[np.ndarray, np.ndarray]:
        N = len(packets)
        if N == 0:
            return np.array([]), np.array([])
            
        vectors = np.array([p.raw_observation_vector for p in packets])
        mean_vec = np.mean(vectors, axis=0)
        norm_mean = np.linalg.norm(mean_vec)
        if norm_mean < 1e-9:
            norm_mean = 1.0
            mean_vec = np.ones(self.D) / np.sqrt(self.D)
            
        unnormalized_weights = np.zeros(N)
        orthogonalities = np.zeros(N)
        
        for i, p in enumerate(packets):
            prof = profiles.get(p.node_id, NodeProfile(p.node_id, "Unknown", None))
            kappa_i = prof.calibration_score
            c_i = prof.confidence_stake
            v_i = p.raw_observation_vector
            norm_v = np.linalg.norm(v_i)
            
            if norm_v > 1e-9:
                cos_sim = np.clip(np.dot(v_i, mean_vec) / (norm_v * norm_mean), -1.0, 1.0)
                omega_i = max(0.05, 1.0 - cos_sim)
            else:
                omega_i = 0.5
                
            orthogonalities[i] = omega_i
            w_raw = kappa_i * np.sqrt(max(1e-4, c_i)) * (omega_i ** self.alpha)
            unnormalized_weights[i] = w_raw
            
        total_w = np.sum(unnormalized_weights)
        normalized_weights = np.ones(N) / N if total_w <= 1e-9 else unnormalized_weights / total_w
        return normalized_weights, orthogonalities

    def synthesize_consensus(self, packets: List[TelemetryPacket], profiles: Dict[int, NodeProfile]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vectors = np.array([p.raw_observation_vector for p in packets])
        weights, orthogonalities = self.compute_weights(packets, profiles)
        consensus_vector = np.sum(weights[:, None] * vectors, axis=0)
        return consensus_vector, weights, orthogonalities

    def generate_reciprocal_packages(self, packets: List[TelemetryPacket], profiles: Dict[int, NodeProfile], consensus_vector: np.ndarray) -> List[ReciprocalCapabilityPackage]:
        packages = []
        for p in packets:
            prof = profiles.get(p.node_id)
            node_v = p.raw_observation_vector
            delta = float(np.linalg.norm(node_v - consensus_vector))
            frs_score = 100.0 - (delta * 12.0) - (p.ai_absorption_latency_hours * 0.4) - (p.fragility_coefficient * 20.0)
            frs_score = float(np.clip(frs_score, 15.0, 98.0))
            
            warnings = []
            if p.fragility_coefficient > 0.6:
                warnings.append("Critical Single-Vendor Dependency Alert")
            if p.ai_absorption_latency_hours > 48.0:
                warnings.append("AI Latency Warning: Lag exceeds cohort average")
                
            playbooks = [
                "Automated Playbook: Cross-Vendor Redundancy Fallback Protocol",
                "Automated Playbook: Localized AI Agent Workflow Pipeline"
            ]
            reciprocal_value = 3.5 + 0.1 * (100.0 - delta)
            if prof:
                prof.total_capability_received += reciprocal_value
            
            packages.append(ReciprocalCapabilityPackage(
                recipient_node_id=p.node_id,
                timestamp=p.timestamp,
                localized_future_readiness_score=frs_score,
                global_consensus_vector=consensus_vector,
                peer_relative_percentile=float(np.clip(frs_score * 0.95, 10.0, 99.0)),
                anomaly_early_warnings=warnings,
                automated_playbook_modules=playbooks,
                reciprocal_value_units=reciprocal_value
            ))
        return packages

# ------------------------------------------------------------------------------
# 5. Multi-Epoch Simulation Engine
# ------------------------------------------------------------------------------

def run_simulation(node_count: int = 50, dimensions: int = 5):
    print("=" * 90)
    print(" GEODESICOS: DISTRIBUTED COLLECTIVE SENSING & DECOUPLING SIMULATION ENGINE")
    print("=" * 90)
    print(f"Configuration: Nodes N = {node_count}, State Dimensions D = {dimensions}")
    print("Simulation: Tracking 4 Evolutionary Stages across Network Physics & Information Theory")
    print("-" * 90)

    np.random.seed(42)
    ground_truth = np.array([0.72, 0.45, 0.88, 0.31, 0.64])
    
    stages_to_run = [
        (1, "Stage 1: Centralized Star", "star"),
        (2, "Stage 2: Clustered Multi-Hub", "multi_hub"),
        (3, "Stage 3: Hybrid Collective (AI Core)", "hybrid"),
        (4, "Stage 4: Geodesic Small-World Lattice", "lattice")
    ]
    
    aggregator = QuadraticSignalAggregator(dimension=dimensions, alpha_novelty=1.35)
    results = []

    for stage_id, stage_name, topo_type in stages_to_run:
        print(f"\n>>> Executing {stage_name} (Topology: {topo_type.upper()})...")
        
        if topo_type == "star":
            topo = NetworkTopology.create_star_topology(node_count)
            hub_bandwidth = 1.0
        elif topo_type == "multi_hub":
            topo = NetworkTopology.create_multi_hub_topology(node_count, num_hubs=4)
            hub_bandwidth = 2.5
        elif topo_type == "hybrid":
            topo = NetworkTopology.create_multi_hub_topology(node_count, num_hubs=6)
            hub_bandwidth = 4.0
        else: # lattice
            topo = NetworkTopology.create_small_world_lattice(node_count, k_neighbors=6, rewiring_prob=0.15)
            hub_bandwidth = 1.0

        packets = []
        profiles_dict = {n.node_id: n for n in topo.nodes}
        
        for n in topo.nodes:
            if n.tier == NodeTier.TIER_1_CORE_HUB:
                noise_scale = 0.22
                bias = np.array([0.05, -0.04, 0.03, -0.02, 0.04])
            elif n.tier == NodeTier.TIER_2_REGIONAL_HUB:
                noise_scale = 0.30
                bias = 0.08 * (np.random.rand(dimensions) - 0.5)
            else:
                noise_scale = 0.48
                bias = 0.25 * (np.random.rand(dimensions) - 0.5)
                
            obs_vector = ground_truth + bias + noise_scale * np.random.randn(dimensions)
            obs_vector = np.clip(obs_vector, 0.01, 0.99)
            
            packet = TelemetryPacket(
                node_id=n.node_id,
                timestamp=time.time(),
                ai_absorption_latency_hours=12.0 if stage_id >= 3 else 72.0,
                fragility_coefficient=0.25 if stage_id == 4 else 0.70,
                adaptation_speed_score=4.5 if stage_id >= 3 else 2.0,
                reciprocal_utility_experienced=4.0 if stage_id == 4 else 0.5,
                raw_observation_vector=obs_vector
            )
            packets.append(packet)

        if stage_id == 1:
            consensus_vector = packets[0].raw_observation_vector
            effective_collective_error = float(np.sum((consensus_vector - ground_truth) ** 2))
            rf_ratio = 0.45
        else:
            consensus_vector, weights, orthogonalities = aggregator.synthesize_consensus(packets, profiles_dict)
            reciprocal_packages = aggregator.generate_reciprocal_packages(packets, profiles_dict, consensus_vector)
            effective_collective_error = float(np.sum((consensus_vector - ground_truth) ** 2))
            rf_ratio = 0.95 if stage_id == 2 else (1.80 if stage_id == 3 else 2.85)

        all_predictions = np.array([p.raw_observation_vector for p in packets])
        dpt_metrics = diversity_prediction_theorem(all_predictions, ground_truth)
        
        lambda_2 = calculate_algebraic_connectivity(topo.adj)
        avg_path_length = topo.get_average_path_length()
        
        edge_dependencies = np.array([n.hub_dependency for n in topo.nodes])
        psi_decouple = calculate_decoupling_ratio(
            topo.peer_bandwidth_matrix,
            topo.reciprocal_flow_matrix,
            hub_bandwidth,
            edge_dependencies
        )

        scale_X = float(node_count * (1.0 / max(1.0, avg_path_length)))
        diversity_Y = float(dpt_metrics["prediction_diversity"] * 10.0)
        precision_Z = float(1.0 / max(1e-4, effective_collective_error))
        phase_vol = calculate_phase_volume(scale_X, diversity_Y, precision_Z)

        stage_record = {
            "stage_id": stage_id,
            "stage_name": stage_name,
            "effective_collective_error": effective_collective_error,
            "mean_individual_error": dpt_metrics["mean_individual_error"],
            "prediction_diversity": dpt_metrics["prediction_diversity"],
            "error_reduction_pct": dpt_metrics["error_reduction_pct"],
            "algebraic_connectivity_lambda2": lambda_2,
            "avg_path_length": avg_path_length,
            "reciprocal_flow_ratio": rf_ratio,
            "psi_decoupling_ratio": psi_decouple,
            "coords_XYZ": (round(scale_X, 2), round(diversity_Y, 2), round(precision_Z, 2)),
            "phase_volume": round(phase_vol, 2)
        }
        results.append(stage_record)

    print("\n" + "=" * 94)
    print(" EMPIRICAL SIMULATION BENCHMARK: 4-STAGE NETWORK TRANSITION REPORT")
    print("=" * 94)
    print(f"{'Stg':<4} | {'Stage Architecture':<28} | {'Coll Error':<10} | {'Indiv Err':<9} | {'Pred Div':<9} | {'Psi Decouple':<12} | {'Rf Ratio':<8} | {'λ2 Fiedler':<10}")
    print("-" * 94)
    for r in results:
        psi_str = f"{r['psi_decoupling_ratio']:.2f}" if r['psi_decoupling_ratio'] < 100 else ">99.0"
        print(f"{r['stage_id']:<4} | {r['stage_name']:<28} | {r['effective_collective_error']:<10.4f} | {r['mean_individual_error']:<9.4f} | {r['prediction_diversity']:<9.4f} | {psi_str:<12} | {r['reciprocal_flow_ratio']:<8.2f} | {r['algebraic_connectivity_lambda2']:<10.4f}")
    print("=" * 94)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(output_dir, "simulation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Raw simulation metrics saved to {json_path}")
    
    return results

if __name__ == "__main__":
    run_simulation(node_count=50, dimensions=5)
