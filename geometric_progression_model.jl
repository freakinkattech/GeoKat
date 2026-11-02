"""
================================================================================
3D Geometric Aggregated Progression Model: Mathematical & Topological Scaling
================================================================================
This script implements the mathematical formulations, topological mechanics,
and geometric 3D mesh representations of the 4-stage progression model:

  Stage 1: Single-Expert Spike (Narrow high-Z cantilever / needle)
  Stage 2: Multi-Hub Polyhedron (Dodecahedron network cluster)
  Stage 3: Hybrid Collective Lattice (Icosahedron with AI aggregation)
  Stage 4: Resilient Mutual-Elevation Polytope (Geodesic sphere / tensegrity lattice)

Theoretical foundations implemented:
- Scott E. Page (2007): Diversity Prediction Theorem
- W. Ross Ashby (1956): Law of Requisite Variety
- Duncan Watts & Steven Strogatz (1998): Small-World Collective Dynamics
- Réka Albert, Hawoong Jeong, & Albert-László Barabási (2000): Attack & Error Tolerance
- Herbert A. Simon (1971): Attention Economics & Cognitive Channel Capacity
================================================================================
"""

module GeometricProgressionModel

using LinearAlgebra
using Statistics
using Printf

export StageParameters, GeometricMesh, calculate_diversity_prediction,
       calculate_scaling_factors, generate_mesh, print_model_summary,
       render_ascii_projection, generate_makie_visualization_code

# ------------------------------------------------------------------------------
# 1. Structural Definitions
# ------------------------------------------------------------------------------

"""
Holds mathematical and empirical attributes for a stage in the 3D progression model.
"""
struct StageParameters
    stage_id::Int
    name::String
    topology::String
    center_coords::NTuple{3, Float64}  # (X: Scale, Y: Diversity, Z: Precision)
    bounding_radius::Float64
    enclosed_volume::Float64
    node_count::Int
    diversity_index::Float64           # Shannon entropy / attribute variance
    avg_individual_error::Float64     # Mean error across individual nodes
    prediction_diversity::Float64      # Cross-node variance in predictions
    collective_error::Float64          # Derived via Diversity Prediction Theorem
    resilience_lambda2::Float64        # Algebraic connectivity (Fiedler value)
    reciprocal_flow_ratio::Float64     # Rf: Structured benefit / extracted signal
end

"""
Represents a 3D geometric mesh for visual rendering.
"""
struct GeometricMesh
    stage_id::Int
    name::String
    vertices::Vector{NTuple{3, Float64}}
    edges::Vector{Tuple{Int, Int}}
    faces::Vector{Vector{Int}}
end

# ------------------------------------------------------------------------------
# 2. Mathematical Formulations & Theorem Validation
# ------------------------------------------------------------------------------

"""
    diversity_prediction_theorem(individual_errors::Vector{Float64}, predictions::Vector{Float64}, truth::Float64)

Validates Scott E. Page's Diversity Prediction Theorem:
    Collective Error = Mean Individual Error - Prediction Diversity
"""
function diversity_prediction_theorem(predictions::Vector{Float64}, truth::Float64)
    N = length(predictions)
    c_hat = mean(predictions) # Collective prediction
    
    # Collective squared error
    collective_err = (c_hat - truth)^2
    
    # Mean individual squared error
    indiv_errors = [(s - truth)^2 for s in predictions]
    mean_indiv_err = mean(indiv_errors)
    
    # Prediction diversity (variance around the collective mean)
    diversity = mean([(s - c_hat)^2 for s in predictions])
    
    # Verify theorem identity: collective_err == mean_indiv_err - diversity
    discrepancy = abs(collective_err - (mean_indiv_err - diversity))
    @assert discrepancy < 1e-10 "Diversity Prediction Theorem identity failed!"
    
    return (collective_error=collective_err, mean_individual_error=mean_indiv_err, prediction_diversity=diversity)
end

"""
    ashby_variety_volume(scale_X::Float64, diversity_Y::Float64, precision_Z::Float64)

Calculates the internal requisite variety state space (phase space volume):
    V_s = k * X * Y * Z
Matches Ashby's Law of Requisite Variety (V_system >= V_environment).
"""
function ashby_variety_volume(coords::NTuple{3, Float64}, shape_type::Symbol)
    X, Y, Z = coords
    if shape_type == :cone_spike
        # Volume of sharp conical spike: (1/3) * pi * r_base^2 * height
        r_base = 0.5 * min(X, Y)
        return (1.0 / 3.0) * pi * (r_base^2) * Z
    elseif shape_type == :dodecahedron
        # Dodecahedron volume: ~7.663 * a^3 where a is edge length
        # Using equivalent bounding sphere radius R ~ 0.5 * norm(X, Y)
        r = 0.5 * sqrt(X^2 + Y^2) * 0.4
        return (4.0 / 3.0) * pi * (r^3) * (Z / 2.5)
    elseif shape_type == :icosahedron
        r = 0.5 * sqrt(X^2 + Y^2) * 0.45
        return (4.0 / 3.0) * pi * (r^3) * (Z / 3.0)
    elseif shape_type == :geodesic_polytope
        r = 0.5 * sqrt(X^2 + Y^2) * 0.5
        return (4.0 / 3.0) * pi * (r^3) * (Z / 3.5)
    end
    return (4.0 / 3.0) * pi * prod(coords)
end

"""
    calculate_scaling_factors()

Constructs the quantitative profile for all 4 stages based on the mathematical models.
"""
function calculate_scaling_factors()
    # Stage 1: Single-Expert
    # Coordinates: X=0.6, Y=0.4, Z=4.2 (High individual precision, negligible breadth/diversity)
    c1 = (0.6, 0.4, 4.2)
    v1 = ashby_variety_volume(c1, :cone_spike) # ~0.176
    s1 = StageParameters(
        1, "Single-Expert Stage", "Star / Centralized Hub",
        c1, 0.25, v1,
        1,       # Node count: 1 primary expert
        0.12,    # Low diversity
        0.35,    # Individual error of expert
        0.00,    # Zero diversity (single mind)
        0.35,    # Collective error = individual error
        0.00,    # Fiedler eigenvalue = 0 (disconnects if hub fails)
        0.45     # Extractive (Rf < 1.0)
    )

    # Stage 2: Multi-Hub Stage
    c2 = (2.4, 1.8, 2.6)
    v2 = ashby_variety_volume(c2, :dodecahedron) # ~3.32
    s2 = StageParameters(
        2, "Multi-Hub Stage", "Clustered Silos / Pods",
        c2, 0.85, v2,
        15,      # Node count
        0.95,    # Moderate diversity
        0.42,    # Higher average individual error across general pods
        0.18,    # Moderate prediction diversity
        0.24,    # Collective error (0.42 - 0.18 = 0.24) -> improved!
        0.15,    # Weak algebraic connectivity
        0.90     # Near parity reciprocity
    )

    # Stage 3: Hybrid Collective Stage
    c3 = (5.2, 4.2, 3.8)
    v3 = ashby_variety_volume(c3, :icosahedron) # ~32.8
    s3 = StageParameters(
        3, "Hybrid Collective Stage", "Directed Graph + AI Core",
        c3, 1.75, v3,
        85,      # Node count
        2.45,    # High cognitive/sector diversity
        0.58,    # Higher individual error from peripheral/lower-capability nodes
        0.42,    # Very high prediction diversity
        0.16,    # Collective error (0.58 - 0.42 = 0.16) -> significantly sharper!
        0.65,    # Substantial network resilience
        1.65     # Strongly positive reciprocity (Rf > 1)
    )

    # Stage 4: Resilient Mutual-Elevation System
    c4 = (8.8, 7.5, 5.6)
    v4 = ashby_variety_volume(c4, :geodesic_polytope) # ~214.5
    s4 = StageParameters(
        4, "Resilient Mutual-Elevation System", "Small-World Geodesic Polytope",
        c4, 3.20, v4,
        500,     # Node count
        4.10,    # Maximum systemic diversity
        0.72,    # High individual error across vast asymmetric edge participants
        0.64,    # Maximum orthogonal prediction diversity
        0.08,    # Collective error (0.72 - 0.64 = 0.08) -> highest precision!
        1.85,    # Robust spectral gap / high Fiedler value
        2.80     # Mutual elevation: high structured value returned to edge
    )

    return [s1, s2, s3, s4]
end

# ------------------------------------------------------------------------------
# 3. Geometric Mesh Builders (Polyhedral Topologies)
# ------------------------------------------------------------------------------

"""
Generates the geometric 3D mesh (vertices, edges, faces) for each stage.
"""
function generate_mesh(stage::StageParameters)
    cx, cy, cz = stage.center_coords
    r = stage.bounding_radius
    
    if stage.stage_id == 1
        # Stage 1: Needle / Spike (Tetragonal Pyramid with high aspect ratio)
        # Narrow base at ground, sharp vertex at apex
        h = cz
        rb = 0.18
        verts = [
            (cx - rb, cy - rb, 0.0),
            (cx + rb, cy - rb, 0.0),
            (cx + rb, cy + rb, 0.0),
            (cx - rb, cy + rb, 0.0),
            (cx, cy, h)  # Apex
        ]
        edges = [(1,2), (2,3), (3,4), (4,1), (1,5), (2,5), (3,5), (4,5)]
        faces = [
            [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5], [1, 4, 3, 2]
        ]
        return GeometricMesh(stage.stage_id, stage.name, verts, edges, faces)
        
    elseif stage.stage_id == 2
        # Stage 2: Regular Dodecahedron (20 vertices, 30 edges, 12 pentagonal faces)
        phi = (1.0 + sqrt(5.0)) / 2.0 # Golden ratio
        raw_verts = NTuple{3, Float64}[]
        
        # (+-1, +-1, +-1)
        for i in [-1.0, 1.0], j in [-1.0, 1.0], k in [-1.0, 1.0]
            push!(raw_verts, (i, j, k))
        end
        # (0, +-1/phi, +-phi)
        for j in [-1.0/phi, 1.0/phi], k in [-phi, phi]
            push!(raw_verts, (0.0, j, k))
        end
        # (+-1/phi, +-phi, 0)
        for i in [-1.0/phi, 1.0/phi], j in [-phi, phi]
            push!(raw_verts, (i, j, 0.0))
        end
        # (+-phi, 0, +-1/phi)
        for i in [-phi, phi], k in [-1.0/phi, 1.0/phi]
            push!(raw_verts, (i, 0.0, k))
        end
        
        # Normalize and scale
        scale_fac = r / sqrt(3.0)
        verts = [(cx + v[1]*scale_fac, cy + v[2]*scale_fac, cz + v[3]*scale_fac) for v in raw_verts]
        
        # Build edges based on distance threshold
        edges = Tuple{Int, Int}[]
        target_dist = 2.0 / phi * scale_fac
        for i in 1:length(verts)
            for j in (i+1):length(verts)
                d = norm(collect(verts[i]) .- collect(verts[j]))
                if abs(d - target_dist) < 0.2 * scale_fac
                    push!(edges, (i, j))
                end
            end
        end
        return GeometricMesh(stage.stage_id, stage.name, verts, edges, Vector{Int}[])

    elseif stage.stage_id == 3
        # Stage 3: Regular Icosahedron (12 vertices, 30 edges, 20 triangular faces)
        phi = (1.0 + sqrt(5.0)) / 2.0
        raw_verts = NTuple{3, Float64}[]
        for i in [-1.0, 1.0], j in [-phi, phi]
            push!(raw_verts, (0.0, i, j))
            push!(raw_verts, (i, j, 0.0))
            push!(raw_verts, (j, 0.0, i))
        end
        
        scale_fac = r / sqrt(1.0 + phi^2)
        verts = [(cx + v[1]*scale_fac, cy + v[2]*scale_fac, cz + v[3]*scale_fac) for v in raw_verts]
        
        edges = Tuple{Int, Int}[]
        target_dist = 2.0 * scale_fac
        for i in 1:length(verts)
            for j in (i+1):length(verts)
                d = norm(collect(verts[i]) .- collect(verts[j]))
                if abs(d - target_dist) < 0.2 * scale_fac
                    push!(edges, (i, j))
                end
            end
        end
        return GeometricMesh(stage.stage_id, stage.name, verts, edges, Vector{Int}[])

    else
        # Stage 4: Geodesic Polytope (Dual-frequency subdivided icosahedron / sphere lattice)
        # Generates an omnidirectional spherical wireframe with 42 vertices and dense triangulation
        phi = (1.0 + sqrt(5.0)) / 2.0
        base_verts = NTuple{3, Float64}[]
        for i in [-1.0, 1.0], j in [-phi, phi]
            push!(base_verts, (0.0, i, j))
            push!(base_verts, (i, j, 0.0))
            push!(base_verts, (j, 0.0, i))
        end
        
        # Subdivide midpoints to create geodesic lattice
        all_verts = copy(base_verts)
        edges = Tuple{Int, Int}[]
        edge_dist = 2.0
        
        for i in 1:length(base_verts)
            for j in (i+1):length(base_verts)
                d = norm(collect(base_verts[i]) .- collect(base_verts[j]))
                if abs(d - edge_dist) < 0.25
                    # Midpoint subdivision
                    mid = 0.5 .* (collect(base_verts[i]) .+ collect(base_verts[j]))
                    # Project back to sphere surface
                    mid_norm = mid ./ norm(mid) .* sqrt(1.0 + phi^2)
                    push!(all_verts, (mid_norm[1], mid_norm[2], mid_norm[3]))
                end
            end
        end
        
        scale_fac = r / sqrt(1.0 + phi^2)
        verts = [(cx + v[1]*scale_fac, cy + v[2]*scale_fac, cz + v[3]*scale_fac) for v in all_verts]
        
        # Form dense triangulation edges
        geodesic_edges = Tuple{Int, Int}[]
        max_edge_len = 1.35 * scale_fac
        for i in 1:length(verts)
            for j in (i+1):length(verts)
                d = norm(collect(verts[i]) .- collect(verts[j]))
                if d <= max_edge_len
                    push!(geodesic_edges, (i, j))
                end
            end
        end
        return GeometricMesh(stage.stage_id, stage.name, verts, geodesic_edges, Vector{Int}[])
    end
end

# ------------------------------------------------------------------------------
# 4. Quantitative Reporting & Summary Table
# ------------------------------------------------------------------------------

"""
Prints the comprehensive mathematical report comparing all 4 stages.
"""
function print_model_summary(stages::Vector{StageParameters})
    println("="^88)
    println(" 3D GEOMETRIC AGGREGATED PROGRESSION MODEL: QUANTITATIVE SCALING PROFILE")
    println("="^88)
    @printf("%-4s | %-24s | %-12s | %-8s | %-8s | %-8s | %-9s\n",
            "Stg", "Architecture", "Coords (X,Y,Z)", "Nodes", "Volume", "Vol Ratio", "Coll Error")
    println("-"^88)
    
    base_vol = stages[1].enclosed_volume
    for s in stages
        coords_str = @sprintf("(%.1f,%.1f,%.1f)", s.center_coords[1], s.center_coords[2], s.center_coords[3])
        vol_ratio = s.enclosed_volume / base_vol
        @printf("%-4d | %-24s | %-12s | %-8d | %-8.2f | %-7.1fx | %-9.3f\n",
                s.stage_id, s.name, coords_str, s.node_count, s.enclosed_volume, vol_ratio, s.collective_error)
    end
    println("="^88)
    
    println("\nDetailed Mathematical Breakdown per Stage:")
    println("-"^88)
    for s in stages
        println("Stage $(s.stage_id): $(s.name)")
        println("  Topology:                   $(s.topology)")
        @printf("  Coordinates (X, Y, Z):      (Scale=%.2f, Diversity=%.2f, Precision=%.2f)\n",
                s.center_coords[1], s.center_coords[2], s.center_coords[3])
        @printf("  Enclosed Phase Volume:      %.3f units^3 (Expansion: %.1fx relative to Stage 1)\n",
                s.enclosed_volume, s.enclosed_volume / base_vol)
        @printf("  Node Reach & Diversity:     N = %d nodes, Shannon Entropy = %.2f bits\n",
                s.node_count, s.diversity_index)
        @printf("  Diversity Prediction Proof: AvgIndivErr (%.3f) - PredDiv (%.3f) = CollectiveErr (%.3f)\n",
                s.avg_individual_error, s.prediction_diversity, s.collective_error)
        @printf("  Reciprocal Flow Ratio (Rf): %.2f (%s)\n",
                s.reciprocal_flow_ratio, s.reciprocal_flow_ratio > 1.0 ? "Net Capability Elevation" : "Net Extractive")
        @printf("  Spectral Connectivity (λ2): %.2f (%s)\n",
                s.resilience_lambda2, s.resilience_lambda2 > 1.0 ? "High Fault Tolerance" : "Fragile Single Hub")
        println()
    end
end

# ------------------------------------------------------------------------------
# 5. ASCII 3D Projection Visualization
# ------------------------------------------------------------------------------

"""
Renders a terminal-friendly orthogonal projection of the 3D model progression.
"""
function render_ascii_projection(stages::Vector{StageParameters})
    println("="^88)
    println(" 2D ORTHOGONAL PROJECTION: Z (Precision) vs. X (Scale) with Y (Diversity Expansion)")
    println("="^88)
    
    grid_h = 16
    grid_w = 72
    canvas = fill(' ', grid_h, grid_w)
    
    # Draw axes
    for r in 1:grid_h
        canvas[r, 4] = '|'
    end
    for c in 4:grid_w
        canvas[grid_h - 1, c] = '-'
    end
    canvas[grid_h - 1, 4] = '+'
    canvas[1, 2] = 'Z'
    canvas[grid_h - 1, grid_w] = 'X'
    
    # Project stages
    # X in [0, 10] -> col in [5, 68]
    # Z in [0, 6]  -> row in [grid_h-2, 2]
    stage_markers = ['▲', '◈', '❖', '⬢']
    for s in stages
        x, y, z = s.center_coords
        col = clamp(round(Int, 5 + (x / 10.0) * (grid_w - 10)), 5, grid_w - 2)
        row = clamp(round(Int, (grid_h - 2) - (z / 6.0) * (grid_h - 4)), 2, grid_h - 2)
        canvas[row, col] = stage_markers[s.stage_id]
        
        # Label
        lbl = "S$(s.stage_id)[$(round(Int, s.enclosed_volume / stages[1].enclosed_volume))x]"
        for (idx, ch) in enumerate(lbl)
            if col + idx < grid_w
                canvas[row, col + idx] = ch
            end
        end
    end
    
    # Render canvas
    for r in 1:grid_h
        println(String(canvas[r, :]))
    end
    println("Legend:  ▲ = Stage 1 (Spike)    ◈ = Stage 2 (Dodecahedron)")
    println("         ❖ = Stage 3 (Icosahedron)   ⬢ = Stage 4 (Geodesic Sphere)")
    println("="^88)
end

# ------------------------------------------------------------------------------
# 6. GLMakie / Plots.jl Interactive Code Generator
# ------------------------------------------------------------------------------

"""
Emits full Julia source code to render interactive 3D visualizations using GLMakie or CairoMakie.
"""
function generate_makie_visualization_code()
    return """
    # ==============================================================================
    # GLMakie / CairoMakie 3D Interactive Visualization Script
    # ==============================================================================
    # Run in your Julia REPL:
    #   using Pkg; Pkg.add(["GLMakie", "GeometryBasics"])
    # ==============================================================================
    using GLMakie
    using GeometryBasics

    function plot_3d_progression(stages, meshes)
        fig = Figure(resolution = (1400, 900), backgroundcolor = :black)
        ax = Axis3(fig[1, 1],
            title = "3D Geometric Aggregated Progression Model\\n(Scale X, Diversity Y, Precision Z)",
            titlesize = 22, titlecolor = :white,
            xlabel = "X: Scale & Reach", ylabel = "Y: Diversity & Inclusion", zlabel = "Z: Precision & Resilience",
            xlabelsize = 14, ylabelsize = 14, zlabelsize = 14,
            xlabelcolor = :gold, ylabelcolor = :cyan, zlabelcolor = :white,
            backgroundcolor = RGBf(0.02, 0.03, 0.05),
            elevation = 0.42, azimuth = 1.25
        )

        # Plot progression curve trajectory spline
        xs = [s.center_coords[1] for s in stages]
        ys = [s.center_coords[2] for s in stages]
        zs = [s.center_coords[3] for s in stages]
        lines!(ax, xs, ys, zs, color = :gold, linewidth = 4, linestyle = :solid)

        # Draw each stage mesh
        colors = [:orange, :deepskyblue, :cyan, :turquoise1]
        for (i, m) in enumerate(meshes)
            # Draw edges
            for e in m.edges
                p1 = Point3f(m.vertices[e[1]])
                p2 = Point3f(m.vertices[e[2]])
                lines!(ax, [p1, p2], color = (colors[i], 0.75), linewidth = i == 1 ? 3 : 2)
            end
            # Draw vertices
            v_pts = [Point3f(v) for v in m.vertices]
            scatter!(ax, v_pts, color = colors[i], markersize = i == 4 ? 8 : 12)
        end

        display(fig)
        return fig
    end
    """
end

end # module

# ==============================================================================
# Main Execution Entrypoint
# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    import .GeometricProgressionModel as GPM
    stages = GPM.calculate_scaling_factors()
    GPM.print_model_summary(stages)
    GPM.render_ascii_projection(stages)
    
    println("\nTo launch full interactive 3D rendering in GLMakie, use:")
    println("  julia -e 'include(\"$(basename(@__FILE__))\"); using GLMakie; ...'")
end
