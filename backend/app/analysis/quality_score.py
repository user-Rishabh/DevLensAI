import os
import json
import re
from datetime import datetime, timezone
from app.db import supabase, LOCAL_FILE_CONTENTS_CACHE, CACHE_DIR, get_repo_analysis, get_file_content
from app.search.chunker import is_excluded_file, detect_language

# Local cache for calculated quality scores across requests
LOCAL_QUALITY_SCORES_CACHE: dict[str, dict] = {}

# Code file extensions recognized for syntactic branching complexity analysis
SUPPORTED_CODE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".rs", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".sh", ".bash", ".scala"
}

# Metric weight configuration for Absolute Repository Health (Total = 1.0)
WEIGHT_COMPLEXITY = 0.25
WEIGHT_FILE_SIZE = 0.20
WEIGHT_DUPLICATION = 0.15
WEIGHT_COUPLING = 0.20
WEIGHT_CHURN = 0.20

# Decision predicates in McCabe cyclomatic complexity:
# Measures independent decision branching points.
# Note: 'else' and 'switch' are intentionally NOT included as decision predicates
# because the decision is evaluated at 'if' (else is the alternate path) and 'case' (individual branches).
# Counting 'else' would double-count every if-else construct.
COMPLEXITY_PATTERN = re.compile(
    r'\b(if|elif|for|while|case|catch|except|and|or)\b|&&|\|\||\?|\b(throw|raise)\b', 
    re.IGNORECASE
)


def is_supported_code_file(file_path: str) -> bool:
    """Returns True if the file has a recognizable source code extension."""
    _, ext = os.path.splitext(file_path.lower())
    return ext in SUPPORTED_CODE_EXTENSIONS


def strip_comments_and_strings(code: str, file_path: str = "") -> str:
    """
    Strips comments and string literals across Python, JS/TS, and C-style languages
    to avoid false-positive decision matches in docstrings, log text, or SQL queries.
    """
    if not code:
        return ""
    
    clean = code
    is_python = file_path.lower().endswith(".py")
    
    if is_python:
        # Strip Python triple-quoted docstrings / multiline strings
        clean = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', '', clean)
        # Strip single-line comments (# ...)
        clean = re.sub(r'#.*$', '', clean, flags=re.MULTILINE)
    else:
        # C-style / JS-style block comments /* ... */
        clean = re.sub(r'/\*[\s\S]*?\*/', '', clean)
        # Line comments // ...
        clean = re.sub(r'//.*$', '', clean, flags=re.MULTILINE)
        if file_path.lower().endswith((".sh", ".bash", ".rb", ".yml", ".yaml")):
            clean = re.sub(r'#.*$', '', clean, flags=re.MULTILINE)

    # Strip single, double, and backtick string literals (preserving empty placeholders)
    clean = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`', '""', clean)
    return clean


def compute_file_complexity(content: str, language: str = "", file_path: str = "") -> int:
    """
    Computes a clean McCabe-compliant cyclomatic complexity proxy for a file content
    by counting decision predicates on stripped code (excluding comments and strings).
    """
    if not content:
        return 0
    clean_code = strip_comments_and_strings(content, file_path)
    matches = COMPLEXITY_PATTERN.findall(clean_code)
    return len(matches)


def compute_duplication_ratio(file_data_list: list[dict], min_block_lines: int = 5) -> tuple[float, int, int]:
    """
    Computes code duplication ratio across non-trivial code lines using rolling block hashing.
    Excludes comments, blank lines, import statements, and lone structural punctuation.
    Uses a set of (file_index, line_index) tuples so overlapping ranges are never double-counted.
    Returns (duplication_ratio, duplicate_lines_count, total_non_trivial_lines).
    """
    cleaned_lines_by_file = []
    total_non_trivial_lines = 0

    comment_or_import = re.compile(
        r'^\s*(//|#|/\*|\*|import\s|from\s|require\(|#include|using\s|<!--|package\s)', 
        re.IGNORECASE
    )
    structural_only = re.compile(r'^\s*([\{\}\(\)\[\];,\.]\s*)+$')

    for item in file_data_list:
        content = item.get("content", "") or ""
        lines = content.splitlines()
        file_cleaned = []
        for line in lines:
            trimmed = line.strip()
            if not trimmed or comment_or_import.match(trimmed) or structural_only.match(trimmed):
                continue
            file_cleaned.append(trimmed)
        cleaned_lines_by_file.append(file_cleaned)
        total_non_trivial_lines += len(file_cleaned)

    if total_non_trivial_lines < min_block_lines * 2:
        return 0.0, 0, total_non_trivial_lines

    # Hash rolling blocks of min_block_lines
    block_occurrences = {}
    for f_idx, lines in enumerate(cleaned_lines_by_file):
        if len(lines) < min_block_lines:
            continue
        for i in range(len(lines) - min_block_lines + 1):
            block_key = "\n".join(lines[i:i + min_block_lines])
            block_hash = hash(block_key)
            if block_hash not in block_occurrences:
                block_occurrences[block_hash] = []
            block_occurrences[block_hash].append((f_idx, i))

    duplicate_lines_set = set()
    for block_hash, locs in block_occurrences.items():
        if len(locs) > 1:
            for f_idx, start_line in locs:
                for offset in range(min_block_lines):
                    duplicate_lines_set.add((f_idx, start_line + offset))

    dup_count = len(duplicate_lines_set)
    dup_ratio = min(1.0, dup_count / max(1, total_non_trivial_lines))
    return dup_ratio, dup_count, total_non_trivial_lines


def compute_absolute_repository_health(
    file_data_list: list[dict],
    dependencies: list[dict] = None,
    hotspots: list[dict] = None,
) -> dict:
    """
    Computes an Absolute Repository Health Score (0-100) based on fixed, explainable
    engineering benchmarks, completely independent of relative intra-repo percentiles.
    
    Evaluates 5 dimensions:
    1. Complexity: Decision branching density per 100 LOC (weight 0.25)
    2. File Size & Modularity: Large and monolithic file concentration (weight 0.20)
    3. Duplication: Cross-file code block duplication ratio (weight 0.15)
    4. Coupling: Internal dependency edges per file & fan-in concentration (weight 0.20)
    5. Churn: Git commit concentration in high-complexity hotspot files (weight 0.20)
    """
    total_files = len(file_data_list)
    if total_files == 0:
        return {
            "score": 100,
            "rating": "Healthy Codebase",
            "metric_coverage": 0.0,
            "coverage_details": {
                "dimension_coverage": 0.0,
                "dimension_coverage_percent": 0,
                "files_analyzed": 0,
                "code_files_analyzed": 0,
                "total_loc_analyzed": 0,
                "dimensions_available": [],
                "dimensions_unavailable": ["complexity", "file_size", "duplication", "coupling", "churn"]
            },
            "metrics": {},
            "summary_explanation": "No files available for quality analysis."
        }

    code_files = [f for f in file_data_list if is_supported_code_file(f.get("file_path", ""))]
    total_loc = sum(f.get("loc", 0) for f in file_data_list)
    code_loc = sum(f.get("loc", 0) for f in code_files)

    metrics: dict[str, dict] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Dimension 1: Cyclomatic & Branching Complexity Density (weight 0.25)
    # ──────────────────────────────────────────────────────────────────────────
    if len(code_files) == 0 or code_loc == 0:
        metrics["complexity"] = {
            "score": 75,
            "weight": WEIGHT_COMPLEXITY,
            "available": False,
            "measurement": "0 recognized code files",
            "rating": "Not Measured",
            "reason": "No supported programming language files found to evaluate branching complexity."
        }
    else:
        total_branches = sum(f.get("complexity", 0) for f in code_files)
        branches_per_100_loc = (total_branches / code_loc) * 100.0

        # Benchmarks:
        # <= 2 absolute branches in small repo or <= 3.5 branches / 100 LOC: Clean, linear logic (score 90-100)
        # 3.5 - 7.0: Standard readable business logic (score 70-89)
        # 7.0 - 14.0: High branching density (score 45-69)
        # > 14.0: Extreme cyclomatic spaghetti risk (score 10-44)
        if total_branches <= 2 and code_loc < 100:
            cx_score = 95
        elif branches_per_100_loc <= 3.5:
            cx_score = round(100 - (branches_per_100_loc / 3.5) * 10)
        elif branches_per_100_loc <= 7.0:
            cx_score = round(90 - (branches_per_100_loc - 3.5) * 5.5)
        else:
            cx_score = round(max(10, 70 - (branches_per_100_loc - 7.0) * 4.5))
        cx_score = max(10, min(100, cx_score))

        cx_rating = "Low Complexity" if cx_score >= 80 else "Moderate Complexity" if cx_score >= 60 else "High Complexity"
        metrics["complexity"] = {
            "score": cx_score,
            "weight": WEIGHT_COMPLEXITY,
            "available": True,
            "measurement": f"{branches_per_100_loc:.1f} branches / 100 LOC",
            "rating": cx_rating,
            "reason": (
                f"Evaluated {total_branches} control-flow decision predicates across {code_loc} lines of code "
                f"({branches_per_100_loc:.1f} decision constructs per 100 LOC; comments and string literals excluded)."
            )
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Dimension 2: File Size & Modularity (weight 0.20)
    # ──────────────────────────────────────────────────────────────────────────
    if total_loc == 0:
        metrics["file_size"] = {
            "score": 80,
            "weight": WEIGHT_FILE_SIZE,
            "available": False,
            "measurement": "0 lines of code",
            "rating": "Not Measured",
            "reason": "Files are empty or without measurable line counts."
        }
    else:
        oversized_files = [f for f in file_data_list if f.get("loc", 0) > 350]
        extreme_files = [f for f in file_data_list if f.get("loc", 0) > 750]

        # Complexity-moderated penalty:
        # Avoid treating file size alone as proof of poor modularity.
        # If an oversized file has very low complexity (e.g. declarations/tables), reduce its penalty weight.
        oversized_weighted_loc = 0
        extreme_weighted_loc = 0

        for f in oversized_files:
            f_loc = f.get("loc", 0)
            f_cx = f.get("complexity", 0)
            cx_density = f_cx / max(1, f_loc)
            # Moderation factor: 0.6 for simple files up to 1.4 for complex files
            factor = max(0.6, min(1.4, cx_density / 0.05))
            oversized_weighted_loc += f_loc * factor
            if f_loc > 750:
                extreme_weighted_loc += f_loc * factor

        oversized_ratio = oversized_weighted_loc / total_loc
        extreme_ratio = extreme_weighted_loc / total_loc

        penalty = (oversized_ratio * 45.0) + (extreme_ratio * 55.0)
        size_score = max(15, min(100, round(100.0 - penalty)))

        size_rating = "Modular Structure" if size_score >= 80 else "Acceptable Modularity" if size_score >= 60 else "Monolithic File Risk"
        metrics["file_size"] = {
            "score": size_score,
            "weight": WEIGHT_FILE_SIZE,
            "available": True,
            "measurement": f"{len(oversized_files)} of {total_files} files > 350 LOC ({oversized_ratio*100:.1f}%)",
            "rating": size_rating,
            "reason": (
                f"{len(oversized_files)} files exceed 350 LOC ({len(extreme_files)} exceed 750 LOC). "
                f"Size penalty moderated by function-level decision density."
            )
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Dimension 3: Code Duplication (weight 0.15)
    # ──────────────────────────────────────────────────────────────────────────
    dup_ratio, dup_lines, total_clean_lines = compute_duplication_ratio(file_data_list, min_block_lines=5)
    if total_clean_lines < 20:
        metrics["duplication"] = {
            "score": 85,
            "weight": WEIGHT_DUPLICATION,
            "available": False,
            "measurement": "Insufficient code lines",
            "rating": "Not Measured",
            "reason": "Repository does not have enough non-trivial lines to evaluate block duplication."
        }
    else:
        # Benchmarks: < 3% duplication is excellent; > 15% is heavy technical debt
        dup_score = max(10, min(100, round(100.0 - (dup_ratio * 380.0))))
        dup_rating = "Minimal Duplication" if dup_score >= 80 else "Moderate Duplication" if dup_score >= 60 else "High Duplication"
        metrics["duplication"] = {
            "score": dup_score,
            "weight": WEIGHT_DUPLICATION,
            "available": True,
            "measurement": f"{dup_ratio*100:.1f}% duplicate blocks",
            "rating": dup_rating,
            "reason": (
                f"{dup_lines} of {total_clean_lines} non-trivial lines ({dup_ratio*100:.1f}%) belong to "
                f"repeated 5-line structural code sequences (no overlapping range double-counting)."
            )
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Dimension 4: Dependency Coupling & Cohesion (weight 0.20)
    # ──────────────────────────────────────────────────────────────────────────
    if dependencies is None:
        metrics["coupling"] = {
            "score": 75,
            "weight": WEIGHT_COUPLING,
            "available": False,
            "measurement": "No dependency data",
            "rating": "Not Measured",
            "reason": "Dependency graph was not extracted or unsupported for this repository."
        }
    elif total_files <= 2 and len(dependencies) == 0:
        metrics["coupling"] = {
            "score": 80,
            "weight": WEIGHT_COUPLING,
            "available": False,
            "measurement": "Single or dual file project",
            "rating": "Not Measured",
            "reason": "Repository has 2 or fewer files; internal import coupling is not applicable."
        }
    else:
        # Deduplicate edges and exclude self-dependencies
        unique_edges = set()
        for edge in dependencies:
            src = edge.get("from") or edge.get("source") or ""
            tgt = edge.get("to") or edge.get("target") or ""
            if src and tgt and src != tgt:
                unique_edges.add((src, tgt))

        edge_count = len(unique_edges)
        avg_imports_per_file = edge_count / max(1, total_files)

        # In-degree fan-in check: God-module detection
        in_degrees = {}
        for src, tgt in unique_edges:
            in_degrees[tgt] = in_degrees.get(tgt, 0) + 1

        max_in_degree = max(in_degrees.values()) if in_degrees else 0
        fan_in_ratio = max_in_degree / max(1, total_files)

        # Benchmarks: 0.5 to 2.5 imports/file is clean modular coupling (score 85-100)
        # > 4.5 imports/file indicates tightly coupled spaghetti (score < 60)
        if avg_imports_per_file <= 2.5:
            coupling_score = round(100.0 - max(0.0, (avg_imports_per_file - 1.0) * 10.0))
        else:
            coupling_score = round(max(20.0, 85.0 - (avg_imports_per_file - 2.5) * 15.0))

        # God-module penalty if a single file has extreme fan-in (> 40% of all files)
        if fan_in_ratio > 0.40 and total_files >= 5:
            coupling_score = max(15, coupling_score - 15)

        coupling_score = max(15, min(100, coupling_score))
        coupling_rating = "Loosely Coupled" if coupling_score >= 80 else "Moderate Coupling" if coupling_score >= 60 else "Tightly Coupled"

        metrics["coupling"] = {
            "score": coupling_score,
            "weight": WEIGHT_COUPLING,
            "available": True,
            "measurement": f"{avg_imports_per_file:.1f} imports / file ({edge_count} edges)",
            "rating": coupling_rating,
            "reason": (
                f"Mapped {edge_count} internal import dependencies across {total_files} files "
                f"(average {avg_imports_per_file:.1f} imports/file, max fan-in: {max_in_degree}; self-loops excluded)."
            )
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Dimension 5: Git Churn & Hotspot Concentration Risk (weight 0.20)
    # ──────────────────────────────────────────────────────────────────────────
    if not hotspots or len(hotspots) == 0:
        metrics["churn"] = {
            "score": 75,
            "weight": WEIGHT_CHURN,
            "available": False,
            "measurement": "No git history",
            "rating": "Not Measured",
            "reason": "Git commit history is not available (e.g. shallow clone or fresh commit)."
        }
    else:
        total_commits = sum(h.get("commit_count", 0) for h in hotspots)
        if total_commits == 0:
            metrics["churn"] = {
                "score": 75,
                "weight": WEIGHT_CHURN,
                "available": False,
                "measurement": "0 commits recorded",
                "rating": "Not Measured",
                "reason": "No commit frequency recorded in git hotspots."
            }
        else:
            # Check the "Hotspot Anti-Pattern": commits landing in oversized or complex files
            complex_files_set = {
                f["file_path"] for f in file_data_list 
                if f.get("loc", 0) > 350 or f.get("complexity", 0) > 20
            }
            risky_commits = sum(
                h.get("commit_count", 0) for h in hotspots 
                if h.get("path") in complex_files_set
            )
            risky_churn_ratio = risky_commits / max(1, total_commits)

            # Benchmarks: If commits are spread across modular files -> low risk (score 85-100)
            # If > 50% of commits hit monster complex files -> high risk (score 30-55)
            churn_score = max(20, min(100, round(100.0 - (risky_churn_ratio * 70.0))))
            churn_rating = "Low Churn Risk" if churn_score >= 80 else "Moderate Churn Risk" if churn_score >= 60 else "High Churn Risk"

            metrics["churn"] = {
                "score": churn_score,
                "weight": WEIGHT_CHURN,
                "available": True,
                "measurement": f"{risky_churn_ratio*100:.1f}% commits in complex files",
                "rating": churn_rating,
                "reason": (
                    f"{risky_commits} of {total_commits} analyzed commits ({risky_churn_ratio*100:.1f}%) "
                    f"targeted high-complexity or oversized hotspot files."
                )
            }

    # ──────────────────────────────────────────────────────────────────────────
    # Weighted Aggregation & Small-Repository Calibration
    # ──────────────────────────────────────────────────────────────────────────
    available_metrics = {k: v for k, v in metrics.items() if v["available"]}
    available_weight = sum(v["weight"] for v in available_metrics.values())
    metric_coverage = round(available_weight, 2)

    if available_weight > 0:
        weighted_score_sum = sum(v["score"] * v["weight"] for v in available_metrics.values())
        raw_score = weighted_score_sum / available_weight
    else:
        raw_score = 75.0

    # Small-repository sample size confidence adjustment:
    # A tiny 1-2 file project cannot be awarded a blind 95 without sufficient sample size.
    if total_loc < 250 or total_files <= 2:
        confidence = max(0.35, min(1.0, total_loc / 250.0))
        final_score = round(raw_score * confidence + 70.0 * (1.0 - confidence))
        summary_note = f"Score calibrated for statistical sample size ({total_files} files, {total_loc} LOC)."
    else:
        final_score = round(raw_score)
        summary_note = f"Computed across {len(available_metrics)} of 5 measurable quality dimensions ({total_files} files scanned)."

    final_score = max(10, min(100, final_score))

    if final_score >= 80:
        overall_rating = "Healthy Codebase"
    elif final_score >= 60:
        overall_rating = "Moderate Maintainability"
    elif final_score >= 45:
        overall_rating = "Needs Attention"
    else:
        overall_rating = "High Risk / Technical Debt"

    summary_explanation = (
        f"Absolute Repository Health: {final_score}/100 ({overall_rating}) with {int(metric_coverage * 100)}% dimension coverage. "
        f"{summary_note}"
    )

    coverage_details = {
        "dimension_coverage": metric_coverage,
        "dimension_coverage_percent": int(metric_coverage * 100),
        "files_analyzed": total_files,
        "code_files_analyzed": len(code_files),
        "total_loc_analyzed": total_loc,
        "dimensions_available": [k for k, v in metrics.items() if v["available"]],
        "dimensions_unavailable": [k for k, v in metrics.items() if not v["available"]],
    }

    return {
        "score": final_score,
        "rating": overall_rating,
        "metric_coverage": metric_coverage,
        "coverage_details": coverage_details,
        "metrics": metrics,
        "summary_explanation": summary_explanation
    }


def _compute_percentile_health_scores(file_data_list: list[dict]) -> list[dict]:
    """
    Normalizes churn, size (LOC), and complexity relative to the repository's file distribution
    using percentile ranks.
    
    Preserves relative ranking for identifying files that are riskier compared with other files
    in the same repository.
    
    Returns a list of dicts with calculated relative scores on a 0-100 scale:
    Where 100 = Lowest Risk in repo, and 0 = Highest Risk in repo.
    """
    n = len(file_data_list)
    if n == 0:
        return []

    churns = [f["churn_count"] for f in file_data_list]
    locs = [f["loc"] for f in file_data_list]
    complexities = [f["complexity"] for f in file_data_list]

    def get_health_scores_for_metric(metric_values: list[int]) -> list[int]:
        min_v, max_v = min(metric_values), max(metric_values)
        if min_v == max_v:
            return [100] * n

        health_scores = []
        for v in metric_values:
            less_count = sum(1 for x in metric_values if x < v)
            equal_count = sum(1 for x in metric_values if x == v)
            # Midpoint percentile rank of risk (0 to 100)
            risk_percentile = ((less_count + 0.5 * equal_count) / n) * 100
            # Higher risk percentile means lower health score
            health = round(100.0 - risk_percentile)
            health_scores.append(max(0, min(100, health)))
        return health_scores

    churn_scores = get_health_scores_for_metric(churns)
    size_scores = get_health_scores_for_metric(locs)
    complexity_scores = get_health_scores_for_metric(complexities)

    results = []
    for i, item in enumerate(file_data_list):
        c_score = churn_scores[i]
        s_score = size_scores[i]
        cx_score = complexity_scores[i]
        comp_score = round((c_score + s_score + cx_score) / 3.0)

        risk_level = "healthy" if comp_score >= 70 else "review" if comp_score >= 45 else "risk"

        results.append({
            "file_path": item["file_path"],
            "churn_count": item["churn_count"],
            "loc": item["loc"],
            "complexity": item["complexity"],
            "churn_score": c_score,
            "size_score": s_score,
            "complexity_score": cx_score,
            "composite_score": comp_score,
            "relative_risk_percentile": 100 - comp_score,
            "risk_level": risk_level
        })

    return results


def compute_quality_score(churn_count: int, loc: int, complexity: int, repo_stats: dict = None) -> dict:
    """
    Computes component health scores and composite health score for a single file.
    If repo_stats distributions are provided, computes percentile ranks; otherwise uses default fallback.
    """
    if repo_stats and "churns" in repo_stats and "locs" in repo_stats and "complexities" in repo_stats:
        dummy_list = [
            {"file_path": "target", "churn_count": churn_count, "loc": loc, "complexity": complexity}
        ]
        for c, l, cx in zip(repo_stats["churns"], repo_stats["locs"], repo_stats["complexities"]):
            dummy_list.append({"file_path": "ref", "churn_count": c, "loc": l, "complexity": cx})
            
        calculated = _compute_percentile_health_scores(dummy_list)
        target_res = calculated[0]
        return {
            "churn_score": target_res["churn_score"],
            "size_score": target_res["size_score"],
            "complexity_score": target_res["complexity_score"],
            "composite_score": target_res["composite_score"]
        }
    
    churn_score = max(0, 100 - min(100, churn_count * 5))
    size_score = max(0, 100 - min(100, int(loc / 10)))
    complexity_score = max(0, 100 - min(100, complexity * 2))
    composite_score = round((churn_score + size_score + complexity_score) / 3.0)

    return {
        "churn_score": churn_score,
        "size_score": size_score,
        "complexity_score": complexity_score,
        "composite_score": composite_score
    }


def _get_repo_files_for_analysis(repo_id: str) -> list[dict]:
    raw_files = []

    # 1. First, check in-memory cache (populated immediately after ingest)
    try:
        for (r_id, f_path), content in list(LOCAL_FILE_CONTENTS_CACHE.items()):
            if r_id == repo_id and content:
                raw_files.append({"file_path": f_path, "content": content})
    except Exception as e:
        print(f"[DevLens AI Warning] Failed to inspect LOCAL_FILE_CONTENTS_CACHE: {e}")

    # 2. If memory cache was empty, check local disk cache
    if not raw_files:
        try:
            repo_cache_dir = os.path.join(CACHE_DIR, repo_id)
            if os.path.exists(repo_cache_dir) and os.path.isdir(repo_cache_dir):
                for root, _, files in os.walk(repo_cache_dir):
                    for file_name in files:
                        full_path = os.path.join(root, file_name)
                        rel_path = os.path.relpath(full_path, repo_cache_dir).replace("\\", "/")
                        if rel_path.startswith("summaries"):
                            continue
                        try:
                            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read()
                                if content:
                                    raw_files.append({"file_path": rel_path, "content": content})
                        except Exception:
                            pass
        except Exception as e:
            print(f"[DevLens AI Warning] Failed to inspect disk cache: {e}")

    # 3. If still empty, attempt Supabase file_contents table
    if not raw_files and supabase is not None:
        try:
            files_res = supabase.table("file_contents").select("file_path, content").eq("repo_id", repo_id).execute()
            if files_res.data:
                raw_files = files_res.data
        except Exception as e:
            print(f"[DevLens AI Warning] Could not fetch file_contents from Supabase: {e}")

    # 4. Fallback: retrieve from repo analysis file tree if available
    if not raw_files:
        try:
            analysis = get_repo_analysis(repo_id)
            if analysis and analysis.get("file_tree"):
                from app.ingestion.filter import get_all_project_files
                project_files = get_all_project_files(analysis["file_tree"])
                for f_path in project_files:
                    c = get_file_content(repo_id, f_path)
                    if c:
                        raw_files.append({"file_path": f_path, "content": c})
        except Exception as e:
            print(f"[DevLens AI Warning] Failed fallback workspace file retrieval: {e}")

    return raw_files


def compute_repo_quality_scores(repo_id: str, force_recompute: bool = False) -> dict:
    """
    Computes absolute repository health and relative file quality scores.
    Maintains complete backward compatibility while adding metric-level explainability.
    """
    # 1. Check existing cached quality scores if not force_recompute
    if not force_recompute:
        # A. Memory cache (must have repository_health to be valid DevLens 2.0 cache)
        if repo_id in LOCAL_QUALITY_SCORES_CACHE:
            cached_data = dict(LOCAL_QUALITY_SCORES_CACHE[repo_id])
            if "repository_health" in cached_data:
                cached_data["cached"] = True
                return cached_data

        # B. Disk cache
        try:
            disk_cache_path = os.path.join(CACHE_DIR, f"{repo_id}_quality.json")
            if os.path.exists(disk_cache_path):
                with open(disk_cache_path, "r", encoding="utf-8") as f:
                    disk_data = json.load(f)
                    if "repository_health" in disk_data:
                        LOCAL_QUALITY_SCORES_CACHE[repo_id] = disk_data
                        disk_data["cached"] = True
                        return disk_data
        except Exception as e:
            print(f"[DevLens AI Warning] Could not read disk quality score cache: {e}")

    # 2. Fetch raw file contents resiliently
    raw_files = _get_repo_files_for_analysis(repo_id)

    if not raw_files:
        empty_res = {
            "repo_id": repo_id,
            "file_count": 0,
            "average_composite_score": 100,
            "needs_attention_count": 0,
            "riskiest_files": [],
            "scores": [],
            "repository_health": {
                "score": 100,
                "rating": "Healthy Codebase",
                "metric_coverage": 0.0,
                "coverage_details": {
                    "dimension_coverage": 0.0,
                    "dimension_coverage_percent": 0,
                    "files_analyzed": 0,
                    "code_files_analyzed": 0,
                    "total_loc_analyzed": 0,
                    "dimensions_available": [],
                    "dimensions_unavailable": ["complexity", "file_size", "duplication", "coupling", "churn"]
                },
                "metrics": {},
                "summary_explanation": "No files found to analyze."
            },
            "cached": False
        }
        LOCAL_QUALITY_SCORES_CACHE[repo_id] = empty_res
        return empty_res

    # 3. Fetch repo analysis metadata (hotspots and dependencies)
    analysis_data = {}
    hotspot_churn_map = {}
    dependencies_list = []
    hotspots_list = []

    try:
        analysis_data = get_repo_analysis(repo_id) or {}
        hotspots_list = analysis_data.get("hotspots", []) if analysis_data else []
        dependencies_list = analysis_data.get("dependencies", []) if analysis_data else []
        for h in hotspots_list:
            if isinstance(h, dict) and "path" in h:
                hotspot_churn_map[h["path"]] = h.get("commit_count", 0)
    except Exception as e:
        print(f"[DevLens AI Warning] Could not fetch repo analysis hotspots/dependencies: {e}")

    # 4. Filter files and compute raw LOC and complexity
    file_data_list = []
    for item in raw_files:
        file_path = item.get("file_path", "")
        content = item.get("content", "") or ""

        # Skip excluded files (lockfiles, minified files, files > 200KB)
        if is_excluded_file(file_path, content):
            continue

        language = detect_language(file_path)
        loc = len(content.splitlines())
        complexity = compute_file_complexity(content, language, file_path)
        churn = hotspot_churn_map.get(file_path, 0)

        file_data_list.append({
            "file_path": file_path,
            "content": content,
            "churn_count": churn,
            "loc": loc,
            "complexity": complexity
        })

    if not file_data_list:
        empty_res = {
            "repo_id": repo_id,
            "file_count": 0,
            "average_composite_score": 100,
            "needs_attention_count": 0,
            "riskiest_files": [],
            "scores": [],
            "repository_health": {
                "score": 100,
                "rating": "Healthy Codebase",
                "metric_coverage": 0.0,
                "coverage_details": {
                    "dimension_coverage": 0.0,
                    "dimension_coverage_percent": 0,
                    "files_analyzed": 0,
                    "code_files_analyzed": 0,
                    "total_loc_analyzed": 0,
                    "dimensions_available": [],
                    "dimensions_unavailable": ["complexity", "file_size", "duplication", "coupling", "churn"]
                },
                "metrics": {},
                "summary_explanation": "No supported source files found."
            },
            "cached": False
        }
        LOCAL_QUALITY_SCORES_CACHE[repo_id] = empty_res
        return empty_res

    # 5. Compute Relative File Risk Scores (percentile ranking within repo)
    scored_files = _compute_percentile_health_scores(file_data_list)
    sorted_scores = sorted(scored_files, key=lambda x: x["composite_score"])
    needs_attention = sum(1 for s in scored_files if s["composite_score"] < 60)

    # 6. Compute Absolute Repository Health Score (fixed, explainable engineering thresholds)
    repo_health = compute_absolute_repository_health(
        file_data_list,
        dependencies=dependencies_list if dependencies_list else None,
        hotspots=hotspots_list if hotspots_list else None
    )

    abs_repo_score = repo_health["score"]

    final_result = {
        "repo_id": repo_id,
        "file_count": len(scored_files),
        "average_composite_score": abs_repo_score,
        "needs_attention_count": needs_attention,
        "riskiest_files": sorted_scores[:5],
        "scores": scored_files,
        "repository_health": repo_health,
        "cached": False
    }

    # 7. Save results into local memory cache & disk cache
    LOCAL_QUALITY_SCORES_CACHE[repo_id] = final_result
    try:
        disk_cache_path = os.path.join(CACHE_DIR, f"{repo_id}_quality.json")
        with open(disk_cache_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f)
    except Exception as e:
        print(f"[DevLens AI Warning] Failed to write quality scores to disk cache: {e}")

    # 8. Optionally save to Supabase `quality_scores` table if available
    if supabase is not None:
        records_to_upsert = []
        timestamp = datetime.now(timezone.utc).isoformat()
        for item in scored_files:
            records_to_upsert.append({
                "repo_id": repo_id,
                "file_path": item["file_path"],
                "churn_score": item["churn_score"],
                "size_score": item["size_score"],
                "complexity_score": item["complexity_score"],
                "composite_score": item["composite_score"],
                "computed_at": timestamp
            })

        try:
            chunk_size = 100
            for i in range(0, len(records_to_upsert), chunk_size):
                chunk = records_to_upsert[i:i + chunk_size]
                supabase.table("quality_scores").upsert(chunk).execute()
        except Exception as e:
            print(f"[DevLens AI Warning] Failed to upsert quality_scores to Supabase (cached locally): {e}")

    return final_result
