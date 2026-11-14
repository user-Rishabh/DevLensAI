import unittest
from app.analysis.quality_score import (
    compute_absolute_repository_health,
    _compute_percentile_health_scores,
    compute_file_complexity,
    compute_duplication_ratio,
    strip_comments_and_strings,
    WEIGHT_COMPLEXITY,
    WEIGHT_FILE_SIZE,
    WEIGHT_DUPLICATION,
    WEIGHT_COUPLING,
    WEIGHT_CHURN,
)

class TestQualityScores(unittest.TestCase):

    def test_01_small_clean_repository(self):
        """Test a small, clean repository with moderate LOC, low complexity, and clean structure."""
        files = [
            {
                "file_path": "math_utils.py",
                "content": (
                    "def add(a: int, b: int) -> int:\n"
                    "    return a + b\n\n"
                    "def subtract(a: int, b: int) -> int:\n"
                    "    return a - b\n"
                ),
                "loc": 6,
                "complexity": 0,
                "churn_count": 1
            },
            {
                "file_path": "string_utils.py",
                "content": (
                    "def capitalize_word(word: str) -> str:\n"
                    "    if not word:\n"
                    "        return ''\n"
                    "    return word[0].upper() + word[1:]\n"
                ),
                "loc": 5,
                "complexity": 1,
                "churn_count": 1
            }
        ]
        res = compute_absolute_repository_health(files, dependencies=[], hotspots=[])
        self.assertIn("score", res)
        self.assertIn("metric_coverage", res)
        self.assertGreater(res["score"], 65)
        self.assertIn("complexity", res["metrics"])
        self.assertEqual(res["metrics"]["complexity"]["rating"], "Low Complexity")

    def test_02_large_clean_repository(self):
        """Test a multi-file, larger clean repository with low complexity density, modular files, and no duplication."""
        files = []
        for i in range(15):
            content = f"// Module {i}\n" + "\n".join([f"export const value_{j} = {j};" for j in range(40)])
            files.append({
                "file_path": f"src/module_{i}.ts",
                "content": content,
                "loc": 42,
                "complexity": 0,
                "churn_count": 2
            })
        deps = [{"from": f"src/module_{i}.ts", "to": f"src/module_{i+1}.ts"} for i in range(14)]
        hotspots = [{"path": f"src/module_{i}.ts", "commit_count": 2} for i in range(15)]

        res = compute_absolute_repository_health(files, dependencies=deps, hotspots=hotspots)
        self.assertGreaterEqual(res["score"], 85)
        self.assertEqual(res["rating"], "Healthy Codebase")
        self.assertGreaterEqual(res["metric_coverage"], 0.95)

    def test_03_high_complexity_repository(self):
        """Test a repository with excessive branching, nested loops, and conditionals."""
        spaghetti_code = "def complex_engine(x, y, z):\n"
        for i in range(50):
            spaghetti_code += f"    if x > {i} and y < {i} or z == {i}:\n        x += 1\n    elif x == {i} and z:\n        y -= 1\n"
        
        files = [
            {
                "file_path": "engine.py",
                "content": spaghetti_code,
                "loc": 200,
                "complexity": 150,
                "churn_count": 5
            }
        ]
        res = compute_absolute_repository_health(files)
        cx_metric = res["metrics"]["complexity"]
        self.assertTrue(cx_metric["available"])
        self.assertLess(cx_metric["score"], 45)
        self.assertEqual(cx_metric["rating"], "High Complexity")

    def test_04_extensive_duplication_repository(self):
        """Test a repository with significant copy-paste duplicated code blocks."""
        repeated_block = (
            "data = fetch_records(database)\n"
            "filtered = [d for d in data if d.is_active]\n"
            "sorted_data = sorted(filtered, key=lambda x: x.created_at)\n"
            "summary = aggregate_totals(sorted_data)\n"
            "notify_observers(summary)\n"
        )
        files = [
            {
                "file_path": f"services/service_{i}.py",
                "content": f"# Service {i}\n" + (repeated_block * 5),
                "loc": 30,
                "complexity": 2,
                "churn_count": 1
            }
            for i in range(4)
        ]
        res = compute_absolute_repository_health(files)
        dup_metric = res["metrics"]["duplication"]
        self.assertTrue(dup_metric["available"])
        self.assertLess(dup_metric["score"], 65)
        self.assertEqual(dup_metric["rating"], "High Duplication")

    def test_05_high_dependency_coupling_repository(self):
        """Test a repository with high coupling ratio and god-module fan-in."""
        files = [{"file_path": f"file_{i}.ts", "loc": 50, "complexity": 2, "churn_count": 1} for i in range(10)]
        deps = []
        for i in range(1, 10):
            deps.append({"from": f"file_{i}.ts", "to": "file_0.ts"})
            for j in range(1, 6):
                if i != j:
                    deps.append({"from": f"file_{i}.ts", "to": f"file_{j}.ts"})

        res = compute_absolute_repository_health(files, dependencies=deps)
        coupling_metric = res["metrics"]["coupling"]
        self.assertTrue(coupling_metric["available"])
        self.assertLess(coupling_metric["score"], 60)
        self.assertEqual(coupling_metric["rating"], "Tightly Coupled")

    def test_06_high_churn_hotspots_repository(self):
        """Test a repository where 90% of commit churn is concentrated in an oversized/complex file."""
        files = [
            {"file_path": "monster_legacy.py", "loc": 900, "complexity": 80, "churn_count": 95},
            {"file_path": "helper1.py", "loc": 30, "complexity": 1, "churn_count": 2},
            {"file_path": "helper2.py", "loc": 40, "complexity": 1, "churn_count": 3},
        ]
        hotspots = [
            {"path": "monster_legacy.py", "commit_count": 95},
            {"path": "helper1.py", "commit_count": 2},
            {"path": "helper2.py", "commit_count": 3},
        ]
        res = compute_absolute_repository_health(files, hotspots=hotspots)
        churn_metric = res["metrics"]["churn"]
        self.assertTrue(churn_metric["available"])
        self.assertLess(churn_metric["score"], 50)
        self.assertEqual(churn_metric["rating"], "High Churn Risk")

    def test_07_missing_git_history(self):
        """Test graceful degradation when git history is absent."""
        files = [
            {"file_path": "app.py", "content": "print('hello')", "loc": 10, "complexity": 0, "churn_count": 0}
        ]
        res = compute_absolute_repository_health(files, hotspots=None)
        self.assertFalse(res["metrics"]["churn"]["available"])
        self.assertEqual(res["metrics"]["churn"]["rating"], "Not Measured")
        self.assertLess(res["metric_coverage"], 1.0)

    def test_08_unsupported_languages(self):
        """Test that plain text or unsupported formats do not fake complexity measurements."""
        files = [
            {"file_path": "notes.txt", "content": "Meeting notes\nTasks to do\nItem 3\n", "loc": 3, "complexity": 0, "churn_count": 0},
            {"file_path": "config.json", "content": '{\n  "version": "1.0"\n}\n', "loc": 3, "complexity": 0, "churn_count": 0}
        ]
        res = compute_absolute_repository_health(files)
        self.assertFalse(res["metrics"]["complexity"]["available"])
        self.assertEqual(res["metrics"]["complexity"]["rating"], "Not Measured")

    def test_09_incomplete_metric_coverage_and_differentiation(self):
        """Verify that repositories with materially different quality receive materially different scores."""
        clean_files = [
            {"file_path": f"src/comp_{i}.py", "content": f"def func_{i}():\n    return {i}\n", "loc": 40, "complexity": 1, "churn_count": 1}
            for i in range(8)
        ]
        clean_res = compute_absolute_repository_health(clean_files)

        dirty_files = [
            {"file_path": f"src/bad_{i}.py", "content": "if x:\n" * 40, "loc": 800, "complexity": 90, "churn_count": 50}
            for i in range(8)
        ]
        dirty_res = compute_absolute_repository_health(dirty_files)

        self.assertGreater(clean_res["score"], 75)
        self.assertLess(dirty_res["score"], 50)
        self.assertGreater(clean_res["score"] - dirty_res["score"], 25)

    def test_10_relative_file_ranking_independence(self):
        """Verify that relative file ranking continues to rank files 0-100 within a repo independently."""
        files = [
            {"file_path": "safest.py", "loc": 10, "complexity": 0, "churn_count": 1},
            {"file_path": "moderate.py", "loc": 100, "complexity": 10, "churn_count": 5},
            {"file_path": "riskiest.py", "loc": 600, "complexity": 50, "churn_count": 25},
        ]
        ranked = _compute_percentile_health_scores(files)
        self.assertEqual(len(ranked), 3)

        safest = next(f for f in ranked if f["file_path"] == "safest.py")
        riskiest = next(f for f in ranked if f["file_path"] == "riskiest.py")

        self.assertGreater(safest["composite_score"], riskiest["composite_score"])
        self.assertEqual(safest["risk_level"], "healthy")
        self.assertEqual(riskiest["risk_level"], "risk")

    def test_11_audit_score_exact_calculation(self):
        """
        Validates the exact weights and arithmetic that produced overall score 51:
        Complexity: 13 * 0.25 = 3.25
        Modularity: 46 * 0.20 = 9.20
        Duplication: 70 * 0.15 = 10.50
        Coupling: 98 * 0.20 = 19.60
        Churn: 42 * 0.20 = 8.40
        Sum = 50.95 -> round(50.95) = 51
        """
        weights = [WEIGHT_COMPLEXITY, WEIGHT_FILE_SIZE, WEIGHT_DUPLICATION, WEIGHT_COUPLING, WEIGHT_CHURN]
        self.assertEqual(sum(weights), 1.0)

        scores = [13, 46, 70, 98, 42]
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        self.assertAlmostEqual(weighted_sum, 50.95, places=2)
        self.assertEqual(round(weighted_sum), 51)

    def test_12_strip_comments_and_strings_cyclomatic_accuracy(self):
        """Verifies that SQL strings and comments with 'if/and/or' do not inflate cyclomatic complexity."""
        content = (
            "# This function checks if user is authenticated or verified\n"
            "def query_database():\n"
            "    sql = 'SELECT * FROM users WHERE active = true AND age > 21 OR role = admin'\n"
            "    if is_ready:\n"
            "        return True\n"
            "    return False\n"
        )
        # Without comment/string stripping, this would match 'if', 'or', 'AND', 'OR', 'if' = 5 branches!
        clean = strip_comments_and_strings(content, "db.py")
        self.assertNotIn("SELECT", clean)
        self.assertNotIn("authenticated", clean)

        cx = compute_file_complexity(content, "python", "db.py")
        # Only the actual 'if is_ready' control flow branch should be counted
        self.assertEqual(cx, 1)

    def test_13_large_file_with_simple_declarations(self):
        """Verifies that a large file with zero branching is moderated compared to a complex file."""
        declarative_lines = "\n".join([f"export const CONST_{i} = '{i}';" for i in range(400)])
        complex_lines = "\n".join([f"if (x === {i}) {{ y += {i}; }}" for i in range(400)])

        files_decl = [{"file_path": "constants.ts", "loc": 400, "complexity": 0, "churn_count": 0}]
        files_complex = [{"file_path": "engine.ts", "loc": 400, "complexity": 400, "churn_count": 0}]

        res_decl = compute_absolute_repository_health(files_decl)
        res_complex = compute_absolute_repository_health(files_complex)

        # Declarative file size penalty is moderated
        self.assertGreater(res_decl["metrics"]["file_size"]["score"], res_complex["metrics"]["file_size"]["score"])

    def test_14_coupling_self_loops_and_duplicates_deduplicated(self):
        """Verifies that duplicate dependency edges and self-loops are filtered out."""
        files = [
            {"file_path": "a.ts", "loc": 20, "complexity": 0, "churn_count": 0},
            {"file_path": "b.ts", "loc": 20, "complexity": 0, "churn_count": 0},
        ]
        # Include self-loop (a -> a) and duplicate edge (a -> b twice)
        deps = [
            {"from": "a.ts", "to": "a.ts"},
            {"from": "a.ts", "to": "b.ts"},
            {"from": "a.ts", "to": "b.ts"},
        ]
        res = compute_absolute_repository_health(files, dependencies=deps)
        # Only 1 unique edge (a.ts -> b.ts) should be counted across 2 files -> 0.5 edges/file
        coupling = res["metrics"]["coupling"]
        self.assertTrue(coupling["available"])
        self.assertIn("1 edges", coupling["measurement"])

    def test_15_tie_breaking_in_relative_file_risk(self):
        """Verifies symmetric midpoint handling when multiple files tie on metrics."""
        files = [
            {"file_path": "file1.py", "loc": 50, "complexity": 2, "churn_count": 0},
            {"file_path": "file2.py", "loc": 50, "complexity": 2, "churn_count": 0},
            {"file_path": "file3.py", "loc": 50, "complexity": 2, "churn_count": 0},
        ]
        ranked = _compute_percentile_health_scores(files)
        # All three tied files must receive identical composite scores
        scores = [f["composite_score"] for f in ranked]
        self.assertEqual(scores[0], scores[1])
        self.assertEqual(scores[1], scores[2])

if __name__ == "__main__":
    unittest.main()
