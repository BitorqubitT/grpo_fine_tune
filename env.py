import re
from typing import Optional
import subprocess
import shutil
import time
from pathlib import Path
from uuid import uuid4
from dataclasses import dataclass
from typing import Dict, List, Optional

class env():

    def __init__(self, cargo_toml: str, template_rs: str):
        self.cargo_toml = cargo_toml
        self.template_rs = template_rs

    def step(self, answer: List[str]) -> List[Dict[str, float]]:
        all_rewards = []
        
        for code in answer:
            format_results = self._format_rewards(code)

            # HACK: Ugly
            # Build and clippy can pass if code is empty
            if format_results["not empty"] == 0 or format_results["code block"] == 0:
                # Code is empty we can skip further
                run_rust_rewards = {'build': 0, 'clippy': 0, 'test': 0, 'result_output': ''}
            else:
                rust_results = self._run_rust_tests(code, self.cargo_toml, self.template_rs)
                run_rust_rewards = self._rust_rewards(rust_results)
                # Test can pass if no test block is present
                if format_results["test block"] == 0:
                    run_rust_rewards['test'] = 0

            all_rewards.append({**format_results, **run_rust_rewards})

        return all_rewards

    def _format_rewards(self, code) -> Dict[str, float]:
        format_rewards = {"not empty": 0, "code block": 0, "test block": 0, "asserts": 0}
        if self._check_code_not_empty(code):
            format_rewards["not empty"] = 1
        if self._check_code_block(code):
            format_rewards["code block"] = 1
        if self._check_test_block(code):
            format_rewards["test block"] = 1
        format_rewards["asserts"] = self._response_contains_asserts(code)
        return format_rewards

    def _rust_rewards(self, results):
        rust_tool_reward = {'build': 0, 'clippy': 0, 'test': 0, 'result_output': ''} 
        for tool_name, result in results.items():
            if result.passed:
                rust_tool_reward[tool_name] = 1
            else:
                rust_tool_reward[tool_name] = 0
            if result.stderr:
                rust_tool_reward['result_output'] = result.stderr
            if result.stdout:
                rust_tool_reward['result_output'] = result.stdout
        
        return rust_tool_reward

    def _setup_rust_project(
        self,
        base_dir: Path,
        cargo_toml: str,
        template_rs: str,
        rust_code: str
    ) -> Optional[Path]:
        """Sets up a temporary Rust project with the given code"""
        try:
            project_dir = base_dir / f"temp_rust_project_{uuid4()}"
            project_dir_src = project_dir / "src"
            project_dir_src.mkdir(parents=True, exist_ok=True)

            # Write source code
            rust_code = self._extract_rust_code(rust_code)
            template = template_rs.replace("// {code}", rust_code)
            (project_dir_src / "main.rs").write_text(template)
            
            # Write Cargo.toml
            (project_dir / "Cargo.toml").write_text(cargo_toml)
            
            return project_dir
        except Exception as e:
            #logger.error(f"Failed to setup project: {str(e)}")
            return None

    def _run_rust_tests(
        self,
        rust_code: str,
        cargo_toml: str,
        template_rs: str
    ):
        """Main function to run Rust code tests"""
        results = {}
        tools = [RustTool("build"), RustTool("clippy"), RustTool("test")]
        shutil.rmtree(Path("outputs") / "tests")
        base_dir = Path("outputs") / "tests"

        project_dir = self._setup_rust_project(base_dir, cargo_toml, template_rs, rust_code)
        if not project_dir:
            return results

        try:
            for tool in tools:
                result = tool.run(project_dir)
                results[tool.name] = result
                #if not result.passed:
                    #logger.warning(f"{tool.name} failed: {result.stderr}")
        finally:
            if project_dir.exists():
                shutil.rmtree(project_dir)
        return results
    
    def _extract_rust_code(self, text: str) -> Optional[str]:
        pattern = r'```rust\n(.*?)\n```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return None

    def _check_code_not_empty(self, code: str) -> bool:
        if len(code) > 10:
            return True
        return False

    def _check_code_block(self, code: str) -> bool:
        if self._extract_rust_code(code):
            return True
        return False

    def _check_test_block(self, code: str) -> bool:
        pattern = r'(#\[cfg\(test\)\]\s*mod\s+tests\s*\{.*?\})'
        match = re.search(pattern, code, re.DOTALL)
        if match:
            return True
        return False

    def _response_contains_asserts(self, code: str) -> float:
        pattern = r'#\[cfg\(test\)\]\s*mod\s+tests\s*\{([^}]*)\}'
        match = re.search(pattern, code, re.DOTALL)

        if not match:
            return 0.0
        
        test_block = match.group(0)

        # Find all assert statements
        assert_pattern = r'assert(?:_eq)?\!(.*?);'
        all_asserts = re.findall(assert_pattern, test_block)
        total_asserts = len(all_asserts)
        
        if total_asserts == 0:
            return 0.0
            
        # Store unique assert statements
        unique_asserts = set(assert_stmt.strip() for assert_stmt in all_asserts)

        #TODO: maybe reduce this or just return true/false 
        return len(unique_asserts) / total_asserts

@dataclass
class RustToolResult:
    passed: bool
    stderr: str
    stdout: str

class RustTool:
    def __init__(self, name: str):
        self.name = name

    def run(self, project_dir: Path) -> RustToolResult:
        try:
            result = subprocess.run(
                ["cargo", self.name, "--quiet"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return RustToolResult(
                passed=result.returncode == 0,
                stderr=result.stderr,
                stdout=result.stdout,
            )
        except Exception as e:
            return RustToolResult(False, str(e), "")