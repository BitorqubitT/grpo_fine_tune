import subprocess
import shutil
import logging
import time
from pathlib import Path
from uuid import uuid4
from dataclasses import dataclass
from typing import Dict, List, Optional

template_rs_file = """
#![allow(dead_code)]
// {code}

fn main() {
    println!("Hello World");
}
"""

cargo_toml_file ="""
[package]
name = "rust-program"
version = "0.1.0"
edition = "2021"

[dependencies]
"""

rustcode = """
fn sort_list(mut list: Vec<i32>) -> Vec<i32> {
    list.sort();
    list
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_list() {
        let unsorted = vec![5, 3, 8, 1, 2];
        let sorted = sort_list(unsorted.clone());
        assert_eq!(sorted, vec![1, 2, 3, 5, 8]);
    }
}
"""

#TODO: Change logging to wandb
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RustToolResult:
    passed: bool
    stderr: str
    stdout: str
    execution_time: float

class RustTool:
    def __init__(self, name: str):
        self.name = name

    def run(self, project_dir: Path) -> RustToolResult:
        try:
            start_time = time.time()
            
            result = subprocess.run(
                ["cargo", self.name, "--quiet"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            execution_time = time.time() - start_time
            
            return RustToolResult(
                passed=result.returncode == 0,
                stderr=result.stderr,
                stdout=result.stdout,
                execution_time=execution_time
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Tool {self.name} timed out")
            return RustToolResult(False, "Execution timed out", "", 60.0)
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {str(e)}")
            return RustToolResult(False, str(e), "", 0.0)

def setup_rust_project(
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
        template = template_rs.replace("// {code}", rust_code)
        (project_dir_src / "main.rs").write_text(template)
        
        # Write Cargo.toml
        (project_dir / "Cargo.toml").write_text(cargo_toml)
        
        return project_dir
    except Exception as e:
        logger.error(f"Failed to setup project: {str(e)}")
        return None

def run_rust_tests(
    rust_code: str,
    tools: List[RustTool],
    base_dir: Path,
    cargo_toml: str,
    template_rs: str
) -> Dict[str, RustToolResult]:
    """Main function to run Rust code tests"""
    results = {}
    
    project_dir = setup_rust_project(base_dir, cargo_toml, template_rs, rust_code)
    if not project_dir:
        return results

    try:
        for tool in tools:
            logger.info(f"Running {tool.name}")
            result = tool.run(project_dir)
            results[tool.name] = result
            
            if not result.passed:
                logger.warning(f"{tool.name} failed: {result.stderr}")
    finally:
        # Clean up
        if project_dir.exists():
            shutil.rmtree(project_dir)
    return results

# Usage
tools = [RustTool("build"), RustTool("clippy"), RustTool("test")]
base_dir = Path("outputs") / "tests"

rust_results = run_rust_tests(
    rustcode,
    tools,
    base_dir,
    cargo_toml_file,
    template_rs_file
)

def rust_rewards(results):
    rust_tool_reward = {'build': 0, 'clippy': 0, 'test': 0, 'test_time': 0, 'result_output': ''} 

    for tool_name, result in results.items():
        if result.passed:
            rust_tool_reward[tool_name] = 1
        else:
            rust_tool_reward[tool_name] = 0
        if result.stderr:
            rust_tool_reward['result_output'] = result.stderr
        if result.stdout:
            rust_tool_reward['test_time'] = result.execution_time
            rust_tool_reward['result_output'] = result.stdout
    
    return rust_tool_reward

all_rust_rewards = rust_rewards(rust_results)
print(all_rust_rewards)
