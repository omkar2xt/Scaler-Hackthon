"""
Pre-submission validation script for MisinfoGuard-Env.
Performs 8 mandatory checks before final submission.
"""

import sys
import os
import re
import json
from pathlib import Path

def check_1_openenv_yaml():
    """Check 1: openenv.yaml exists and contains required fields."""
    try:
        import yaml
        path = Path("openenv.yaml")
        if not path.exists():
            return False, "openenv.yaml not found"
        
        with open(path) as f:
            content = yaml.safe_load(f)
            required_fields = ["name", "version", "interface", "tasks", "environment_variables"]
            missing = [f for f in required_fields if f not in content]
            if missing:
                return False, f"Missing fields: {missing}"
        return True, "openenv.yaml valid"
    except Exception as e:
        return False, f"Error: {e}"


def check_2_models_importable():
    """Check 2: models.py exists and all models are importable."""
    try:
        from misinfoguard_env.models import (
            PostFeature, MisinfoObservation, MisinfoAction,
            MisinfoReward, StepResult, EnvState
        )
        return True, "All models importable"
    except Exception as e:
        return False, f"Import failed: {e}"


def check_3_environment_api():
    """Check 3: Environment has reset, step, and get_state methods."""
    try:
        from misinfoguard_env.environment import MisinfoGuardEnv
        env = MisinfoGuardEnv()
        required_methods = ["reset", "step", "get_state", "close"]
        missing = [m for m in required_methods if not hasattr(env, m)]
        if missing:
            return False, f"Missing methods: {missing}"
        env.close()
        return True, "Environment API valid"
    except Exception as e:
        return False, f"Error: {e}"


def check_4_grader_variability():
    """Check 4: Graders exist and return different scores."""
    try:
        from misinfoguard_env.graders.easy_grader import grade as easy_grade
        from misinfoguard_env.graders.medium_grader import grade as medium_grade
        from misinfoguard_env.graders.hard_grader import grade as hard_grade
        
        # Create dummy trajectory
        trajectory = {
            "episode_rewards": [-50.0, -40.0, -30.0],
            "false_reach": [0.8, 0.5, 0.3],
            "recall": [0.3, 0.5, 0.8],
            "precision": [0.9, 0.85, 0.8]
        }
        
        easy_score = easy_grade(trajectory)
        medium_score = medium_grade(trajectory)
        hard_score = hard_grade(trajectory)
        
        # Tolerance: allow up to 0.001 variance (anything lower suggests hardcoding)
        scores = [easy_score, medium_score, hard_score]
        variance = max(scores) - min(scores)
        
        if variance < 0.001:
            return False, f"Graders appear hardcoded (variance={variance:.6f})"
        
        return True, f"Grader variability OK (scores: easy={easy_score:.3f}, medium={medium_score:.3f}, hard={hard_score:.3f})"
    except Exception as e:
        return False, f"Error: {e}"


def check_5_environment_variables():
    """Check 5: Required environment variables are documented."""
    try:
        env_file = Path(".env.example")
        if not env_file.exists():
            return False, ".env.example not found"
        
        with open(env_file) as f:
            content = f.read()
        
        required_vars = ["OPENAI_API_KEY", "API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
        missing = [v for v in required_vars if v not in content]
        if missing:
            return False, f"Missing env variables: {missing}"
        
        return True, ".env.example valid"
    except Exception as e:
        return False, f"Error: {e}"


def check_6_dockerfile():
    """Check 6: Dockerfile exists and has required components."""
    try:
        path = Path("Dockerfile")
        if not path.exists():
            return False, "Dockerfile not found"
        
        with open(path) as f:
            content = f.read()
        
        # Check for Python 3.11 base
        if "python:3.11" not in content:
            return False, "Dockerfile must use python:3.11 or later"
        
        # Check for uvicorn setup
        if "uvicorn" not in content:
            return False, "Dockerfile must mention uvicorn"
        
        # Check for healthcheck or inference reference
        if "inference.py" not in content and "HEALTHCHECK" not in content:
            return False, "Dockerfile should reference inference.py or have HEALTHCHECK"
        
        return True, "Dockerfile valid"
    except Exception as e:
        return False, f"Error: {e}"


def check_7_inference_structured_logging():
    """Check 7: inference.py has structured JSON logging (START/STEP/END markers)."""
    try:
        path = Path("misinfoguard_env/inference.py")
        if not path.exists():
            return False, "inference.py not found"
        
        with open(path) as f:
            content = f.read()
        
        # Check for structured logging markers
        required_markers = ["START", "STEP", "END"]
        missing = [m for m in required_markers if m not in content]
        if missing:
            return False, f"Missing logging markers: {missing}"
        
        if "json" not in content.lower():
            return False, "inference.py should use JSON structured logging"
        
        return True, "Structured logging present"
    except Exception as e:
        return False, f"Error: {e}"


def check_8_readme():
    """Check 8: README.md exists at root level with baseline information."""
    try:
        path = Path("README.md")
        if not path.exists():
            return False, "README.md not found at root"
        
        with open(path) as f:
            content = f.read()
        
        # Check for key sections
        required_sections = ["setup", "train", "eval", "baseline"]
        lowercase_content = content.lower()
        missing = [s for s in required_sections if s not in lowercase_content]
        if missing:
            return False, f"Missing sections in README: {missing}"
        
        return True, "README.md valid"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Run all checks and report results."""
    checks = [
        ("openenv.yaml valid", check_1_openenv_yaml),
        ("Models importable", check_2_models_importable),
        ("Environment API complete", check_3_environment_api),
        ("Grader variability", check_4_grader_variability),
        ("Environment variables documented", check_5_environment_variables),
        ("Dockerfile valid", check_6_dockerfile),
        ("Structured logging in inference.py", check_7_inference_structured_logging),
        ("README.md baseline", check_8_readme),
    ]
    
    print("\n" + "="*60)
    print("MisinfoGuard-Env Pre-Submission Validation")
    print("="*60 + "\n")
    
    results = []
    passed = 0
    failed = 0
    
    for check_name, check_func in checks:
        try:
            success, message = check_func()
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status}: {check_name}")
            print(f"       {message}\n")
            
            if success:
                passed += 1
            else:
                failed += 1
            
            results.append({
                "check": check_name,
                "status": success,
                "message": message
            })
        except Exception as e:
            print(f"❌ ERROR: {check_name}")
            print(f"       {e}\n")
            failed += 1
            results.append({
                "check": check_name,
                "status": False,
                "message": str(e)
            })
    
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    if failed > 0:
        print("⚠️  SUBMISSION BLOCKED: Some checks failed")
        return 1
    else:
        print("✅ All checks passed! Ready for submission.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
