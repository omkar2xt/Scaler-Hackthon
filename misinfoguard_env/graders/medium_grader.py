"""
Medium-difficulty grader for MisinfoGuard-Env.
Scores agent performance on moderate-density, balanced-spreader scenarios.
"""

def grade(trajectory: dict) -> float:
    """
    Grade agent performance on medium task.
    
    Args:
        trajectory: Dictionary with keys:
            - episode_rewards: List of step rewards
            - false_reach: List of false information reach values (0-1)
            - recall: List of recall values (0-1)
            - precision: List of precision values (0-1)
    
    Returns:
        float: Score in range [0.0, 1.0]
    """
    if not trajectory:
        return 0.0
    
    # Extract metrics
    episode_rewards = trajectory.get("episode_rewards", [0])
    false_reach = trajectory.get("false_reach", [1.0])
    recall = trajectory.get("recall", [0.0])
    precision = trajectory.get("precision", [0.0])
    
    # Normalize metrics
    final_reward = episode_rewards[-1] if episode_rewards else 0
    final_false_reach = false_reach[-1] if false_reach else 1.0
    final_recall = recall[-1] if recall else 0.0
    final_precision = precision[-1] if precision else 0.0
    
    # Reward component: normalize to [0, 1] range
    reward_score = min(1.0, max(0.0, (final_reward + 150) / 150))
    
    # False reach component: lower is better
    false_reach_score = 1.0 - final_false_reach
    
    # Recall component: higher is better
    recall_score = final_recall
    
    # Precision component: higher is better (medium task requires better precision)
    precision_score = final_precision
    
    # Weighted average (medium task has balanced requirements)
    score = (
        0.25 * reward_score +
        0.30 * false_reach_score +
        0.25 * recall_score +
        0.20 * precision_score
    )
    
    return min(1.0, max(0.0, score))
