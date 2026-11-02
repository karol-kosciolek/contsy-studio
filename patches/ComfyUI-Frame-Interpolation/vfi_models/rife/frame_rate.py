def frame_rate_plan(frame_count, source_fps, target_fps):
    """RIFE timesteps on the output clock, retaining the final source endpoint."""
    if frame_count < 1 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("Frame count and frame rates must be positive")
    count = ((frame_count - 1) * target_fps + source_fps - 1) // source_fps + 1
    tasks, layout, per_pair = [], [], {}
    for index in range(count):
        pair, remainder = divmod(index * source_fps, target_fps)
        if remainder == 0 or pair >= frame_count - 1:
            layout.append((min(pair, frame_count - 1), -1))
        else:
            layout.append((pair, per_pair.get(pair, 0)))
            per_pair[pair] = per_pair.get(pair, 0) + 1
            tasks.append((pair, remainder / target_fps))
    return tasks, layout
