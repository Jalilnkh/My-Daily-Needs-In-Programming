# The code is written by Jalil Nourmohammadi Khiarak and all copy rights is reserved.

import cv2
import numpy as np

def get_sad(block1, block2):
    """Sum of Absolute Differences: Measures how different two blocks are."""
    return np.sum(np.abs(block1.astype(np.int32) - block2.astype(np.int32)))

def diamond_search_block(curr_frame, ref_frame, x, y, b_size=16):
    """Finds the best motion vector for a single block using LDSP and SDSP."""
    h, w = curr_frame.shape
    curr_block = curr_frame[y:y+b_size, x:x+b_size]
    
    # Patterns: Large (9 points) and Small (5 points)
    ldsp = [(0,0), (0,-2), (1,-1), (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1)]
    sdsp = [(0,0), (0,-1), (1,0), (0,1), (-1,0)]
    
    cx, cy = x, y # Start search at the current block position
    
    # Step 1: Large Diamond Search (LDSP) - Coarse Move
    while True:
        best_sad = float('inf')
        best_pos = (0, 0)
        
        for dx, dy in ldsp:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx <= w - b_size and 0 <= ny <= h - b_size:
                sad = get_sad(curr_block, ref_frame[ny:ny+b_size, nx:nx+b_size])
                if sad < best_sad:
                    best_sad, best_pos = sad, (dx, dy)
        
        if best_pos == (0, 0): break # Found local best, move to refinement
        cx, cy = cx + best_pos[0], cy + best_pos[1]

    # Step 2: Small Diamond Search (SDSP) - Fine Tuning
    final_sad = float('inf')
    mv = (0, 0)
    for dx, dy in sdsp:
        nx, ny = cx + dx, cy + dy
        if 0 <= nx <= w - b_size and 0 <= ny <= h - b_size:
            sad = get_sad(curr_block, ref_frame[ny:ny+b_size, nx:nx+b_size])
            if sad < final_sad:
                final_sad, mv = sad, (nx - x, ny - y)
    return mv

def compress_frame_ds(current, reference, block_size=16):
    """Processes an entire frame to find all motion vectors and the residual."""
    h, w = current.shape
    residual = np.zeros_like(current, dtype=np.int16)
    m_vectors = []

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            # 1. Get Motion Vector
            dx, dy = diamond_search_block(current, reference, x, y, block_size)
            m_vectors.append(((x, y), (dx, dy)))
            
            # 2. Calculate Residual (The difference to be stored)
            ref_block = reference[y+dy : y+dy+block_size, x+dx : x+dx+block_size]
            curr_block = current[y:y+block_size, x:x+block_size]
            residual[y:y+block_size, x:x+block_size] = curr_block.astype(np.int16) - ref_block.astype(np.int16)
            
    return m_vectors, residual
