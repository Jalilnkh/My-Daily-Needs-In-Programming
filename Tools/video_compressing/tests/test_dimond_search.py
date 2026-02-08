import unittest
import numpy as np
from utils import diamond_search_block

class TestDiamondSearch(unittest.TestCase):
    def setUp(self):
        # Create two 64x64 gray frames
        self.frame_size = 64
        self.block_size = 16
        self.ref_frame = np.zeros((self.frame_size, self.frame_size), dtype=np.uint8)
        self.curr_frame = np.zeros((self.frame_size, self.frame_size), dtype=np.uint8)

    def test_horizontal_motion(self):
        """Test if DS detects a block moving 2 pixels to the right."""
        # 1. Place a white square in the reference frame at (20, 20)
        self.ref_frame[20:36, 20:36] = 255
        
        # 2. Shift that square in the current frame to (22, 20) -> Motion: dx=2, dy=0
        self.curr_frame[20:36, 22:38] = 255
        
        # 3. Run Diamond Search
        mv = diamond_search_block(self.curr_frame, self.ref_frame, x=22, y=20, b_size=16)
        
        # 4. Assert that it found the shift from the reference (which was at 20, 20)
        # mv = (ref_x - curr_x), so (20 - 22) = -2
        self.assertEqual(mv, (-2, 0))

    def test_static_block(self):
        """Test if DS returns (0,0) when there is no movement."""
        self.ref_frame[20:36, 20:36] = 255
        self.curr_frame[20:36, 20:36] = 255
        
        mv = diamond_search_block(self.curr_frame, self.ref_frame, x=20, y=20, b_size=16)
        self.assertEqual(mv, (0, 0))

if __name__ == '__main__':
    unittest.main()
